import os
import re
import uuid
import logging
import threading
import unicodedata
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from db.mongo_client import connect_to_mongo, get_database
from core.pdf_reader import PDFReader
from core.docx_reader import DocxReader
from core.structure_detector import StructureDetector
from core.semantic_chunker import SemanticChunker
from core.chunk_validator import ChunkValidator
from core.embedder import Embedder
from core.mongo_writer import MongoWriter
from core.gemini_corrector import GeminiCorrector
from core.gemini_analyzer import GeminiAnalyzer
from models.chunk_schema import ChunkSchema, ChunkPosition, ChunkMetadata
from providers.gemini_provider import gemini_provider
from google.genai import types
from schemas.suggestion_schema import SuggestedQuestionsOut
from config.prompt_template import SUGGESTED_QUESTIONS_PROMPT_TEMPLATE
import json
logger = logging.getLogger("rag-service.services.ingest-service")
logging.basicConfig(level=logging.INFO)


def remove_vietnamese_accents(text: str) -> str:
    """Helper to convert Vietnamese text to clean lowercase ASCII-like text for FTS."""
    if not text:
        return ""
    normalized = unicodedata.normalize('NFD', text)
    stripped = "".join([c for c in normalized if not unicodedata.combining(c)])
    stripped = stripped.replace('Đ', 'D').replace('đ', 'd')
    return re.sub(r'\s+', ' ', stripped).strip().lower()


def parse_filename_metadata(pdf_path: str) -> dict:
    """Extract grade level (lop) and semester (hoc_ki) from textbook filename."""
    filename = os.path.basename(pdf_path).lower()
    
    # Extract grade (e.g. "ngu-van-12" or "lop-12")
    grade_match = re.search(r'(?:ngu-van-|lop-)(\d+)', filename)
    lop = int(grade_match.group(1)) if grade_match else 12

    # Extract semester (e.g. "tap-2" -> hoc_ki=2)
    semester_match = re.search(r'(?:tap-|hk)(\d+)', filename)
    hoc_ki = int(semester_match.group(1)) if semester_match else 1

    return {"lop": lop, "hoc_ki": hoc_ki}


class IngestService:
    """
    IngestService orchestrates the ingestion pipeline:
    calls PDFReader -> StructureDetector -> SemanticChunker -> ChunkValidator -> Embedder -> MongoWriter.
    Runs asynchronously in a background thread and tracks status in MongoDB.
    """

    def __init__(self, db_name: Optional[str] = None) -> None:
        """
        Verify database connection and initialize job tracking collection.
        """
        connect_to_mongo()
        self.db = get_database()
        self.jobs_collection = self.db["ingestion_jobs"]
        
        # Load known authors configuration
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(current_dir, "..", "config", "ingest_service_config.json"))
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.known_authors = config.get("known_authors", {})
            self.work_to_author = config.get("work_to_author", {})
            self.excluded_titles = config.get("excluded_titles", ["SÁCH GIÁO KHOA", "ĐỌC THÊM", "GIỚI THIỆU", "TỔNG KẾT", "MỞ ĐẦU", "LỜI NÓI ĐẦU"])
        self.page_offsets = config.get("page_offsets", {})
        
        logger.info("IngestService initialized successfully.")

    def start_ingestion(self, pdf_path_or_dir: str, use_llm_corrector: bool = False) -> str:
        """
        Start the ingestion process asynchronously.
        Returns the job_id immediately.
        """
        # Resolve target files
        pdf_files = []
        if os.path.isdir(pdf_path_or_dir):
            for filename in os.listdir(pdf_path_or_dir):
                if filename.endswith(".pdf") or filename.endswith(".docx"):
                    pdf_files.append(os.path.normpath(os.path.join(pdf_path_or_dir, filename)))
        elif os.path.isfile(pdf_path_or_dir):
            pdf_files.append(os.path.normpath(pdf_path_or_dir))
        else:
            raise FileNotFoundError(f"Path not found: {pdf_path_or_dir}")

        if not pdf_files:
            raise ValueError(f"No PDF/Docx files found in: {pdf_path_or_dir}")

        # Generate unique job ID
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        # Save pending job status to DB
        job_doc = {
            "job_id": job_id,
            "pdf_path": pdf_path_or_dir,
            "status": "pending",
            "total_files": len(pdf_files),
            "processed_files": 0,
            "errors": [],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        self.jobs_collection.insert_one(job_doc)
        logger.info(f"Ingestion job '{job_id}' created at pending state. Total files: {len(pdf_files)}")

        # Launch background execution in a separate daemon thread
        thread = threading.Thread(
            target=self._run_ingestion_sync,
            args=(job_id, pdf_files, use_llm_corrector)
        )
        thread.daemon = True
        thread.start()

        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve current job status from MongoDB.
        """
        job = self.jobs_collection.find_one({"job_id": job_id}, {"_id": 0})
        return job

    def _clean_title_and_author(self, title: str) -> tuple[str, str]:
        if not title:
            return "", "Bộ Giáo Dục và Đào Tạo"
            
        upper_title = title.upper()
        
        # 1. Try to extract explicit author from title (e.g. "TITLE - AUTHOR")
        for raw_auth, clean_auth in self.known_authors.items():
            pattern = rf"\s*[\s_\-\—\–:]+\s*{re.escape(raw_auth)}\s*$"
            pattern_space = rf"\s+{re.escape(raw_auth)}\s*$"
            
            if re.search(pattern, upper_title):
                cleaned_title = re.sub(pattern, "", title, flags=re.IGNORECASE).strip()
                return cleaned_title.upper(), clean_auth
            elif re.search(pattern_space, upper_title):
                cleaned_title = re.sub(pattern_space, "", title, flags=re.IGNORECASE).strip()
                return cleaned_title.upper(), clean_auth
                
        # 2. If no explicit author suffix, lookup the cleaned title in the work_to_author map
        # Strip parens and normalize to match dictionary keys
        title_no_parens = re.sub(r'\(.*?\)', '', title).strip().lower()
        if title_no_parens in self.work_to_author:
            return title.upper(), self.work_to_author[title_no_parens]
            
        # 3. Fallback to default
        return title.upper(), "Bộ Giáo Dục và Đào Tạo"

    def _generate_and_save_suggestions(self, work_title: str, author: str, grade: int, semester: int, chunk_texts: list[str]) -> None:
        """
        Generates 3 suggested questions for a unique work and saves to works_metadata.
        """
        try:
            client = gemini_provider.get_client()
            if not client:
                logger.warning("Không có GEMINI_API_KEY. Bỏ qua việc tự động sinh câu hỏi gợi ý.")
                return
            
            sample_text = "\n\n".join(chunk_texts)[:2000]
            
            prompt = SUGGESTED_QUESTIONS_PROMPT_TEMPLATE.format(
                title=work_title,
                author=author,
                grade=grade,
                semester=semester,
                sample_text=sample_text
            )
            
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                
            resp = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SuggestedQuestionsOut,
                    temperature=0.7
                )
            )
            
            data = json.loads(resp.text)
            questions = data.get("questions", [])
            if not questions:
                logger.warning(f"Gemini trả về danh sách câu hỏi rỗng cho tác phẩm '{work_title}'")
                return
                
            works_col = self.db["works_metadata"]
            works_col.update_one(
                {"work_title": work_title.upper()},
                {
                    "$set": {
                        "work_title": work_title.upper(),
                        "author_name": author,
                        "grade": grade,
                        "semester": semester,
                        "suggested_questions": questions,
                        "updated_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )
            logger.info(f"Đã sinh và lưu 3 câu hỏi gợi ý cho tác phẩm: {work_title.upper()}")
        except Exception as e:
            logger.error(f"Lỗi khi sinh câu hỏi gợi ý cho '{work_title}': {e}")


    def _run_ingestion_sync(self, job_id: str, pdf_files: List[str], use_llm_corrector: bool = False) -> None:
        """
        Synchronous background runner function executed in a separate thread.
        """
        logger.info(f"Starting background execution for job '{job_id}'...")
        
        # Update state to running
        self.jobs_collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "running",
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )

        processed_count = 0
        errors_list = []

        try:
            # Initialize components once
            pdf_reader = PDFReader()
            docx_reader = DocxReader()
            # corrector disabled per user request
            corrector = None
            analyzer = GeminiAnalyzer()
            detector = StructureDetector()
            chunker = SemanticChunker()
            validator = ChunkValidator()
            embedder = Embedder()
            
            # Fetch mongo client config
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/rag_db")
            writer = MongoWriter(mongo_uri=mongodb_uri)

            for pdf_path in pdf_files:
                filename = os.path.basename(pdf_path)
                logger.info(f"[{job_id}] Processing file: {filename}")
                try:
                    # 1. Read Document
                    if filename.lower().endswith(".docx"):
                        elements = docx_reader.read(pdf_path)
                    else:
                        elements = pdf_reader.read(pdf_path)
                        
                    if not elements:
                        raise ValueError(
                            f"Tệp '{filename}' không chứa văn bản dạng số hoặc không thể trích xuất."
                        )
                        
                    # 1.5 Gemini Corrector (Always check everything if enabled)
                    if corrector:
                        elements = corrector.correct(elements, force_all=True)
                        
                    # Truncate elements for fast testing if DEBUG_MAX_BATCHES is set
                    max_batches = int(os.getenv("DEBUG_MAX_BATCHES", "0"))
                    if max_batches > 0:
                        batch_size = int(os.getenv("GEMINI_CORRECTOR_BATCH_SIZE", "30"))
                        limit = max_batches * batch_size
                        if len(elements) > limit:
                            logger.info(f"DEBUG MODE: Cắt giảm dữ liệu từ {len(elements)} xuống còn {limit} elements để test cực nhanh.")
                            elements = elements[:limit]
                        
                    # 2. Detect Structure
                    logger.info("Bắt đầu Detect Structure...")
                    sections = detector.detect(elements)
                    logger.info(f"Hoàn thành Detect Structure. Nhận diện được {len(sections)} sections.")
                    
                    # 3. Chunking
                    logger.info("Bắt đầu Semantic Chunker...")
                    chunks = chunker.chunk(sections)
                    logger.info(f"Hoàn thành Semantic Chunker. Tạo được {len(chunks)} chunks.")
                    
                    # 4. Validation
                    validated = validator.validate(chunks)
                    passed_chunks = [vc.chunk for vc in validated if vc.validation.passed]

                    # Group by work_title directly (populated by StructureDetector Level 0 heading)
                    for chunk in passed_chunks:
                        chunk._group_title = chunk.work_title or "Sách Giáo Khoa"

                    # 4.5 Gemini Analyzer (Extract metadata)
                    if analyzer:
                        passed_chunks = analyzer.analyze(passed_chunks)

                    # Aggregation pass: unify AI metadata per group to prevent per-chunk hallucinations
                    group_meta = {}
                    for chunk in passed_chunks:
                        g = chunk._group_title
                        if g not in group_meta:
                            group_meta[g] = {
                                "author_names": [], "author_periods": [],
                                "work_periods": [], "genres": [], "sub_genres": [], "publish_years": []
                            }
                        gm = group_meta[g]
                        if chunk.author_name and "Bộ Giáo Dục" not in chunk.author_name:
                            gm["author_names"].append(chunk.author_name)
                        if getattr(chunk, 'author_period', None): gm["author_periods"].append(chunk.author_period)
                        if getattr(chunk, 'work_period', None): gm["work_periods"].append(chunk.work_period)
                        if getattr(chunk, 'genre', None): gm["genres"].append(chunk.genre)
                        if getattr(chunk, 'sub_genre', None): gm["sub_genres"].append(chunk.sub_genre)
                        if getattr(chunk, 'publish_year', None): gm["publish_years"].append(chunk.publish_year)

                    def most_frequent(lst):
                        return max(set(lst), key=lst.count) if lst else None

                    # Broadcast aggregated metadata back to all chunks
                    for chunk in passed_chunks:
                        g = chunk._group_title
                        gm = group_meta[g]
                        
                        # Never let AI touch work_title! Trust StructureDetector's hierarchy.
                        clean_title, resolved_author = self._clean_title_and_author(g)
                        
                        chunk.work_title = clean_title
                        chunk.author_name = most_frequent(gm["author_names"]) or resolved_author
                        chunk.author_period = most_frequent(gm["author_periods"])
                        chunk.work_period = most_frequent(gm["work_periods"])
                        chunk.genre = most_frequent(gm["genres"])
                        chunk.sub_genre = most_frequent(gm["sub_genres"])
                        chunk.publish_year = most_frequent(gm["publish_years"])

                    if not passed_chunks:
                        # Find the first few validation errors to report
                        errors_summary = []
                        for vc in validated[:3]:
                            if not vc.validation.passed:
                                errors_summary.append(f"{vc.chunk.chunk_id}: {', '.join(vc.validation.errors)}")
                        raise ValueError(
                            f"Không có chunk nào trong '{filename}' vượt qua quy tắc kiểm duyệt chất lượng. "
                            f"Lỗi ví dụ: {'; '.join(errors_summary)}"
                        )

                    # Deactivate existing chunks for this file (soft delete old run)
                    source_doc_id = os.path.splitext(filename)[0]
                    writer.deactivate_document(source_doc_id)

                    file_metadata = parse_filename_metadata(pdf_path)
                    total_chunks = len(passed_chunks)

                    # 5. Embedding & Saving
                    # Resolve page offset for this specific PDF file
                    page_offset = self.page_offsets.get(filename, 0)
                    if page_offset != 0:
                        logger.info(f"Áp dụng page_offset={page_offset} cho file '{filename}'")

                    ai_metadata_count = 0
                    fallback_metadata_count = 0

                    work_groups = {}
                    work_info = {}
                    work_genres = {}

                    # First pass: Extract genre for each literary work
                    for chunk in passed_chunks:
                        resolved_title = chunk.work_title or "Sách Giáo Khoa"
                        clean_title, _ = self._clean_title_and_author(resolved_title)
                        title_upper = clean_title.upper().strip()
                        
                        match = re.search(r'Thể loại:\s*([^\n]+)', chunk.content, re.IGNORECASE)
                        if match and title_upper not in work_genres:
                            raw_genre = match.group(1).strip()
                            # Normalize Vietnamese genre text to slug via taxonomy
                            if analyzer:
                                raw_genre = analyzer.normalize_genre(raw_genre)
                            work_genres[title_upper] = raw_genre

                    for idx, chunk in enumerate(passed_chunks):
                        # Generate embedding
                        emb_vector = embedder.embed_query(chunk.content)
                        
                        # Build position with page_offset applied
                        adjusted_page = chunk.page_start + page_offset
                        position = ChunkPosition(
                            page=max(1, adjusted_page),  # Ensure page >= 1
                            chunk_index=idx,
                            total_chunks=total_chunks
                        )
                        resolved_title = chunk.work_title or "Sách Giáo Khoa"
                        clean_title, resolved_author = self._clean_title_and_author(resolved_title)

                        # Prefer AI-extracted metadata if available
                        if chunk.work_title:
                            final_title = chunk.work_title
                            ai_metadata_count += 1
                        else:
                            final_title = clean_title
                            fallback_metadata_count += 1
                        final_author = chunk.author_name if chunk.author_name else resolved_author

                        title_upper = final_title.upper().strip()
                        if title_upper and title_upper not in self.excluded_titles:
                            if title_upper not in work_groups:
                                work_groups[title_upper] = []
                            if len(work_groups[title_upper]) < 5:
                                work_groups[title_upper].append(chunk.content)
                            work_info[title_upper] = {
                                "tac_gia": final_author,
                                "lop": file_metadata["lop"],
                                "hoc_ki": file_metadata["hoc_ki"]
                            }

                        # Use Gemini-extracted year, or None if unavailable
                        final_year = getattr(chunk, 'publish_year', None)
                        final_genre = getattr(chunk, 'genre', None) or work_genres.get(title_upper, "van_hoc")
                        final_sub_genre = getattr(chunk, 'sub_genre', None)
                        final_author_period = getattr(chunk, 'author_period', None) or "trung_dai"
                        final_work_period = getattr(chunk, 'work_period', None) or "trung_dai"

                        # Safety net: normalize genre/sub_genre to snake_case slug via taxonomy
                        if analyzer:
                            final_genre = analyzer.normalize_genre(final_genre)
                            final_sub_genre = analyzer.normalize_sub_genre(final_sub_genre)

                        def make_slug(s):
                            if not s:
                                return ""
                            s = remove_vietnamese_accents(s).lower()
                            s = re.sub(r'[^a-z0-9]+', '_', s)
                            return s.strip('_')

                        metadata = ChunkMetadata(
                            schema_version="literature_seed.v1",
                            work_title=final_title.upper(),
                            work_slug=make_slug(final_title),
                            author_name=final_author,
                            author_slug=make_slug(final_author),
                            author_period=final_author_period,
                            work_period=final_work_period,
                            genre=final_genre,
                            sub_genre=final_sub_genre,
                            grade=file_metadata["lop"],
                            semester=file_metadata["hoc_ki"],
                            publish_year=final_year,
                            chunk_category=chunk.chunk_category or "text_section",
                            section_slug=make_slug(chunk.section_slug) if chunk.section_slug else None,
                            section_title=chunk.section_title,
                            section_order=chunk.section_order,
                            content_type=chunk.content_type.upper() if chunk.content_type else "MIXED"
                        )
                        search_text = remove_vietnamese_accents(chunk.content)

                        chunk_doc = ChunkSchema(
                            chunk_id=chunk.chunk_id,
                            source_doc_id=source_doc_id,
                            content=chunk.content,
                            content_type=chunk.content_type,
                            position=position,
                            metadata=metadata,
                            token_count=chunk.token_count,
                            char_count=chunk.char_count,
                            has_overlap=chunk.has_overlap,
                            embedding=emb_vector,
                            search_text=search_text,
                            model_version=embedder.model_name,
                            is_active=True
                        )

                        writer.upsert_chunk(chunk_doc)

                    logger.info(
                        f"[{job_id}] Metadata source: {ai_metadata_count} chunks từ Gemini AI, "
                        f"{fallback_metadata_count} chunks dùng fallback (rule-based).")

                    # Sinh và lưu câu hỏi gợi ý cho tất cả các tác phẩm tìm thấy trong file này
                    for work_title, chunk_texts in work_groups.items():
                        info = work_info[work_title]
                        self._generate_and_save_suggestions(
                            work_title=work_title,
                            author=info["tac_gia"],
                            grade=info["lop"],
                            semester=info["hoc_ki"],
                            chunk_texts=chunk_texts
                        )

                    processed_count += 1
                    # Update progress in DB
                    self.jobs_collection.update_one(
                        {"job_id": job_id},
                        {
                            "$set": {
                                "processed_files": processed_count,
                                "updated_at": datetime.now(timezone.utc)
                            }
                        }
                    )
                    logger.info(f"[{job_id}] Successfully ingested file: {filename}")

                except Exception as file_error:
                    err_msg = f"Failed to ingest file '{filename}': {str(file_error)}"
                    logger.error(err_msg)
                    errors_list.append(err_msg)
                    
                    # Update error list in DB
                    self.jobs_collection.update_one(
                        {"job_id": job_id},
                        {
                            "$push": {"errors": err_msg},
                            "$set": {"updated_at": datetime.now(timezone.utc)}
                        }
                    )

            # Determine final status
            if processed_count == len(pdf_files):
                final_status = "done"
            elif processed_count == 0:
                final_status = "error"
            else:
                final_status = "done"  # Partial success is marked as done with errors in doc

            self.jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": final_status,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            logger.info(f"Job '{job_id}' finished with status '{final_status}'. Processed {processed_count}/{len(pdf_files)} files.")

        except Exception as e:
            err_msg = f"Fatal error in ingestion job execution: {str(e)}"
            logger.error(err_msg)
            self.jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "error",
                        "updated_at": datetime.now(timezone.utc)
                    },
                    "$push": {"errors": err_msg}
                }
            )
