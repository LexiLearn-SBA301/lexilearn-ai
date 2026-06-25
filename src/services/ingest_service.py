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
from core.structure_detector import StructureDetector
from core.semantic_chunker import SemanticChunker
from core.chunk_validator import ChunkValidator
from core.embedder import Embedder
from core.mongo_writer import MongoWriter
from models.chunk_schema import ChunkSchema, ChunkPosition, ChunkMetadata

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
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.known_authors = config.get("known_authors", {})
        self.work_to_author = config.get("work_to_author", {})
        
        logger.info("IngestService initialized successfully.")

    def start_ingestion(self, pdf_path_or_dir: str) -> str:
        """
        Start the ingestion process asynchronously.
        Returns the job_id immediately.
        """
        # Resolve target files
        pdf_files = []
        if os.path.isdir(pdf_path_or_dir):
            for filename in os.listdir(pdf_path_or_dir):
                if filename.endswith(".pdf"):
                    pdf_files.append(os.path.normpath(os.path.join(pdf_path_or_dir, filename)))
        elif os.path.isfile(pdf_path_or_dir):
            pdf_files.append(os.path.normpath(pdf_path_or_dir))
        else:
            raise FileNotFoundError(f"Path not found: {pdf_path_or_dir}")

        if not pdf_files:
            raise ValueError(f"No PDF files found in: {pdf_path_or_dir}")

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
            args=(job_id, pdf_files)
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

    def _resolve_work_title(self, chunk_title: str, page_start: int, sections: list) -> str:
        """
        Traces the parent section chain in the DocumentSections to resolve the actual literary work name (Level 0).
        """
        matching_section = None
        for sec in sections:
            if sec.title == chunk_title and sec.page_start <= page_start <= sec.page_end:
                matching_section = sec
                break
                
        if not matching_section:
            for sec in sections:
                if sec.title == chunk_title:
                    matching_section = sec
                    break
                    
        if not matching_section:
            return chunk_title or "Sách Giáo Khoa"
            
        current = matching_section
        visited = set()
        while current and current.level > 0 and current.title not in visited:
            visited.add(current.title)
            parent = next((s for s in reversed(sections) if s.title == current.parent_title), None)
            if not parent:
                break
            current = parent
            
        return current.title if current else chunk_title or "Sách Giáo Khoa"

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
                return cleaned_title, clean_auth
            elif re.search(pattern_space, upper_title):
                cleaned_title = re.sub(pattern_space, "", title, flags=re.IGNORECASE).strip()
                return cleaned_title, clean_auth
                
        # 2. If no explicit author suffix, lookup the cleaned title in the work_to_author map
        # Strip parens and normalize to match dictionary keys
        title_no_parens = re.sub(r'\(.*?\)', '', title).strip().lower()
        if title_no_parens in self.work_to_author:
            return title, self.work_to_author[title_no_parens]
            
        # 3. Fallback to default
        return title, "Bộ Giáo Dục và Đào Tạo"


    def _run_ingestion_sync(self, job_id: str, pdf_files: List[str]) -> None:
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
            reader = PDFReader()
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
                    # 1. Read PDF
                    elements = reader.read(pdf_path)
                    if not elements:
                        gemini_key = os.getenv("GEMINI_API_KEY")
                        raise ValueError(
                            f"Tệp PDF '{filename}' không chứa văn bản dạng số (digital text) và tất cả OCR dự phòng thất bại. "
                            f"Kiểm tra GEMINI_API_KEY (hiện tại: {'có' if gemini_key else 'chưa cấu hình'})"
                        )
                        
                    # 2. Detect Structure
                    sections = detector.detect(elements)
                    
                    # 3. Chunking
                    chunks = chunker.chunk(sections)
                    
                    # 4. Validation
                    validated = validator.validate(chunks)
                    passed_chunks = [vc.chunk for vc in validated if vc.validation.passed]

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
                    for idx, chunk in enumerate(passed_chunks):
                        # Generate embedding
                        emb_vector = embedder.embed_query(chunk.content)
                        
                        # Build position & metadata
                        position = ChunkPosition(
                            page=chunk.page_start,
                            chunk_index=idx,
                            total_chunks=total_chunks
                        )
                        resolved_title = self._resolve_work_title(
                            chunk.section_title,
                            chunk.page_start,
                            sections
                        )
                        clean_title, resolved_author = self._clean_title_and_author(resolved_title)

                        metadata = ChunkMetadata(
                            ten_tac_pham=clean_title,
                            tac_gia=resolved_author,
                            lop=file_metadata["lop"],
                            the_loai=chunk.content_type,
                            hoc_ki=file_metadata["hoc_ki"],
                            nam_sang_tac=2023,
                            tags=chunk.tags
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
