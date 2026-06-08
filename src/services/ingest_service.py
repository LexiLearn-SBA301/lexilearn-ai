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
                        try:
                            import pytesseract
                            import pdf2image
                            has_ocr_deps = True
                        except ImportError:
                            has_ocr_deps = False

                        if not has_ocr_deps:
                            raise ValueError(
                                f"Tệp PDF '{filename}' không chứa văn bản dạng số (digital text). "
                                "Không thể kích hoạt OCR dự phòng do thiếu thư viện 'pytesseract' hoặc 'pdf2image'. "
                                "Vui lòng kiểm tra xem bạn đã kích hoạt môi trường ảo (.venv) chưa bằng cách chạy: "
                                ".venv\\Scripts\\python main.py ..."
                            )
                        else:
                            raise ValueError(
                                f"Tệp PDF '{filename}' không chứa văn bản dạng số (digital text) và OCR dự phòng thất bại. "
                                "Vui lòng kiểm tra lại đường dẫn cài đặt TESSERACT_CMD và POPPLER_PATH trong tệp .env."
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
                        metadata = ChunkMetadata(
                            ten_tac_pham=chunk.section_title or "Sách Giáo Khoa",
                            tac_gia="Bộ Giáo Dục và Đào Tạo",
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
