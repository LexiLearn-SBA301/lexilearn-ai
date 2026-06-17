import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.ingest_service import IngestService


def _make_mock_chunk(chunk_id="chunk_1"):
    """Helper to create a mock chunk."""
    mock_chunk = MagicMock()
    mock_chunk.chunk_id = chunk_id
    mock_chunk.content = "Content"
    mock_chunk.page_start = 1
    mock_chunk.content_type = "prose"
    mock_chunk.token_count = 10
    mock_chunk.char_count = 50
    mock_chunk.has_overlap = False
    mock_chunk.tags = []
    mock_chunk.section_title = "Title"
    return mock_chunk


class TestIngestService:

    @patch("services.ingest_service.connect_to_mongo")
    @patch("services.ingest_service.get_database")
    def test_init_success(self, mock_get_db, mock_connect):
        """Test successful initialization of IngestService."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        service = IngestService()

        # Verify DB connection helpers were called
        mock_connect.assert_called_once()
        mock_get_db.assert_called_once()
        assert service.db is mock_db
        assert service.jobs_collection is mock_db["ingestion_jobs"]

    @patch("services.ingest_service.connect_to_mongo")
    @patch("services.ingest_service.get_database")
    @patch("services.ingest_service.os.path.isfile", return_value=True)
    @patch("services.ingest_service.os.path.isdir", return_value=False)
    @patch("services.ingest_service.threading.Thread")
    def test_start_ingestion_returns_job_id(
        self, mock_thread, mock_isdir, mock_isfile, mock_get_db, mock_connect
    ):
        """Test start_ingestion registers pending status and returns a job_id immediately."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_collection = mock_db["ingestion_jobs"]

        service = IngestService()

        # Mock thread to not run synchronously to test non-blocking return
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        job_id = service.start_ingestion("dummy.pdf")

        # Verify job_id is returned and record inserted
        assert isinstance(job_id, str)
        assert job_id.startswith("job_")
        
        mock_collection.insert_one.assert_called_once()
        inserted_doc = mock_collection.insert_one.call_args[0][0]
        assert inserted_doc["job_id"] == job_id
        assert inserted_doc["status"] == "pending"
        assert inserted_doc["total_files"] == 1

        # Verify background thread started
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()

    @patch("services.ingest_service.connect_to_mongo")
    @patch("services.ingest_service.get_database")
    def test_get_job_status(self, mock_get_db, mock_connect):
        """Test retrieving job status details from DB."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_collection = mock_db["ingestion_jobs"]
        
        mock_job = {
            "job_id": "job_123",
            "status": "running",
            "total_files": 2,
            "processed_files": 1,
            "errors": []
        }
        mock_collection.find_one.return_value = mock_job

        service = IngestService()
        status = service.get_job_status("job_123")

        assert status == mock_job
        mock_collection.find_one.assert_called_once_with({"job_id": "job_123"}, {"_id": 0})

    @patch("services.ingest_service.connect_to_mongo")
    @patch("services.ingest_service.get_database")
    @patch("services.ingest_service.os.path.isfile", return_value=True)
    @patch("services.ingest_service.os.path.isdir", return_value=False)
    @patch("services.ingest_service.PDFReader")
    @patch("services.ingest_service.StructureDetector")
    @patch("services.ingest_service.SemanticChunker")
    @patch("services.ingest_service.ChunkValidator")
    @patch("services.ingest_service.Embedder")
    @patch("services.ingest_service.MongoWriter")
    def test_run_ingestion_success(
        self,
        mock_writer_cls,
        mock_embedder_cls,
        mock_validator_cls,
        mock_chunker_cls,
        mock_detector_cls,
        mock_reader_cls,
        mock_isfile,
        mock_isdir,
        mock_get_db,
        mock_connect
    ):
        """Test the synchronous execution of the background task on success."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_collection = mock_db["ingestion_jobs"]

        # Setup mock pipeline components
        mock_reader = mock_reader_cls.return_value
        mock_el = MagicMock()
        mock_el.page = 1
        mock_el.raw_text = "para1"
        mock_reader.read.return_value = [mock_el]

        mock_detector = mock_detector_cls.return_value
        mock_detector.detect.return_value = ["section1"]

        mock_chunk = _make_mock_chunk()
        mock_chunker = mock_chunker_cls.return_value
        mock_chunker.chunk.return_value = [mock_chunk]

        # Mock Validator to return passed chunk
        mock_validated = MagicMock()
        mock_validated.validation.passed = True
        mock_validated.chunk = mock_chunk
        mock_validator = mock_validator_cls.return_value
        mock_validator.validate.return_value = [mock_validated]

        # Mock Embedder & Writer
        mock_embedder = mock_embedder_cls.return_value
        mock_embedder.embed_query.return_value = [0.1] * 1024
        mock_embedder.model_name = "bge-m3"

        mock_writer = mock_writer_cls.return_value

        # Initialize and call
        service = IngestService()
        service._run_ingestion_sync("job_123", ["dummy.pdf"])

        # Verify update_one calls
        # 1. Transition to running
        # 2. Progress update (processed_files = 1)
        # 3. Transition to done
        assert mock_collection.update_one.call_count == 3
        
        call_1_args = mock_collection.update_one.call_args_list[0][0]
        assert call_1_args[0] == {"job_id": "job_123"}
        assert call_1_args[1]["$set"]["status"] == "running"
        assert isinstance(call_1_args[1]["$set"]["updated_at"], datetime)

        call_2_args = mock_collection.update_one.call_args_list[1][0]
        assert call_2_args[0] == {"job_id": "job_123"}
        assert call_2_args[1]["$set"]["processed_files"] == 1

        call_3_args = mock_collection.update_one.call_args_list[2][0]
        assert call_3_args[0] == {"job_id": "job_123"}
        assert call_3_args[1]["$set"]["status"] == "done"

        # Assert document chunk writer was called
        mock_writer.upsert_chunk.assert_called_once()
        mock_writer.deactivate_document.assert_called_once_with("dummy")

    @patch("services.ingest_service.connect_to_mongo")
    @patch("services.ingest_service.get_database")
    @patch("services.ingest_service.os.path.isfile", return_value=True)
    @patch("services.ingest_service.os.path.isdir", return_value=False)
    @patch("services.ingest_service.PDFReader")
    @patch("services.ingest_service.StructureDetector")
    @patch("services.ingest_service.SemanticChunker")
    @patch("services.ingest_service.ChunkValidator")
    @patch("services.ingest_service.Embedder")
    @patch("services.ingest_service.MongoWriter")
    def test_run_ingestion_failure(
        self,
        mock_writer_cls,
        mock_embedder_cls,
        mock_validator_cls,
        mock_chunker_cls,
        mock_detector_cls,
        mock_reader_cls,
        mock_isfile,
        mock_isdir,
        mock_get_db,
        mock_connect
    ):
        """Test that background task handles exceptions, records them, and marks job as error."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_collection = mock_db["ingestion_jobs"]

        # Mock PDFReader to crash
        mock_reader = mock_reader_cls.return_value
        mock_reader.read.side_effect = Exception("PDF corruption")

        # Mock other components just in case
        mock_writer = mock_writer_cls.return_value
        mock_embedder = mock_embedder_cls.return_value
        mock_embedder.model_name = "bge-m3"

        service = IngestService()
        service._run_ingestion_sync("job_123", ["corrupted.pdf"])

        # Verify job is marked as error and error logged to list
        # Call 1: transition to running
        # Call 2: push error details
        # Call 3: transition to error status
        assert mock_collection.update_one.call_count == 3
        
        call_2_args = mock_collection.update_one.call_args_list[1][0]
        assert call_2_args[0] == {"job_id": "job_123"}
        assert "Failed to ingest file 'corrupted.pdf': PDF corruption" in call_2_args[1]["$push"]["errors"]

        call_3_args = mock_collection.update_one.call_args_list[2][0]
        assert call_3_args[0] == {"job_id": "job_123"}
        assert call_3_args[1]["$set"]["status"] == "error"
