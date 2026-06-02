import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.chunk_schema import ChunkSchema, ChunkPosition, ChunkMetadata
from core.mongo_writer import MongoWriter


def _make_dummy_chunk(chunk_id="vo-nhat_p012_c001", source_doc_id="vo-nhat_12"):
    """Helper to create a valid ChunkSchema instance for testing."""
    return ChunkSchema(
        chunk_id=chunk_id,
        source_doc_id=source_doc_id,
        content="Tràng là người dân ngụ cư...",
        content_type="prose",
        position=ChunkPosition(page=12, chunk_index=1, total_chunks=18),
        metadata=ChunkMetadata(
            ten_tac_pham="Vợ Nhặt",
            tac_gia="Kim Lân",
            lop=12,
            the_loai="truyen_ngan",
            hoc_ki=1,
            nam_sang_tac=1962,
            tags=["nhan_vat_trang"]
        ),
        token_count=312,
        char_count=687,
        has_overlap=True,
        embedding=[0.023] * 1024,
        search_text="trang la nguoi dan ngu cu",
        model_version="bge-m3-v1.0",
        is_active=True
    )


class TestMongoWriter:

    @patch("core.mongo_writer.MongoClient")
    def test_init_success(self, mock_mongo_client):
        """Test successful initialization and connection verification."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance

        # Call constructor
        writer = MongoWriter("mongodb+srv://user:pass@cluster.mongodb.net/test")

        # Assert client called and connected
        mock_mongo_client.assert_called_once_with("mongodb+srv://user:pass@cluster.mongodb.net/test")
        mock_client_instance.admin.command.assert_called_once_with("ping")

        # Assert database and collection assigned
        assert writer.db is not None
        assert writer.collection is not None

    @patch("core.mongo_writer.MongoClient")
    def test_init_connection_failure(self, mock_mongo_client):
        """Test connection failure raises exception during initialization."""
        mock_client_instance = MagicMock()
        mock_client_instance.admin.command.side_effect = Exception("Connection timed out")
        mock_mongo_client.return_value = mock_client_instance

        with pytest.raises(Exception, match="Connection timed out"):
            MongoWriter("mongodb://invalid-uri")

    @patch("core.mongo_writer.MongoClient")
    def test_create_indexes(self, mock_mongo_client):
        """Test index creation logic handles unique, text, metadata and search indexes."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance

        mock_collection = mock_client_instance["rag_db"]["document_chunks"]
        mock_collection.list_search_indexes.return_value = []

        writer = MongoWriter("mongodb://localhost")

        # Verify unique index on chunk_id was created
        mock_collection.create_index.assert_any_call(
            [("chunk_id", 1)], unique=True, name="unique_chunk_id"
        )

        # Verify metadata indexes were created
        mock_collection.create_index.assert_any_call(
            [("metadata.ten_tac_pham", 1)], name="index_metadata_ten_tac_pham"
        )
        mock_collection.create_index.assert_any_call(
            [("metadata.tac_gia", 1)], name="index_metadata_tac_gia"
        )
        mock_collection.create_index.assert_any_call(
            [("metadata.lop", 1)], name="index_metadata_lop"
        )
        mock_collection.create_index.assert_any_call(
            [("source_doc_id", 1)], name="index_source_doc_id"
        )
        mock_collection.create_index.assert_any_call(
            [("is_active", 1)], name="index_is_active"
        )

        # Verify text index on search_text was created
        mock_collection.create_index.assert_any_call(
            [("search_text", "text")], name="search_text_index", default_language="none"
        )

        # Verify create_search_index called for Vector Search index
        mock_collection.create_search_index.assert_called_once()

    @patch("core.mongo_writer.MongoClient")
    def test_insert_chunk(self, mock_mongo_client):
        """Test inserting a single chunk document."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        
        mock_collection = mock_client_instance["rag_db"]["document_chunks"]
        mock_insert_result = MagicMock()
        mock_insert_result.inserted_id = "mock_bson_id"
        mock_collection.insert_one.return_value = mock_insert_result

        writer = MongoWriter("mongodb://localhost")
        chunk = _make_dummy_chunk()
        inserted_id = writer.insert_chunk(chunk)

        # Verify result and call
        assert inserted_id == "mock_bson_id"
        mock_collection.insert_one.assert_called_once()
        # Verify document content was dumped from Pydantic
        call_args = mock_collection.insert_one.call_args[0][0]
        assert call_args["chunk_id"] == "vo-nhat_p012_c001"
        assert call_args["content_type"] == "prose"

    @patch("core.mongo_writer.MongoClient")
    def test_insert_chunks_bulk(self, mock_mongo_client):
        """Test bulk inserting multiple chunks."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        
        mock_collection = mock_client_instance["rag_db"]["document_chunks"]
        mock_bulk_result = MagicMock()
        mock_bulk_result.inserted_ids = ["id_1", "id_2"]
        mock_collection.insert_many.return_value = mock_bulk_result

        writer = MongoWriter("mongodb://localhost")
        chunks = [
            _make_dummy_chunk("chunk_1", "doc_1"),
            _make_dummy_chunk("chunk_2", "doc_1")
        ]
        inserted_ids = writer.insert_chunks(chunks)

        # Verify bulk insert calls
        assert inserted_ids == ["id_1", "id_2"]
        mock_collection.insert_many.assert_called_once()
        call_args = mock_collection.insert_many.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["chunk_id"] == "chunk_1"
        assert call_args[1]["chunk_id"] == "chunk_2"

    @patch("core.mongo_writer.MongoClient")
    def test_upsert_chunk(self, mock_mongo_client):
        """Test upserting a chunk document."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        
        mock_collection = mock_client_instance["rag_db"]["document_chunks"]
        mock_update_result = MagicMock()
        mock_update_result.matched_count = 1
        mock_update_result.modified_count = 0
        mock_update_result.upserted_id = None
        mock_collection.update_one.return_value = mock_update_result

        writer = MongoWriter("mongodb://localhost")
        chunk = _make_dummy_chunk()
        result = writer.upsert_chunk(chunk)

        # Verify upsert details
        assert result["matched_count"] == 1
        assert result["modified_count"] == 0
        assert result["upserted_id"] is None
        mock_collection.update_one.assert_called_once()
        filter_arg = mock_collection.update_one.call_args[0][0]
        update_arg = mock_collection.update_one.call_args[0][1]
        kwargs = mock_collection.update_one.call_args[1]

        assert filter_arg == {"chunk_id": "vo-nhat_p012_c001"}
        assert "$set" in update_arg
        assert kwargs.get("upsert") is True

    @patch("core.mongo_writer.MongoClient")
    def test_deactivate_document(self, mock_mongo_client):
        """Test deactivating a document (soft delete)."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        
        mock_collection = mock_client_instance["rag_db"]["document_chunks"]
        mock_update_result = MagicMock()
        mock_update_result.modified_count = 15
        mock_collection.update_many.return_value = mock_update_result

        writer = MongoWriter("mongodb://localhost")
        modified_count = writer.deactivate_document("vo-nhat_12")

        # Verify soft delete
        assert modified_count == 15
        mock_collection.update_many.assert_called_once_with(
            {"source_doc_id": "vo-nhat_12"},
            {"$set": {"is_active": False}}
        )

    @patch("core.mongo_writer.MongoClient")
    def test_utilities(self, mock_mongo_client):
        """Test document_exists, count_chunks, and count_document_chunks utilities."""
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        
        mock_collection = mock_client_instance["rag_db"]["document_chunks"]
        mock_collection.count_documents.side_effect = [1, 42, 18]

        writer = MongoWriter("mongodb://localhost")

        # Test document_exists
        exists = writer.document_exists("vo-nhat_12")
        assert exists is True
        mock_collection.count_documents.assert_any_call(
            {"source_doc_id": "vo-nhat_12", "is_active": True},
            limit=1
        )

        # Test count_chunks
        total_count = writer.count_chunks()
        assert total_count == 42
        mock_collection.count_documents.assert_any_call({})

        # Test count_document_chunks
        doc_count = writer.count_document_chunks("vo-nhat_12")
        assert doc_count == 18
        mock_collection.count_documents.assert_any_call({"source_doc_id": "vo-nhat_12"})
