import logging
import time
import functools
from typing import List, Optional
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ConnectionFailure
from models.chunk_schema import ChunkSchema

logger = logging.getLogger("rag-service.mongo-writer")
logging.basicConfig(level=logging.INFO)


def retry_on_transient_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry database operations on transient PyMongo connection failures.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while True:
                try:
                    return func(*args, **kwargs)
                except (AutoReconnect, ConnectionFailure) as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Transient MongoDB error persisted after {max_retries} retries. Raising error.")
                        raise e
                    logger.warning(
                        f"Transient MongoDB error caught: {e}. "
                        f"Retrying '{func.__name__}' in {current_delay}s (Attempt {retries}/{max_retries})..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


class MongoWriter:
    """
    MongoDB Writer layer responsible for persisting ChunkSchema objects into MongoDB Atlas.
    """

    def __init__(
        self,
        mongo_uri: str,
        database_name: str = "rag_db",
        collection_name: str = "document_chunks"
    ) -> None:
        """
        Initialize the MongoDB client, verify connection, and set up database/collection.
        Automatically triggers index creation during startup.
        """
        logger.info("Initializing MongoWriter client...")
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name

        try:
            # Initialize MongoClient
            self.client: MongoClient = MongoClient(self.mongo_uri)
            # Verify database connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection verified successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise e

        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]

        # Auto create indexes
        self.create_indexes()

    def create_indexes(self) -> None:
        """
        Idempotently create unique, metadata, text, and vector search indexes.
        """
        logger.info("Verifying/Creating database indexes...")

        # 1. Unique index on chunk_id
        try:
            self.collection.create_index([("chunk_id", 1)], unique=True, name="unique_chunk_id")
            logger.info("Unique index on 'chunk_id' created/verified.")
        except Exception as e:
            logger.error(f"Failed to create unique index on 'chunk_id': {e}")
            raise e

        # 2. Metadata & query filtering indexes
        metadata_fields = [
            ("metadata.ten_tac_pham", 1),
            ("metadata.tac_gia", 1),
            ("metadata.lop", 1),
            ("source_doc_id", 1),
            ("is_active", 1)
        ]
        for field, direction in metadata_fields:
            try:
                idx_name = f"index_{field.replace('.', '_')}"
                self.collection.create_index([(field, direction)], name=idx_name)
                logger.info(f"Index on '{field}' created/verified.")
            except Exception as e:
                logger.error(f"Failed to create index on '{field}': {e}")
                raise e

        # 3. Text search index on search_text (optimized for Vietnamese RAG)
        try:
            self.collection.create_index(
                [("search_text", "text")],
                name="search_text_index",
                default_language="none"  # Prevent default English stemmer from breaking Vietnamese tokens
            )
            logger.info("Text search index 'search_text_index' created/verified.")
        except Exception as e:
            logger.error(f"Failed to create text search index: {e}")
            raise e

        # 4. Atlas Vector Search Index (embedding_vector_index)
        # Wrap in a separate try-except block so it doesn't crash on local/mock environments.
        vector_index_name = "embedding_vector_index"
        try:
            # Check search indexes status
            existing_search_indexes = list(self.collection.list_search_indexes())
            has_vector_index = any(idx.get("name") == vector_index_name for idx in existing_search_indexes)
        except Exception as e:
            logger.warning(
                f"Failed to list Atlas Search indexes (possibly local MongoDB or free tier): {e}. "
                "Skipping Vector Index existence check."
            )
            has_vector_index = False

        if not has_vector_index:
            try:
                from pymongo.operations import SearchIndexModel
                model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 1024,
                                "similarity": "cosine"
                            }
                        ]
                    },
                    name=vector_index_name,
                    type="vectorSearch"
                )
                self.collection.create_search_index(model=model)
                logger.info(f"Atlas Vector Search index '{vector_index_name}' creation request submitted.")
            except Exception as e:
                logger.warning(
                    f"Failed to submit Atlas Search Vector index creation (unsupported on local/free-tier): {e}."
                )

    @retry_on_transient_error()
    def insert_chunk(self, chunk: ChunkSchema) -> str:
        """
        Inserts a single ChunkSchema document.
        """
        doc = chunk.model_dump()
        result = self.collection.insert_one(doc)
        logger.info(f"Chunk '{chunk.chunk_id}' inserted successfully. DB ID: {result.inserted_id}")
        return str(result.inserted_id)

    @retry_on_transient_error()
    def insert_chunks(self, chunks: List[ChunkSchema]) -> List[str]:
        """
        Bulk inserts a list of ChunkSchema documents, preserving order.
        """
        if not chunks:
            return []
        
        docs = [c.model_dump() for c in chunks]
        result = self.collection.insert_many(docs, ordered=True)
        logger.info(f"Bulk insert completed. {len(result.inserted_ids)} chunks inserted.")
        return [str(inserted_id) for inserted_id in result.inserted_ids]

    @retry_on_transient_error()
    def upsert_chunk(self, chunk: ChunkSchema) -> dict:
        """
        Upsert a chunk using chunk_id as the primary key.
        """
        doc = chunk.model_dump()
        result = self.collection.update_one(
            {"chunk_id": chunk.chunk_id},
            {"$set": doc},
            upsert=True
        )
        logger.info(
            f"Upsert completed for chunk '{chunk.chunk_id}'. "
            f"Matched: {result.matched_count}, Modified: {result.modified_count}, "
            f"Upserted ID: {result.upserted_id}"
        )
        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None
        }

    @retry_on_transient_error()
    def deactivate_document(self, source_doc_id: str) -> int:
        """
        Soft deletes all chunks of a source document by setting is_active = False.
        """
        result = self.collection.update_many(
            {"source_doc_id": source_doc_id},
            {"$set": {"is_active": False}}
        )
        logger.info(f"Deactivated {result.modified_count} chunks for source_doc_id '{source_doc_id}'.")
        return result.modified_count

    @retry_on_transient_error()
    def document_exists(self, source_doc_id: str) -> bool:
        """
        Check if any active chunks exist for the given source_doc_id.
        """
        count = self.collection.count_documents(
            {"source_doc_id": source_doc_id, "is_active": True},
            limit=1
        )
        return count > 0

    @retry_on_transient_error()
    def count_chunks(self) -> int:
        """
        Returns the total count of documents in the collection.
        """
        return self.collection.count_documents({})

    @retry_on_transient_error()
    def count_document_chunks(self, source_doc_id: str) -> int:
        """
        Returns the total count of chunks for a given source_doc_id.
        """
        return self.collection.count_documents({"source_doc_id": source_doc_id})
