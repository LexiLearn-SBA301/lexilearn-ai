import os
import re
import logging
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from db.mongo_client import connect_to_mongo, get_database
from core.embedder import Embedder
from core.mongo_writer import MongoWriter
from retrievals.rrf import reciprocal_rank_fusion
from providers.ollama_provider import ollama_provider
from config.prompt_template import SYSTEM_PROMPT

logger = logging.getLogger("rag-service.services.rag-service")
logging.basicConfig(level=logging.INFO)


def remove_vietnamese_accents(text: str) -> str:
    """Helper to convert Vietnamese text to clean lowercase ASCII-like text for FTS."""
    if not text:
        return ""
    normalized = unicodedata.normalize('NFD', text)
    stripped = "".join([c for c in normalized if not unicodedata.combining(c)])
    stripped = stripped.replace('Đ', 'D').replace('đ', 'd')
    return re.sub(r'\s+', ' ', stripped).strip().lower()


class RAGService:
    """
    RAGService coordinates the query pipeline:
    embeds queries -> runs VectorSearch and KeywordSearch concurrently ->
    merges rankings using Reciprocal Rank Fusion (RRF) -> retrieves top chunks ->
    synthesizes answers using Ollama LLM.
    """

    def __init__(self, db_name: Optional[str] = None) -> None:
        """
        Initialize connections, writer, and embedder.
        """
        connect_to_mongo()
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/rag_db")
        target_db = db_name or "rag_db"
        self.writer = MongoWriter(mongo_uri=mongodb_uri, database_name=target_db)
        self.db = self.writer.db
        self.collection = self.writer.collection
        self.embedder = Embedder()
        logger.info(f"RAGService initialized successfully with database '{target_db}'.")

    def _vector_search(self, query_vector: List[float], db_filter: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Run Vector Search. Attempts MongoDB Atlas Vector Search first, and falls back to
        in-memory cosine similarity using numpy if Atlas Search is unsupported (local/free tier).
        """
        try:
            # Atlas Vector Search Stage
            pipeline: List[Dict[str, Any]] = [
                {
                    "$vectorSearch": {
                        "index": "embedding_vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": max(100, limit * 10),
                        "limit": max(50, limit * 2)
                    }
                }
            ]
            if db_filter:
                pipeline.append({"$match": db_filter})

            pipeline.append({"$project": {"chunk_id": 1, "score": {"$meta": "searchScore"}}})
            results = list(self.collection.aggregate(pipeline))
            if results:
                logger.info(f"Atlas Vector Search returned {len(results)} chunks.")
                return [{"chunk_id": r["chunk_id"], "score": r.get("score", 0.0)} for r in results]
        except Exception as e:
            logger.debug(f"Atlas Vector Search failed or not supported ({e}). Falling back to in-memory cosine similarity.")

        # Local Fallback: in-memory cosine similarity using numpy
        import numpy as np
        cursor = self.collection.find(db_filter, {"chunk_id": 1, "embedding": 1})
        q_vec = np.array(query_vector)
        q_norm = np.linalg.norm(q_vec)

        results = []
        for doc in cursor:
            emb = doc.get("embedding")
            if not emb or not isinstance(emb, list):
                continue
            emb_vec = np.array(emb)
            emb_norm = np.linalg.norm(emb_vec)
            if q_norm == 0 or emb_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q_vec, emb_vec) / (q_norm * emb_norm))
            results.append({"chunk_id": doc["chunk_id"], "score": sim})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max(50, limit * 2)]

    def _keyword_search(self, query: str, db_filter: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        Run Full-text Search. Attempts MongoDB Text Search first, and falls back to
        regex search on search_text if the text index is unavailable.
        """
        normalized_query = remove_vietnamese_accents(query)
        if not normalized_query:
            return []

        keyword_filter = db_filter.copy()
        keyword_filter["$text"] = {"$search": normalized_query}

        try:
            cursor = self.collection.find(
                keyword_filter,
                {"chunk_id": 1, "score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(max(50, limit * 2))

            results = []
            for doc in cursor:
                results.append({"chunk_id": doc["chunk_id"], "score": doc.get("score", 0.0)})
            return results
        except Exception as e:
            logger.debug(f"Text index search failed or not supported ({e}). Falling back to regex search on search_text.")

            # Fallback: regex search on search_text
            fallback_filter = db_filter.copy()
            escaped_query = re.escape(normalized_query)
            fallback_filter["search_text"] = {"$regex": escaped_query, "$options": "i"}

            cursor = self.collection.find(fallback_filter, {"chunk_id": 1}).limit(max(50, limit * 2))
            return [{"chunk_id": doc["chunk_id"], "score": 1.0} for doc in cursor]

    def hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using Vector & Keyword Search in parallel, merge results with RRF,
        and retrieve top documents.
        """
        if not query or not query.strip():
            return []

        # 1. Preprocess Filters
        db_filter: Dict[str, Any] = {"is_active": True}
        if filters:
            metadata_keys = {"ten_tac_pham", "tac_gia", "lop", "the_loai", "hoc_ki", "nam_sang_tac", "tags"}
            for key, val in filters.items():
                if key.startswith("metadata.") or key in ["source_doc_id", "chunk_id", "is_active"]:
                    db_filter[key] = val
                elif key in metadata_keys:
                    db_filter[f"metadata.{key}"] = val
                else:
                    db_filter[key] = val

        # 2. Get Query Embedding
        query_vector = self.embedder.embed_query(query)

        # 3. Concurrent Retrieval
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_vector = executor.submit(self._vector_search, query_vector, db_filter, limit)
            future_keyword = executor.submit(self._keyword_search, query, db_filter, limit)

            vector_results = future_vector.result()
            keyword_results = future_keyword.result()

        # 4. RRF Merge
        merged_rankings = reciprocal_rank_fusion(vector_results, keyword_results, k=k)
        top_rankings = merged_rankings[:limit]

        if not top_rankings:
            return []

        # 5. Populate and enrich documents preserving order
        top_chunk_ids = [item["chunk_id"] for item in top_rankings]
        rrf_scores = {item["chunk_id"]: item["rrf_score"] for item in top_rankings}

        docs = list(self.collection.find({"chunk_id": {"$in": top_chunk_ids}}))
        doc_map = {doc["chunk_id"]: doc for doc in docs}

        ordered_docs = []
        for chunk_id in top_chunk_ids:
            if chunk_id in doc_map:
                doc = doc_map[chunk_id]
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                doc["rrf_score"] = rrf_scores[chunk_id]
                ordered_docs.append(doc)

        return ordered_docs

    def query(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 5) -> Dict[str, Any]:
        """
        Answer a RAG query by retrieving contexts and calling Ollama LLM to synthesize the final answer.
        """
        # Retrieve context documents
        chunks = self.hybrid_search(query, filters=filters, limit=limit)

        if not chunks:
            context = "Không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu."
        else:
            context_parts = []
            for idx, chunk in enumerate(chunks):
                metadata = chunk.get("metadata", {})
                title = metadata.get("ten_tac_pham", "Không rõ tác phẩm")
                author = metadata.get("tac_gia", "Không rõ tác giả")
                page = chunk.get("position", {}).get("page", "?")
                content = chunk.get("content", "")
                
                part = f"Tài liệu {idx + 1} (Tác phẩm: '{title}' - Tác giả: {author}, Trang: {page}):\n{content}"
                context_parts.append(part)
            context = "\n\n".join(context_parts)

        system_prompt = SYSTEM_PROMPT

        user_prompt = f"Ngữ cảnh:\n---\n{context}\n---\n\nCâu hỏi: {query}\n\nTrả lời:"

        try:
            llm = ollama_provider.get_llm()
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
            if isinstance(response.content, str):
                answer = response.content.strip()
            elif isinstance(response.content, list):
                text_parts = []
                for block in response.content:
                    if isinstance(block, str):
                        text_parts.append(block)
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(str(block["text"]))
                answer = "".join(text_parts).strip()
            else:
                answer = str(response.content).strip()
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            answer = f"Lỗi khi gọi mô hình ngôn ngữ LLM để tạo câu trả lời: {str(e)}"

        return {
            "answer": answer,
            "sources": chunks
        }

    def evaluate(self, ground_truth_path: str = "ground_truth.json", limit: int = 5) -> Dict[str, Any]:
        """
        Evaluate retrieval performance of the RAG system using Hit Rate @ N and MRR @ N.
        """
        import json

        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

        with open(ground_truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_queries = len(data)
        hits = 0
        mrr_sum = 0.0

        print(f"Bắt đầu đánh giá RAG với {total_queries} câu hỏi...")
        for i, item in enumerate(data):
            query_str = item["query"]
            expected_work = item.get("ten_tac_pham", "").strip().lower()

            # Retrieve top chunks
            retrieved_chunks = self.hybrid_search(query_str, limit=limit)

            rank = 0
            for idx, chunk in enumerate(retrieved_chunks):
                retrieved_work = chunk.get("metadata", {}).get("ten_tac_pham", "").strip().lower()
                if retrieved_work == expected_work:
                    rank = idx + 1
                    break

            if rank > 0:
                hits += 1
                mrr_sum += 1.0 / rank

            # Print status every 10 queries
            if (i + 1) % 10 == 0 or (i + 1) == total_queries:
                current_hr = hits / (i + 1)
                current_mrr = mrr_sum / (i + 1)
                print(f" -> Đã xử lý {i + 1}/{total_queries} câu hỏi | Hit Rate@{limit}: {current_hr:.4f} | MRR@{limit}: {current_mrr:.4f}")

        hit_rate = hits / total_queries if total_queries > 0 else 0.0
        mrr = mrr_sum / total_queries if total_queries > 0 else 0.0

        result = {
            "total_queries": total_queries,
            "hits": hits,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "limit": limit
        }
        logger.info(f"RAG Evaluation finished: Hit Rate@{limit} = {hit_rate:.4f}, MRR@{limit} = {mrr:.4f}")
        return result
