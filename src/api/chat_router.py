"""
Chat API router — Hỗ trợ cả hai mô hình (Finetuned và Base) kết hợp RAG.

Mục đích: Cung cấp API cho FE truy vấn hệ thống RAG và tùy chọn model sinh câu trả lời
để phục vụ A/B Testing.
"""
import logging
from typing import Any, List, Optional, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from providers.ollama_provider import FINE_TUNED_OLLAMA_LLM_MODEL, OLLAMA_BASE_LLM_MODEL
from services.rag_service import RAGService

logger = logging.getLogger("rag-service.api.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

# Singleton RAGService for API
rag_service = RAGService()

class ChatRequest(BaseModel):
    message: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 5

class ChatResponse(BaseModel):
    answer: str
    model: str
    sources: List[Any] = []

@router.post("/only-llm", response_model=ChatResponse)
def chat_finetuned(req: ChatRequest) -> ChatResponse:
    """Chat với model FINE-TUNE + hệ thống RAG."""
    logger.info("RAG query -> model %s", FINE_TUNED_OLLAMA_LLM_MODEL)
    result = rag_service.query(query=req.message, filters=req.filters, limit=req.limit, model_name=FINE_TUNED_OLLAMA_LLM_MODEL)
    return ChatResponse(
        answer=result.get("answer", ""),
        model=FINE_TUNED_OLLAMA_LLM_MODEL,
        sources=result.get("sources", [])
    )

@router.post("/base-llm", response_model=ChatResponse)
def chat_base(req: ChatRequest) -> ChatResponse:
    """Chat với model GỐC + hệ thống RAG để so sánh."""
    logger.info("RAG query -> model %s", OLLAMA_BASE_LLM_MODEL)
    result = rag_service.query(query=req.message, filters=req.filters, limit=req.limit, model_name=OLLAMA_BASE_LLM_MODEL)
    return ChatResponse(
        answer=result.get("answer", ""),
        model=OLLAMA_BASE_LLM_MODEL,
        sources=result.get("sources", [])
    )
