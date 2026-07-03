"""
Chat API router — Hỗ trợ cả hai mô hình (Finetuned và Base) kết hợp RAG và workflow.

Mục đích: Cung cấp API cho FE truy vấn hệ thống RAG và tùy chọn model sinh câu trả lời
để phục vụ A/B Testing và luồng multi-agent workflow.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends

from schemas.chat_schema import ChatRequest, ChatResponse, WorkflowResponse
from providers.ollama_provider import FINE_TUNED_OLLAMA_LLM_MODEL, OLLAMA_BASE_LLM_MODEL
from services.rag_service import RAGService
from services.agent_service.workflow_service import WorkflowService
from api.dependencies import get_workflow
from state.agent_state import AgentState

logger = logging.getLogger("rag-service.api.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

# Singleton RAGService for API (Lazy initialization)
_rag_service: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@router.post("/only-llm", response_model=ChatResponse)
def chat_finetuned(req: ChatRequest, rag_service: RAGService = Depends(get_rag_service)) -> ChatResponse:
    """Chat với model FINE-TUNE + hệ thống RAG."""
    logger.info("RAG query -> model %s", FINE_TUNED_OLLAMA_LLM_MODEL)
    result = rag_service.query(query=req.message, filters=req.filters, limit=req.limit, model_name=FINE_TUNED_OLLAMA_LLM_MODEL)
    return ChatResponse(
        answer=result.get("answer", ""),
        model=FINE_TUNED_OLLAMA_LLM_MODEL,
        sources=result.get("sources", [])
    )


@router.post("/base-llm", response_model=ChatResponse)
def chat_base(req: ChatRequest, rag_service: RAGService = Depends(get_rag_service)) -> ChatResponse:
    """Chat với model GỐC + hệ thống RAG để so sánh."""
    logger.info("RAG query -> model %s", OLLAMA_BASE_LLM_MODEL)
    result = rag_service.query(query=req.message, filters=req.filters, limit=req.limit, model_name=OLLAMA_BASE_LLM_MODEL)
    return ChatResponse(
        answer=result.get("answer", ""),
        model=OLLAMA_BASE_LLM_MODEL,
        sources=result.get("sources", [])
    )

@router.post("/llm-extended", response_model=AgentState)
async def chat_with_workflow(req: ChatRequest, wf: WorkflowService = Depends(get_workflow)) -> AgentState:
    """Chat với model FINE-TUNE kèm workflow Multi Agent."""
    thread_id = req.thread_id if req.thread_id else uuid.uuid4().hex
    state = await wf.invoke(req.message, thread_id)
    return state
