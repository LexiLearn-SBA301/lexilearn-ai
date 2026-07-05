"""
Chat API router — Hỗ trợ cả hai mô hình (Finetuned và Base) kết hợp RAG và workflow.

Mục đích: Cung cấp API cho FE truy vấn hệ thống RAG và tùy chọn model sinh câu trả lời
để phục vụ A/B Testing và luồng multi-agent workflow.
"""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse

from schemas.chat_schema import ChatRequest, ChatResponse, WorkflowResponse
from schemas.suggestion_schema import SuggestionsResponse
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
    # final_ai_response = state.get("final_ai_response", "")
    # route = state.get("route", "")
    # return WorkflowResponse(answer=final_ai_response, route=route)
    return state


@router.post("/stream")
async def chat_stream(req: ChatRequest, wf: WorkflowService = Depends(get_workflow)) -> StreamingResponse:
    """Chat workflow Multi Agent — STREAM tiến trình (thinking) + output ra UI qua SSE.

    Mỗi dòng SSE là `data: <StreamEvent json>\\n\\n`. FE đọc bằng fetch + ReadableStream
    (POST nên không dùng EventSource). Kết thúc bằng event type=done từ node finalize.
    """
    thread_id = req.thread_id if req.thread_id else uuid.uuid4().hex

    async def gen():
        try:
            async for ev in wf.astream(req.message, thread_id):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # lỗi giữa chừng -> báo 1 event ERROR rồi đóng stream
            logger.exception("Stream workflow failed thread=%s", thread_id)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # tắt buffer của nginx để FE thấy realtime
        },
    )

@router.get("/works/suggestions", response_model=SuggestionsResponse)
def get_work_suggestions(
    work_title: str = Query(..., description="Tên tác phẩm cần lấy câu hỏi gợi ý"),
    rag_service: RAGService = Depends(get_rag_service)
) -> SuggestionsResponse:
    """Lấy danh sách 3 câu hỏi gợi ý cho một tác phẩm (hỗ trợ lazy-caching)."""
    try:
        res = rag_service.get_suggested_questions(work_title)
        return SuggestionsResponse(
            ten_tac_pham=res["ten_tac_pham"],
            tac_gia=res["tac_gia"],
            suggested_questions=res["suggested_questions"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error("Lỗi khi lấy câu hỏi gợi ý: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi hệ thống khi lấy câu hỏi gợi ý: {str(e)}"
        )
