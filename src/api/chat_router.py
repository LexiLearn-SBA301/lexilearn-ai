"""
Chat API router — Hỗ trợ cả hai mô hình (Finetuned và Base) kết hợp RAG và workflow.

Mục đích: Cung cấp API cho FE truy vấn hệ thống RAG và tùy chọn model sinh câu trả lời
để phục vụ A/B Testing và luồng multi-agent workflow.
"""

import asyncio
import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse

from schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
    DebateOptinRequest,
    DebateReplyRequest,
    SeedRequest,
    WorkflowResponse,
)
from schemas.suggestion_schema import SuggestionsResponse
from providers.ollama_provider import FINE_TUNED_OLLAMA_LLM_MODEL, OLLAMA_BASE_LLM_MODEL
from services.rag_service import RAGService
from services.agent_service import debate_session
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
    state = await wf.invoke(req.message, thread_id, filters=req.filters)
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
            async for ev in wf.astream(req.message, thread_id, filters=req.filters):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            # FE bấm Dừng / đóng popup -> abort fetch. Starlette cancel task này, LangGraph
            # hủy node đang chạy, request tới Ollama bị hủy theo. Re-raise để cancel lan tiếp.
            logger.info("Stream bị hủy bởi client thread=%s", thread_id)
            raise
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

@router.get("/exists/{thread_id}")
async def chat_thread_exists(thread_id: str, wf: WorkflowService = Depends(get_workflow)) -> dict:
    """[Nội bộ, BE gọi] Checkpoint của thread_id còn lịch sử không -> BE quyết seed hay không."""
    return {"thread_id": thread_id, "exists": await wf.thread_exists(thread_id)}


@router.post("/seed")
async def chat_seed(req: SeedRequest, wf: WorkflowService = Depends(get_workflow)) -> dict:
    """[Nội bộ, BE gọi] Nạp lịch sử vào checkpoint khi mở lại chat cũ (checkpoint nguội).

    KHÔNG chạy graph — chỉ ghi messages vào state. BE gọi trước khi stream nếu /exists=false.
    """
    seeded = await wf.seed_history(req.thread_id, req.history)
    return {"thread_id": req.thread_id, "seeded": seeded}


@router.post("/debate/optin")
async def chat_debate_optin(req: DebateOptinRequest) -> dict:
    """[Nội bộ, BE gọi] Người học bấm "Tranh luận cùng AI" TRONG LÚC luồng đang chạy.

    Cửa sổ bấm = lúc Tool 1 chuẩn bị ngữ cảnh + judge chấm. Node debate đọc cờ này đúng
    1 lần lúc vào; bấm muộn hơn thì cờ không ai đọc (FE cũng đã khoá nút khi nhận event
    debate_lock) và sẽ được dọn ở finalize.
    """
    debate_session.mark_optin(req.thread_id)
    return {"thread_id": req.thread_id, "optin": True}


@router.post("/debate/reply")
async def chat_debate_reply(req: DebateReplyRequest) -> dict:
    """[Nội bộ, BE gọi] 1 lượt phát biểu của người học khi node debate đang pause.

    Chạy trên request KHÁC với stream SSE đang mở (SSE không nhận được chiều lên): đẩy
    tin vào queue của phiên -> node tỉnh dậy và chạy tiếp, event mới chảy ra CHÍNH cái
    stream cũ. Không đụng gì tới stream ở đây.

    message rỗng/None = Bỏ qua / Kết thúc phản biện. Lỗi (409/400) do exception_handlers
    map từ DebateNotWaiting / DebateInvalidReply.
    """
    text = (req.message or "").strip()
    reply = debate_session.HumanReply(
        message=text, target_arg_id=req.target_arg_id, stance=req.stance,
    ) if text else None
    debate_session.submit(req.thread_id, reply)
    return {"thread_id": req.thread_id, "accepted": True, "ended": reply is None}


@router.get("/works/suggestions", response_model=SuggestionsResponse)
def get_work_suggestions(
    work_title: str = Query(..., description="Tên tác phẩm cần lấy câu hỏi gợi ý"),
    rag_service: RAGService = Depends(get_rag_service)
) -> SuggestionsResponse:
    """Lấy danh sách 3 câu hỏi gợi ý cho một tác phẩm (hỗ trợ lazy-caching)."""
    try:
        res = rag_service.get_suggested_questions(work_title)
        return SuggestionsResponse(
            work_title=res["work_title"],
            author_name=res["author_name"],
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
