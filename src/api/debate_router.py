"""
Debate API router — endpoint TEST gọi THẲNG Tool 2 (critics_debate) mà không cần
chạy cả pipeline / không cần Tool 1.

Nhập các field state mà critics_debate đọc (work_title, author, context_summary,
chunks — thay cho output Tool 1), server dựng intent+context giả rồi gọi
critics_debate, trả về DebateState.

Lưu ý: endpoint này gọi Qwen THẬT (8 lượt critic) -> cần Ollama đang chạy + đã pull
model. Request DTO để inline vì đây là endpoint debug (không phải API sản phẩm).
"""
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agents.critics_debate import critics_debate
from state.state_schema import (
    DebateState,
    IntentAnalysis,
    PreparedContext,
    Route,
    SourceChunk,
)

router = APIRouter(prefix="/debate", tags=["debate"])


class DebateTestRequest(BaseModel):
    work_title: Optional[str] = None
    author: Optional[str] = None
    context_summary: str = ""
    chunks: list[SourceChunk] = Field(default_factory=list)   # đoạn văn bản chung (thay Tool 1)


@router.post("/test", response_model=DebateState)
def debate_test(req: DebateTestRequest) -> DebateState:
    """Chạy Tool 2 độc lập: dựng intent+context từ input rồi gọi critics_debate."""
    state = {
        "intent": IntentAnalysis(
            raw_query=req.work_title or "",
            route=Route.DEEP,
            work_title=req.work_title,
            author=req.author,
        ),
        "context": PreparedContext(
            summary=req.context_summary,
            chunks=req.chunks,
        ),
    }
    delta = critics_debate(state)
    return delta["debate"]