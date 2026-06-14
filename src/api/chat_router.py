"""
Chat API router — endpoint thử nhanh LLM (gọi THẲNG Ollama, KHÔNG qua RAG).

Mục đích: kiểm tra model fine-tune trả lời ra sao mà không cần dữ liệu trong Mongo.
Phần truy hồi ngữ cảnh (RAG) nằm ở RAGService; router này cố tình giữ tối giản.
"""
import logging
from typing import Any, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

from providers.ollama_provider import ollama_provider, OLLAMA_LLM_MODEL

logger = logging.getLogger("rag-service.api.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    system: Optional[str] = None   # system prompt tùy chọn (mặc định không có)


class ChatResponse(BaseModel):
    answer: str
    model: str


def _extract_text(content: Any) -> str:
    """ChatOllama có thể trả str hoặc list block -> gộp về 1 chuỗi sạch."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts).strip()
    return str(content).strip()


@router.post("/only-llm", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Gửi 1 câu tới LLM fine-tune và nhận câu trả lời (không truy hồi tài liệu)."""
    llm = ollama_provider.get_llm()

    messages: List[Any] = []
    if req.system:
        messages.append(SystemMessage(content=req.system))
    messages.append(HumanMessage(content=req.message))

    logger.info("Chat request (%d ký tự) -> model %s", len(req.message), OLLAMA_LLM_MODEL)
    response = llm.invoke(messages)
    return ChatResponse(answer=_extract_text(response.content), model=OLLAMA_LLM_MODEL)
