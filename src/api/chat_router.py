"""
Chat API router — endpoint thử nhanh LLM (gọi THẲNG Ollama, KHÔNG qua RAG).

Mục đích: kiểm tra model fine-tune trả lời ra sao mà không cần dữ liệu trong Mongo.
Phần truy hồi ngữ cảnh (RAG) nằm ở RAGService; router này cố tình giữ tối giản.
"""

from fastapi import APIRouter
from schemas.chat_schema import ChatRequest, ChatResponse
from providers.ollama_provider import FINE_TUNED_OLLAMA_LLM_MODEL, OLLAMA_BASE_LLM_MODEL
from services.agent_service.chat_service import OllamaChatService
from services.agent_service.workflow_service import WorkflowService
from state.agent_state import AgentState

router = APIRouter(prefix="/chat", tags=["chat"])
chat_service = OllamaChatService()
invoke_service = WorkflowService() # khi khởi tạo app chạy -> chạy luôn hàm init ở đây
# mỗi request không init lại

@router.post("/only-llm", response_model=ChatResponse)
def chat_finetuned(req: ChatRequest) -> ChatResponse:
    """Chat với model FINE-TUNE (chưa có RAG)."""
    return chat_service.run_chat(req, FINE_TUNED_OLLAMA_LLM_MODEL)


@router.post("/base-llm", response_model=ChatResponse)
def chat_base(req: ChatRequest) -> ChatResponse:
    """Chat với model GỐC (chưa fine-tune) để so sánh. Cần đã pull base model trước."""
    return chat_service.run_chat(req, OLLAMA_BASE_LLM_MODEL)

@router.post("/llm-extended", response_model=AgentState)
def chat_with_workflow(req: ChatRequest) -> AgentState:
    """Chat với model FINE-TUNE kèm workflow Multi Agent."""
    return invoke_service.invoke(req.message, "mock_thread_id")