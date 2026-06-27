"""
Chat API router — endpoint thử nhanh LLM (gọi THẲNG Ollama, KHÔNG qua RAG).

Mục đích: kiểm tra model fine-tune trả lời ra sao mà không cần dữ liệu trong Mongo.
Phần truy hồi ngữ cảnh (RAG) nằm ở RAGService; router này cố tình giữ tối giản.
"""

from fastapi import APIRouter, Depends
from schemas.chat_schema import ChatRequest, ChatResponse
from providers.ollama_provider import FINE_TUNED_OLLAMA_LLM_MODEL, OLLAMA_BASE_LLM_MODEL
from services.agent_service.chat_service import OllamaChatService
from services.agent_service.workflow_service import WorkflowService
from api.dependencies import get_workflow, get_chat_svc
from state.agent_state import AgentState

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/only-llm", response_model=ChatResponse)
def chat_finetuned(req: ChatRequest,  chat_service: OllamaChatService = Depends(get_chat_svc)) -> ChatResponse:
    """Chat với model FINE-TUNE (chưa có RAG)."""
    return chat_service.run_chat(req, FINE_TUNED_OLLAMA_LLM_MODEL)


@router.post("/base-llm", response_model=ChatResponse)
def chat_base(req: ChatRequest,  chat_service: OllamaChatService = Depends(get_chat_svc)) -> ChatResponse:
    """Chat với model GỐC (chưa fine-tune) để so sánh. Cần đã pull base model trước."""
    return chat_service.run_chat(req, OLLAMA_BASE_LLM_MODEL)

@router.post("/llm-extended", response_model=AgentState)
def chat_with_workflow(req: ChatRequest, wf: WorkflowService = Depends(get_workflow)) -> AgentState:
    """Chat với model FINE-TUNE kèm workflow Multi Agent."""
    return wf.invoke(req.message, "mock_thread_id")
#Depends() ==  @Autowired