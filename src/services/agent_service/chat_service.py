import logging

from providers.ollama_provider import ollama_provider
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.chat_schema import ChatRequest, ChatResponse
from exceptions import LLMServiceError
from typing import Any, List

logger = logging.getLogger("rag-service.services.chat")


class OllamaChatService:
    def _extract_text(self, content: Any) -> str:
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


    def run_chat(self, req: ChatRequest, model: str) -> ChatResponse:
        """Dựng messages -> gọi LLM `model` -> trả lời. Dùng chung cho các endpoint chat."""
        llm = ollama_provider.get_llm(model)

        messages: List[Any] = []
        if req.system:
            messages.append(SystemMessage(content=req.system))
        messages.append(HumanMessage(content=req.message))

        logger.info("Chat request (%d ký tự) -> model %s", len(req.message), model)
        try:
            response = llm.invoke(messages)
        except Exception as e:
            logger.error("Gọi LLM thất bại (model=%s): %s", model, e)
            raise LLMServiceError(str(e)) from e
        return ChatResponse(answer=self._extract_text(response.content), model=model)