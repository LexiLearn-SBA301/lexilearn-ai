import os
import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings


logger = logging.getLogger("rag-service.ollama")
logging.basicConfig(level=logging.INFO)

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
# Model fine-tune (GGUF) được host trên HuggingFace. Đồng đội chạy 1 lần:
#   ollama run hf.co/Tobi2904/qwen-finetuned-gguf
# rồi đặt FINE_TUNED_OLLAMA_LLM_MODEL=hf.co/Tobi2904/qwen-finetuned-gguf:latest trong .env.
# Provider chỉ gọi Ollama server qua HTTP — KHÔNG nạp adapter PEFT trực tiếp.
FINE_TUNED_OLLAMA_LLM_MODEL = os.getenv("FINE_TUNED_OLLAMA_LLM_MODEL", "qwen2.5:3b")
# Model GỐC (chưa fine-tune) — chỉ để so sánh output. Phải pull trước, ví dụ:
#   docker compose --profile compare up -d     (hoặc: ollama pull qwen2.5:3b)
OLLAMA_BASE_LLM_MODEL = os.getenv("OLLAMA_BASE_LLM_MODEL", "qwen2.5:3b")


class OllamaProvider:
    def __init__(self):
        self._llms = {}          # cache ChatOllama theo TÊN model -> phục vụ nhiều model
        self._embeddings = None

    def get_llm(self, model: str = FINE_TUNED_OLLAMA_LLM_MODEL) -> ChatOllama:
        """ChatOllama cho `model` (mặc định = bản fine-tune). Cache lại theo tên."""
        if model not in self._llms:
            logger.info(f"Initializing ChatOllama with model: {model} on URL: {OLLAMA_URL}")
            self._llms[model] = ChatOllama(
                base_url=OLLAMA_URL,
                model=model,
                temperature=0.0
            )
        return self._llms[model]

    def get_embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            logger.info(f"Initializing OllamaEmbeddings with model: {OLLAMA_EMBED_MODEL} on URL: {OLLAMA_URL}")
            self._embeddings = OllamaEmbeddings(
                base_url=OLLAMA_URL,
                model=OLLAMA_EMBED_MODEL
            )
        return self._embeddings

# Singleton instance for easy import
ollama_provider = OllamaProvider()
