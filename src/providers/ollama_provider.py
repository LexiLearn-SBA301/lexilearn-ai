import os
import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings


logger = logging.getLogger("rag-service.ollama")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
# Timeout (giây) cho MỖI request tới Ollama. Không set -> httpx chờ vô hạn: nếu Ollama
# treo (đang load model / hết VRAM), node kẹt mãi, stream im lặng và FE quay vòng vô tận.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))
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

    def get_llm(self, model: str = FINE_TUNED_OLLAMA_LLM_MODEL, temperature: float = 0.0) -> ChatOllama:
        """ChatOllama cho `model` (mặc định = bản fine-tune). Cache theo (model, temperature)."""
        key = (model, temperature)
        if key not in self._llms:
            logger.info(f"Initializing ChatOllama with model: {model} temperature={temperature} on URL: {OLLAMA_URL}")
            self._llms[key] = ChatOllama(
                base_url=OLLAMA_URL,
                model=model,
                temperature=temperature,
                # Default runtime của Ollama chỉ cấp num_ctx=4096. Lượt write_essay đã ~3.8K token
                # (input 3.3K + output) -> sát trần; khi retry kèm feedback / ngữ liệu dài hơn sẽ
                # tràn và Ollama cắt IM LẶNG phần đầu prompt (system + ngữ liệu) -> model bịa dẫn
                # chứng. Nới 8192 cho dư địa. num_predict=-1: bỏ trần độ dài output, để num_ctx là
                # ràng buộc duy nhất (phục vụ bài phân tích dài hơn).
                num_ctx=8192,
                num_predict=-1,
                client_kwargs={"timeout": OLLAMA_TIMEOUT}
            )
        return self._llms[key]

    def get_embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            logger.info(f"Initializing OllamaEmbeddings with model: {OLLAMA_EMBED_MODEL} on URL: {OLLAMA_URL}")
            self._embeddings = OllamaEmbeddings(
                base_url=OLLAMA_URL,
                model=OLLAMA_EMBED_MODEL,
                client_kwargs={"timeout": OLLAMA_TIMEOUT}
            )
        return self._embeddings

# Singleton instance for easy import
ollama_provider = OllamaProvider()
