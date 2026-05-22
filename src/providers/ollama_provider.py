import os
import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

logger = logging.getLogger("rag-service.ollama")
logging.basicConfig(level=logging.INFO)

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma4:e4b")

class OllamaProvider:
    def __init__(self):
        self._llm = None
        self._embeddings = None

    def get_llm(self) -> ChatOllama:
        if self._llm is None:
            logger.info(f"Initializing ChatOllama with model: {OLLAMA_LLM_MODEL} on URL: {OLLAMA_URL}")
            self._llm = ChatOllama(
                base_url=OLLAMA_URL,
                model=OLLAMA_LLM_MODEL,
                temperature=0.0
            )
        return self._llm

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
