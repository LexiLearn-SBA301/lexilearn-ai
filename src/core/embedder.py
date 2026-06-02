"""
Embedder Module — Production-ready embedding service for Vietnamese Literature RAG.

Uses LangChain's HuggingFaceEmbeddings to ensure compatibility with the
downstream LangChain ecosystem (MongoDBAtlasVectorSearch, Retrievers, etc.).

Responsibilities:
    - Load the embedding model exactly ONCE (singleton)
    - Generate embeddings for single texts and batches
    - Auto-detect optimal compute device (CUDA > MPS > CPU)
    - Provide a reusable, LangChain-compatible embedding service

This module MUST NOT:
    - Read PDFs or chunk documents
    - Connect to databases or perform retrieval
    - Call external LLM APIs
"""

import os
import json
import logging
import threading
from typing import List, Optional

logger = logging.getLogger("rag-service.embedder")


class Embedder:
    """
    Thread-safe singleton embedding service using LangChain HuggingFaceEmbeddings.

    Loads the model once on first instantiation and reuses the same
    instance across all subsequent calls. The underlying LangChain
    Embeddings object can be passed directly to MongoDBAtlasVectorSearch
    or any other LangChain component.
    """

    _instance: Optional["Embedder"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, config_path: Optional[str] = None) -> "Embedder":
        """
        Ensures only one Embedder instance exists (thread-safe singleton).
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the Embedder by loading config and model.
        Skips re-initialization if already loaded (singleton guard).
        """
        if Embedder._initialized:
            return

        with Embedder._lock:
            if Embedder._initialized:
                return

            # ── Load configuration ──────────────────────────────────
            if not config_path:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.normpath(
                    os.path.join(current_dir, "..", "config", "embedder_config.json")
                )

            with open(config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)

            self._model_name: str = self._config["model_name"]
            self._dimension: int = self._config["dimension"]
            self._batch_size: int = self._config["batch_size"]
            self._max_length: int = self._config["max_length"]
            self._normalize: bool = self._config["normalize_embeddings"]
            self._device_priority: List[str] = self._config["device_priority"]
            self._show_progress: bool = self._config["show_progress_bar"]

            # ── Detect device ───────────────────────────────────────
            self._device: str = self._detect_device()

            # ── Load LangChain embedding model ──────────────────────
            self._model = self._load_model()

            Embedder._initialized = True
            logger.info(
                "Embedder initialized — model=%s  device=%s  dim=%d  batch=%d",
                self._model_name,
                self._device,
                self._dimension,
                self._batch_size,
            )

    # ── Public API ──────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents. LangChain-compatible interface.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors (List[List[float]]).

        Raises:
            ValueError: If texts is empty or contains non-string items.
        """
        self._validate_input(texts)
        logger.debug("Embedding %d documents", len(texts))
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text. LangChain-compatible interface.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector (List[float]).

        Raises:
            ValueError: If text is empty or not a string.
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Input text must not be empty or whitespace-only")

        logger.debug("Embedding query: %s...", text[:50])
        return self._model.embed_query(text)

    @property
    def langchain_embeddings(self):
        """
        Returns the underlying LangChain Embeddings object.
        Use this to pass directly to MongoDBAtlasVectorSearch or other
        LangChain components that accept an Embeddings instance.
        """
        return self._model

    @property
    def model_name(self) -> str:
        """Name of the loaded embedding model."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension

    @property
    def device(self) -> str:
        """Compute device in use (cuda / mps / cpu)."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Batch size for encoding."""
        return self._batch_size

    @property
    def max_length(self) -> int:
        """Maximum token length per input text."""
        return self._max_length

    # ── Private helpers ─────────────────────────────────────────────

    def _detect_device(self) -> str:
        """
        Auto-detect the best available compute device.
        Priority order is defined in config: cuda > mps > cpu.
        """
        try:
            import torch
        except ImportError:
            logger.warning("torch not installed, defaulting to CPU")
            return "cpu"

        device_checkers = {
            "cuda": torch.cuda.is_available,
            "mps": lambda: (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "cpu": lambda: True,
        }

        for device in self._device_priority:
            checker = device_checkers.get(device)
            if checker and checker():
                logger.info("Device selected: %s", device)
                return device

        logger.warning("No preferred device available, falling back to CPU")
        return "cpu"

    def _load_model(self):
        """
        Load the HuggingFaceEmbeddings model via LangChain.
        """
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading model '%s' on device '%s' via LangChain...", self._model_name, self._device)

        model = HuggingFaceEmbeddings(
            model_name=self._model_name,
            model_kwargs={"device": self._device},
            encode_kwargs={
                "normalize_embeddings": self._normalize,
                "batch_size": self._batch_size,
                "show_progress_bar": self._show_progress,
            },
        )

        logger.info("Model loaded successfully via LangChain HuggingFaceEmbeddings")
        return model

    @staticmethod
    def _validate_input(texts: List[str]) -> None:
        """
        Validate that the input is a non-empty list of non-empty strings.

        Raises:
            ValueError: On invalid input.
        """
        if not isinstance(texts, list):
            raise ValueError(f"Expected list of strings, got {type(texts).__name__}")
        if len(texts) == 0:
            raise ValueError("Input list must not be empty")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise ValueError(f"Item at index {i} is {type(t).__name__}, expected str")
            if not t.strip():
                raise ValueError(f"Item at index {i} is empty or whitespace-only")

    # ── Class-level reset (for testing only) ────────────────────────

    @classmethod
    def _reset(cls) -> None:
        """
        Reset the singleton state. FOR TESTING ONLY.
        This allows re-initialization with different config in test suites.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.debug("Embedder singleton reset")
