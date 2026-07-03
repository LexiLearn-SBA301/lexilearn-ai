"""
Gemini client provider — lazy singleton, tách khỏi agents/supervisor.py để
agents/supervisor_judge.py (Supervisor judge) dùng lại CHUNG 1 client, không
tự khởi tạo genai.Client riêng.

Mirror kiểu providers/ollama_provider.py: 1 class giữ cache, expose singleton.
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag-service.gemini")


class GeminiProvider:
    def __init__(self):
        self._client = None

    def get_client(self):
        """Lazy-init genai.Client; trả None nếu thiếu GEMINI_API_KEY."""
        if self._client is not None:
            return self._client
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            return None
        from google import genai
        logger.info("Initializing Gemini client")
        self._client = genai.Client(api_key=api_key)
        return self._client


# Singleton instance for easy import
gemini_provider = GeminiProvider()
