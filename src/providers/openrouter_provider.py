"""
OpenRouter client provider — thay Gemini cho supervisor + judge.

Mô phỏng ĐÚNG bề mặt của google-genai client mà 2 agent đang dùng:
    client.models.generate_content(model=..., contents=..., config=...)
    -> resp.parsed (pydantic đã validate) / resp.text (chuỗi thô)

Nhờ giữ nguyên bề mặt này:
  - supervisor.py / supervisor_judge.py chỉ đổi chỗ lấy client, không đổi luồng;
  - tests/test_supervisor_judge.py dựng _FakeClient theo bề mặt cũ vẫn chạy được;
  - nhánh except -> fallback trong 2 agent giữ nguyên, nên bộ dò fallback ở
    data/eval/run_full_system.py vẫn bắt được sự cố như trước.

Mirror kiểu providers/gemini_provider.py: 1 class giữ cache, expose singleton.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger("rag-service.openrouter")

OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "180"))


@dataclass
class GenConfig:
    """Thay cho types.GenerateContentConfig của google-genai (chỉ giữ field đang dùng)."""
    system_instruction: str = ""
    temperature: float = 0.0
    response_schema: Optional[type[BaseModel]] = None


class _Resp:
    """Bắt chước GenerateContentResponse: chỉ cần .parsed và .text."""

    def __init__(self, text: str, parsed: Optional[BaseModel]):
        self.text = text
        self.parsed = parsed


def _extract_json(raw: str) -> Optional[dict]:
    """Lấy JSON từ output model. Chịu được ```json fence và văn xuôi bao quanh."""
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Model đôi khi kèm lời dẫn trước JSON -> lấy khối ngoặc nhọn cân bằng đầu tiên.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


class _Models:
    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate_content(self, model: str, contents: str, config: Any = None) -> _Resp:
        system = getattr(config, "system_instruction", "") or ""
        temperature = getattr(config, "temperature", 0.0) or 0.0
        schema_cls = getattr(config, "response_schema", None)

        payload: dict = {
            "model": model,
            "temperature": temperature,
            "messages": [],
        }

        if schema_cls is not None:
            schema = schema_cls.model_json_schema()
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_cls.__name__, "strict": True, "schema": schema},
            }
            # Nhắc lại schema trong system prompt: model free trên OpenRouter không phải
            # provider nào cũng tôn trọng response_format, còn prompt thì luôn có tác dụng.
            system = (f"{system}\n\nChỉ trả về DUY NHẤT một JSON hợp lệ theo schema sau, "
                      f"không kèm giải thích, không kèm ```:\n"
                      f"{json.dumps(schema, ensure_ascii=False)}").strip()

        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": contents})

        r = httpx.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {self._api_key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=OPENROUTER_TIMEOUT,
        )
        # Lỗi HTTP (401/402/404/429...) -> raise, để nhánh except của agent chạy fallback
        # đúng như thời Gemini; run_full_system.py nhờ đó vẫn bắt được và dừng.
        # Kèm BODY vào message: OpenRouter để lý do thật ở body (vd 404 = model slug sai),
        # raise_for_status() trần chỉ nói "404 Not Found" -> debug mù.
        if r.status_code >= 400:
            raise RuntimeError(f"OpenRouter HTTP {r.status_code} (model={model}): {r.text[:400]}")
        data = r.json()

        # OpenRouter trả lỗi cấp ứng dụng kèm HTTP 200 -> phải tự bắt.
        if "error" in data and not data.get("choices"):
            raise RuntimeError(f"OpenRouter lỗi: {data['error']}")

        text = (data["choices"][0]["message"].get("content") or "").strip()

        parsed: Optional[BaseModel] = None
        if schema_cls is not None:
            obj = _extract_json(text)
            # [debug] key model THỰC SỰ emit. Field nằm trong `missing` sẽ bị Pydantic điền
            # default âm thầm (vd need_retrieval=True), nhìn ở trace không phân biệt được.
            logger.info("OpenRouter %s raw keys=%s missing=%s", schema_cls.__name__,
                        sorted(obj) if obj else None,
                        sorted(set(schema_cls.model_fields) - set(obj or {})))
            if obj is not None:
                try:
                    parsed = schema_cls.model_validate(obj)
                except Exception as e:
                    logger.warning("OpenRouter trả JSON không khớp schema %s: %s",
                                   schema_cls.__name__, e)
        return _Resp(text=text, parsed=parsed)


class _Client:
    def __init__(self, api_key: str):
        self.models = _Models(api_key)


class OpenRouterProvider:
    def __init__(self):
        self._client: Optional[_Client] = None

    def get_client(self) -> Optional[_Client]:
        """Lazy-init client; trả None nếu thiếu OPENROUTER_API_KEY."""
        if self._client is not None:
            return self._client
        api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            return None
        logger.info("Initializing OpenRouter client (model mặc định: %s)", OPENROUTER_MODEL)
        self._client = _Client(api_key)
        return self._client


# Singleton instance for easy import
openrouter_provider = OpenRouterProvider()
