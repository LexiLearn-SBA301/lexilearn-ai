"""
Supervisor node — phân tích intent câu hỏi rồi quyết định route:
  - Route.FACTUAL : câu hỏi TRA CỨU ngắn, cần dẫn chứng (ai viết, năm nào, tóm tắt ngắn)
  - Route.DEEP    : yêu cầu PHÂN TÍCH / CẢM NHẬN / NGHỊ LUẬN sâu

Dùng Gemini (google-genai) structured output -> Pydantic, mirror cách
core/pdf_reader.py gọi Gemini. Thiếu GEMINI_API_KEY hoặc lỗi gọi -> fallback
route=FACTUAL để graph vẫn chạy.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from state.agent_state import AgentState
from state.state_schema import CriticRole, IntentAnalysis, Route, Stage

load_dotenv()

logger = logging.getLogger("rag-service.graph.supervisor")

SUPERVISOR_MODEL = os.getenv("GEMINI_SUPERVISOR_MODEL", "gemini-2.5-flash")

_SYSTEM_PROMPT = """Bạn là Supervisor điều phối của hệ thống phân tích văn học.
Nhiệm vụ: đọc câu hỏi của học sinh và phân loại thành 1 trong 2 route:

- "factual": câu hỏi TRA CỨU ngắn, có đáp án dựa trên dẫn chứng trực tiếp.
  Ví dụ: "Vợ Nhặt của ai?", "Truyện Kiều sáng tác năm nào?", "Tóm tắt ngắn đoạn trích".
- "deep_analysis": yêu cầu PHÂN TÍCH / CẢM NHẬN / NGHỊ LUẬN sâu, nhiều góc nhìn.
  Ví dụ: "Phân tích tâm lý nhân vật Tràng", "Cảm nhận bi kịch của Chí Phèo",
  "So sánh hình tượng người phụ nữ trong hai tác phẩm".

Trả về JSON đúng schema: route, confidence (0..1), work_title, author,
detected_entities, requested_dimensions, reasoning (giải thích ngắn vì sao chọn route).
"""


class _Decision(BaseModel):
    """Phần Gemini sinh ra; raw_query & analyzed_at do server gắn vào sau."""
    route: Route
    confidence: float = 0.0
    work_title: Optional[str] = None
    author: Optional[str] = None
    detected_entities: list[str] = Field(default_factory=list)
    requested_dimensions: list[CriticRole] = Field(default_factory=list)
    reasoning: str = ""


_client = None


def _get_client():
    """Lazy-init genai.Client; trả None nếu thiếu API key."""
    global _client # trỏ tới biến toàn cục bên ngoài để sài (khác java)
    if _client is not None:
        return _client
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None
    from google import genai
    _client = genai.Client(api_key=api_key)
    return _client

def _classify(query: str) -> _Decision:
    """Gọi Gemini phân loại route. Mọi sự cố -> fallback route=factual."""
    client = _get_client()
    if client is None:
        logger.warning("Thiếu GEMINI_API_KEY -> fallback route=factual.")
        return _Decision(route=Route.FACTUAL,
                         reasoning="[fallback] chưa cấu hình GEMINI_API_KEY")
    try:
        from google.genai import types
        resp = client.models.generate_content(
            model=SUPERVISOR_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=_Decision,  # SDK biên dịch thành JSON -> genai sẽ ép gemini điền các json này
            ),
        )
        decision = resp.parsed
        if isinstance(decision, _Decision):
            return decision
        if resp.text:
            return _Decision.model_validate_json(resp.text)
        raise ValueError("Gemini trả về rỗng")
    except Exception as e:
        logger.warning("Supervisor gọi Gemini lỗi (%s) -> fallback route=factual.", e)
        return _Decision(route=Route.FACTUAL,
                         reasoning=f"[fallback] lỗi gọi Gemini: {e}")


def supervisor(state: AgentState) -> dict:
    """Node supervisor: phân tích intent + chọn route. Trả về state delta."""
    query = state.get("human_message", "")
    d = _classify(query)
    intent = IntentAnalysis(
        raw_query=query,
        route=d.route,
        confidence=d.confidence,
        work_title=d.work_title,
        author=d.author,
        detected_entities=d.detected_entities,
        requested_dimensions=d.requested_dimensions,
        reasoning=d.reasoning,
        analyzed_at=datetime.now(timezone.utc),
    )
    logger.info("Supervisor route=%s conf=%.2f", d.route, d.confidence)
    return {
        "intent": intent,
        "route": d.route,
        "current_stage": Stage.INTENT,
        "current_node": "supervisor:intent",
        "status": "running",
    }