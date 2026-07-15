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

from pydantic import BaseModel, Field

from providers.gemini_provider import gemini_provider
from state.agent_state import AgentState
from state.state_schema import (
    CriticRole,
    EventEmitter,
    IntentAnalysis,
    Route,
    Stage,
    safe_stream_writer,
)

logger = logging.getLogger("rag-service.graph.supervisor")

SUPERVISOR_MODEL = os.getenv("GEMINI_SUPERVISOR_MODEL", "gemini-2.5-flash")

_SYSTEM_PROMPT = """Bạn là Supervisor điều phối của hệ thống phân tích văn học.
Nhiệm vụ: đọc câu hỏi của học sinh và phân loại thành 1 trong 2 route:

- "factual": câu hỏi TRA CỨU ngắn, có đáp án dựa trên dẫn chứng trực tiếp.
  Ví dụ: "Vợ Nhặt của ai?", "Truyện Kiều sáng tác năm nào?", "Tóm tắt ngắn đoạn trích".
- "deep_analysis": yêu cầu PHÂN TÍCH / CẢM NHẬN / NGHỊ LUẬN sâu, nhiều góc nhìn.
  Ví dụ: "Phân tích tâm lý nhân vật Tràng", "Cảm nhận bi kịch của Chí Phèo",
  "So sánh hình tượng người phụ nữ trong hai tác phẩm".

Ngoài route, quyết định need_retrieval (có cần tra cứu kho tài liệu không):
- need_retrieval=false khi:
  (a) Lời chào/tán gẫu/câu hỏi meta không liên quan văn học ("xin chào", "bạn là ai",
      "cảm ơn"): không có gì để tra cứu.
  (b) Người dùng đã DÁN SẴN đoạn thơ/văn bản ngay trong câu hỏi và chỉ muốn phân tích
      chính đoạn đó: tra cứu thêm chỉ gây nhiễu.
- need_retrieval=true khi câu hỏi nói VỀ một tác phẩm nhưng KHÔNG kèm sẵn văn bản
  -> cần lấy dẫn chứng từ kho tài liệu.

Và quyết định on_topic (câu hỏi có thuộc phạm vi hỗ trợ không):
- on_topic=false khi câu hỏi NGOÀI văn học Việt Nam: lập trình/code, toán, hình học,
  khoa học, thời tiết, đời sống chung... (vd "java là ngôn ngữ lập trình đúng không").
- on_topic=true khi là lời chào/xã giao HOẶC hỏi–đáp/phân tích về văn học.

Trả về JSON đúng schema: route, confidence (0..1), need_retrieval, on_topic, work_title,
author, detected_entities, requested_dimensions, reasoning (giải thích ngắn vì sao chọn route).
"""


class _Decision(BaseModel):
    """Phần Gemini sinh ra; raw_query & analyzed_at do server gắn vào sau."""
    route: Route
    confidence: float = 0.0
    need_retrieval: bool = True
    on_topic: bool = True
    work_title: Optional[str] = None
    author: Optional[str] = None
    detected_entities: list[str] = Field(default_factory=list)
    requested_dimensions: list[CriticRole] = Field(default_factory=list)
    reasoning: str = ""


def _classify(query: str) -> _Decision:
    """Gọi Gemini phân loại route. Mọi sự cố -> fallback route=factual."""
    client = gemini_provider.get_client()
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
                         reasoning="Hệ thống đang bận, tạm xử lý câu hỏi theo hướng mặc định.")


def supervisor(state: AgentState) -> dict:
    """Node supervisor: phân tích intent + chọn route. Trả về state delta."""
    query = state.get("human_message", "")
    d = _classify(query)
    # Câu ngoài văn học -> luôn về factual để trả lời từ chối cố định (factual_node),
    # kể cả khi Gemini lỡ định tuyến sang deep_analysis.
    route = Route.FACTUAL if not d.on_topic else d.route
    intent = IntentAnalysis(
        raw_query=query,
        route=route,
        confidence=d.confidence,
        need_retrieval=d.need_retrieval,
        on_topic=d.on_topic,
        work_title=d.work_title,
        author=d.author,
        detected_entities=d.detected_entities,
        requested_dimensions=d.requested_dimensions,
        reasoning=d.reasoning,
        analyzed_at=datetime.now(timezone.utc),
    )
    logger.info("Supervisor route=%s conf=%.2f on_topic=%s", route, d.confidence, d.on_topic)

    emitter = EventEmitter(state, writer=safe_stream_writer())
    emitter.intent(
        "supervisor:intent",
        intent.reasoning or f"Phân loại câu hỏi: {route.value}",
        payload={
            "work_title": intent.work_title,
            "author": intent.author,
            "route": route.value,
            "confidence": d.confidence,
            "detected_entities": intent.detected_entities,
        },
    )
    emitter.route("supervisor:intent", route.value)
    return {
        "intent": intent,
        "route": route,
        "current_stage": Stage.INTENT,
        "current_node": "supervisor:intent",
        "status": "running",
        "events": emitter.milestones,
        "event_seq": emitter.seq,
    }