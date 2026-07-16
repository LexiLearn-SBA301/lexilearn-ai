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

YÊU CẦU SỬA LẠI BÀI ĐÃ VIẾT — luật này ĐÈ LÊN cách chọn route ở trên:
Nếu lịch sử hội thoại cho thấy trợ lý ĐÃ viết một bài phân tích/bài văn, và câu hỏi MỚI là
lời chê hoặc đòi sửa CHÍNH bài đó ("thân bài ngắn quá", "kết bài chưa sâu, viết lại", "thêm
dẫn chứng vào", "bỏ mở bài đi", "viết lại dài hơn"):
- route = "factual", KHÔNG phải "deep_analysis" — dù câu chữ nghe hệt như đòi phân tích sâu.
  Bài cũ đã có sẵn, việc cần làm là SỬA nó, không phải phân tích lại từ đầu.
- refine_instruction = MỆNH LỆNH cụ thể, tự đủ nghĩa, viết cho người thợ sửa bài đọc: nêu rõ
  SỬA PHẦN NÀO và SỬA THẾ NÀO. Ví dụ: "Viết lại phần thân bài dài hơn và phân tích sâu hơn,
  giữ nguyên mở bài và kết bài."
- Mọi câu hỏi KHÁC: refine_instruction = null.

Ngoài route, quyết định need_retrieval. Câu hỏi cần trả lời là "câu hỏi này ĐÃ KÈM SẴN
văn bản để phân tích chưa?", KHÔNG phải "yêu cầu này có cần thông tin mới không?".
- need_retrieval=false CHỈ trong 2 ca:
  (a) Lời chào/tán gẫu/câu hỏi meta không liên quan văn học ("xin chào", "bạn là ai",
      "cảm ơn"): không có gì để tra cứu.
  (b) Câu hỏi CHÉP NGUYÊN VĂN đoạn thơ/văn cần phân tích vào ngay trong câu hỏi và chỉ
      muốn phân tích chính đoạn đó: tra cứu thêm chỉ gây nhiễu.
- MỌI trường hợp còn lại: need_retrieval=true. Gồm cả khi câu hỏi chỉ NHẮC TÊN tác phẩm
  mà không chép văn bản vào, và khi người dùng chê/đòi sửa câu trả lời trước ("kết bài
  ngắn quá, viết lại", "phân tích sâu hơn", "thêm dẫn chứng") — những câu này KHÔNG chứa
  văn bản tác phẩm, nên vẫn phải lấy lại từ kho mới viết đúng được.

Và quyết định on_topic (câu hỏi có thuộc phạm vi hỗ trợ không):
- on_topic=false khi câu hỏi NGOÀI văn học Việt Nam: lập trình/code, toán, hình học,
  khoa học, thời tiết, đời sống chung... (vd "java là ngôn ngữ lập trình đúng không").
- on_topic=true khi là lời chào/xã giao HOẶC hỏi–đáp/phân tích về văn học.

Nếu có "Lịch sử hội thoại gần đây": DÙNG nó để hiểu câu hỏi MỚI. Giải nghĩa đại từ/tham chiếu
("tác phẩm đó", "nhân vật ấy", "phân tích tiếp") thành tên tác phẩm/nhân vật/tác giả CỤ THỂ,
và điền work_title/author cho đúng theo ngữ cảnh trước đó.

retrieval_query: viết lại câu hỏi MỚI thành 1 câu tra cứu ĐỘC LẬP, tự đủ nghĩa (thay đại từ
bằng thực thể cụ thể dựa vào lịch sử). Nếu câu hỏi đã tự đủ nghĩa thì retrieval_query = chính
câu hỏi đó. Ví dụ: lịch sử nói về "Chí Phèo" + câu mới "phân tích tác phẩm đó" ->
retrieval_query = "Phân tích tác phẩm Chí Phèo của Nam Cao".

QUY TẮC NGÔN NGỮ CHO `reasoning` (BẮT BUỘC):
`reasoning` hiển thị THẲNG cho người dùng cuối là học sinh/giáo viên. Viết 1–2 câu như đang nói
với học sinh về việc bạn sắp làm cho họ, KHÔNG phải giải trình nội bộ.
CẤM nhắc tên kỹ thuật trong `reasoning`: 'factual', 'deep_analysis', 'route', 'định tuyến',
'need_retrieval', 'refine_instruction', 'Mode A', 'schema', 'pipeline', 'theo luật', 'phân loại'.
VÍ DỤ SAI: "Theo luật, đây là yêu cầu 'factual' để sửa bài, không phải 'deep_analysis' từ đầu."
VÍ DỤ ĐÚNG: "Bạn muốn phần thân bài dài và sâu hơn — mình sẽ tra lại tác phẩm để lấy thêm dẫn
chứng rồi viết lại phần đó, các phần khác giữ nguyên."

Trả về JSON đúng schema: route, confidence (0..1), need_retrieval, on_topic, retrieval_query,
refine_instruction, work_title, author, detected_entities, requested_dimensions, reasoning
(nói ngắn gọn cho học sinh biết bạn sắp làm gì — theo đúng quy tắc ngôn ngữ ở trên).
"""


class _Decision(BaseModel):
    """Phần Gemini sinh ra; raw_query & analyzed_at do server gắn vào sau."""
    route: Route
    confidence: float = 0.0
    need_retrieval: bool = True
    on_topic: bool = True
    retrieval_query: str = ""                  # câu tra cứu độc lập đã resolve tham chiếu từ lịch sử
    refine_instruction: Optional[str] = None   # mệnh lệnh sửa bài lượt trước; null nếu không phải ca sửa bài
    work_title: Optional[str] = None
    author: Optional[str] = None
    detected_entities: list[str] = Field(default_factory=list)
    requested_dimensions: list[CriticRole] = Field(default_factory=list)
    reasoning: str = ""


def _render_history(messages: list, max_assistant_chars: int = 500) -> str:
    """Rút gọn lịch sử thành text cho Gemini resolve tham chiếu. Cắt câu trả lời của trợ lý
    (có thể là bài văn dài) để prompt không phình — chỉ cần đủ ngữ cảnh nhận diện tác phẩm/nhân vật.
    `messages` là list BaseMessage của state (đã trừ câu hiện tại); BE đã giới hạn số lượt gửi lên."""
    if not messages:
        return ""
    lines = []
    for m in messages:
        role = getattr(m, "type", None)
        content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = str(content)
        content = content.strip() # tỉa khoảng trắng/\n dính ở mép ngoài cùng
        if not content:
            continue
        if role == "ai":
            if len(content) > max_assistant_chars:
                content = content[:max_assistant_chars] + "…"
            lines.append(f"Trợ lý: {content}")
        elif role == "human":
            lines.append(f"Người dùng: {content}")
    return "\n".join(lines)


def _classify(query: str, history_text: str = "") -> _Decision:
    """Gọi Gemini phân loại route. Mọi sự cố -> fallback route=factual."""
    client = gemini_provider.get_client()
    if client is None:
        logger.warning("Thiếu GEMINI_API_KEY -> fallback route=factual.")
        return _Decision(route=Route.FACTUAL,
                         reasoning="[fallback] chưa cấu hình GEMINI_API_KEY")
    if history_text:
        contents = f"Lịch sử hội thoại gần đây:\n{history_text}\n\n---\nCâu hỏi mới của người dùng: {query}"
    else:
        contents = query
    try:
        from google.genai import types
        resp = client.models.generate_content(
            model=SUPERVISOR_MODEL,
            contents=contents,
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
    # messages = lịch sử các lượt ĐÃ XONG (câu hiện tại nằm ở human_message, chưa vào messages).
    messages = state.get("messages", []) or []
    history_text = _render_history(messages)
    d = _classify(query, history_text)
    # Ca sửa bài lượt trước -> Mode A thi hành mệnh lệnh (factual_node). Ép route=FACTUAL kể cả
    # khi Gemini lỡ chọn deep_analysis: câu chê bài ("thân bài chưa sâu") nghe hệt yêu cầu phân
    # tích sâu, mà deep sẽ bỏ qua instruction rồi viết lại bài MỚI từ đầu, vứt bài cũ đi.
    refine_instruction = (d.refine_instruction or "").strip() or None
    # Câu ngoài văn học -> luôn về factual để trả lời từ chối cố định (factual_node),
    # kể cả khi Gemini lỡ định tuyến sang deep_analysis.
    route = Route.FACTUAL if (not d.on_topic or refine_instruction) else d.route
    # retrieval_query đã resolve; rỗng (fallback/không có lịch sử) -> dùng chính câu hỏi gốc.
    retrieval_query = (d.retrieval_query or "").strip() or query
    intent = IntentAnalysis(
        raw_query=query,
        retrieval_query=retrieval_query,
        route=route,
        confidence=d.confidence,
        need_retrieval=d.need_retrieval,
        on_topic=d.on_topic,
        refine_instruction=refine_instruction,
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