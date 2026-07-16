"""
Finalize node — phễu CUỐI luồng: gom kết quả nhánh (factual | deep) thành
output chuẩn rồi mới END.

Tách riêng khỏi các node nhánh để:
  - ghi AIMessage vào `messages` (lịch sử hội thoại cho LLM) ĐÚNG 1 chỗ;
  - đóng `final_output` (FinalOutput) + `final_ai_response` (text thuần cho FE);
  - chốt `status=done`.

Node nhánh (factual/deep, kể cả khi thay mock bằng bản thật) chỉ lo SINH kết quả
domain; việc đóng gói hội thoại nằm ở đây -> thêm nhánh mới chỉ cần trỏ về finalize.
Cách lấy answer theo route ("Kiểu 1" dispatch) là phần DUY NHẤT phải thêm khi có
route mới; phần đóng gói bên dưới không đổi.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from state.agent_state import AgentState
from state.state_schema import EventEmitter, FinalOutput, Route, Stage, Verdict, safe_stream_writer

logger = logging.getLogger("rag-service.graph.finalize")


def _no_context_answer(state: AgentState) -> str:
    """Câu trả lời khi judge LOẠI context (kho không có tài liệu về tác phẩm được hỏi).

    Thà nói thẳng là chưa có tài liệu, còn hơn đưa context sai tác phẩm ra làm câu trả lời.
    """
    intent = state.get("intent")
    work = getattr(intent, "work_title", None) if intent else None
    if work:
        return (
            f"Xin lỗi, kho tài liệu hiện chưa có tác phẩm “{work}” nên mình chưa thể phân tích "
            "để tránh đưa thông tin sai. Bạn thử hỏi về một tác phẩm khác trong thư viện nhé."
        )
    return (
        "Xin lỗi, mình không tìm thấy tài liệu phù hợp trong kho để phân tích câu hỏi này, "
        "nên chưa thể trả lời để tránh đưa thông tin sai."
    )


def finalize(state: AgentState) -> dict:
    """Gom kết quả nhánh -> messages + final_ai_response + final_output. Trả state delta."""
    route = state.get("route")

    # Kiểu 1: dispatch theo route -> biết answer (+ citations/sources) nằm field nào.
    if route == Route.DEEP:
        essay = state.get("essay")
        ctx_verdict = (state.get("judges") or {}).get(Stage.PREPARE_CONTEXT.value)
        if essay:
            answer = essay.full_text
        elif ctx_verdict is not None and ctx_verdict.verdict == Verdict.REJECT:
            # judge LOẠI context -> graph cắt thẳng tới đây, KHÔNG chạy debate/essay.
            answer = _no_context_answer(state)
        else:
            ctx = state.get("context")
            answer = ctx.summary if ctx else "[Chưa có kết quả phân tích]"
        citations = essay.citations if essay else []
        sources = []
    else:  # FACTUAL (mặc định / fallback)
        factual = state.get("factual")
        answer = factual.answer if factual else ""
        citations = factual.citations if factual else []
        sources = factual.chunks_used if factual else []

    final_output = FinalOutput(
        answer=answer,
        route=route or Route.FACTUAL,
        citations=citations,
        sources=sources,
        finished_at=datetime.now(timezone.utc),
    )
    logger.info("Finalize route=%s len(answer)=%d", route, len(answer))

    emitter = EventEmitter(state, writer=safe_stream_writer())
    emitter.done(
        "finalize",
        content="Hoàn tất.",
        # answer = text câu trả lời CUỐI để FE hiển thị (khi Tool 3 có token-stream thì
        # đây là bản chốt trùng với text đã stream). chars/citations để FE hiện meta.
        payload={
            "route": (route or Route.FACTUAL).value,
            "answer": answer,
            "chars": len(answer),
            "citations": len(citations),
        },
    )
    # messages = lịch sử các lượt ĐÃ XONG. Nhập cặp của lượt này (human hỏi + AI đáp) vào đây.
    # Giữ sliding window ≤ 10 message (5 cặp): thêm cặp mới -> đẩy cặp cũ nhất ra (RemoveMessage
    # theo id) để checkpoint không phình vô hạn theo phiên chat.
    WINDOW = 10
    existing = state.get("messages", []) or []
    new_pair = [HumanMessage(content=state.get("human_message", "")), AIMessage(content=answer)]
    overflow = len(existing) + len(new_pair) - WINDOW
    removals = [RemoveMessage(id=m.id) for m in existing[:overflow]] if overflow > 0 else []
    return {
        "messages": removals + new_pair,   # add_messages: xóa cặp cũ nhất (nếu tràn) + nối cặp mới
        "final_ai_response": answer,
        "final_output": final_output,
        "current_stage": Stage.DONE,
        "current_node": "finalize",
        "status": "done",
        "events": emitter.milestones,
        "event_seq": emitter.seq,
    }