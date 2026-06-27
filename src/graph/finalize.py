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

from langchain_core.messages import AIMessage

from state.agent_state import AgentState
from state.state_schema import FinalOutput, Route, Stage

logger = logging.getLogger("rag-service.graph.finalize")


def finalize(state: AgentState) -> dict:
    """Gom kết quả nhánh -> messages + final_ai_response + final_output. Trả state delta."""
    route = state.get("route")

    # Kiểu 1: dispatch theo route -> biết answer (+ citations/sources) nằm field nào.
    if route == Route.DEEP:
        essay = state.get("essay")
        answer = essay.full_text if essay else ""
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
    return {
        "messages": [AIMessage(content=answer)],   # add_messages nối vào lịch sử
        "final_ai_response": answer,
        "final_output": final_output,
        "current_stage": Stage.DONE,
        "current_node": "finalize",
        "status": "done",
    }