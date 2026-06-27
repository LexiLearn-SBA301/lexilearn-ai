"""
Mock nodes — stub TẠM để graph chạy end-to-end trong lúc node thật chưa xong.

- factual_node: bản thật do Nhật làm (gọi RAGService). Mock giữ ĐÚNG interface
  (state -> dict delta) để sau cắm node thật vào KHÔNG phải sửa graph.
- deep_node: đại diện cho cả deep pipeline (prepare_context -> debate -> essay),
  sẽ tách thành nhiều node sau.
"""
from __future__ import annotations

from state.agent_state import AgentState
from state.state_schema import EssayDraft, FactualResult, Stage


def factual_node(state: AgentState) -> dict:
    """[MOCK] Mode A factual — sau thay bằng RAGService thật của đồng đội."""
    query = state.get("human_message", "")
    return {
        "factual": FactualResult(
            answer=f"[MOCK factual] trả lời cho: {query}",
            model="mock",
        ),
        "current_stage": Stage.FACTUAL,
        "current_node": "factual",
    }


def deep_node(state: AgentState) -> dict:
    """[MOCK] Deep pipeline (context -> debate -> essay) gộp 1 node tạm."""
    query = state.get("human_message", "")
    return {
        "essay": EssayDraft(
            title="[MOCK] Bài phân tích",
            full_text=f"[MOCK deep] phân tích sâu cho: {query}",
        ),
        "current_stage": Stage.WRITE_ESSAY,
        "current_node": "deep",
    }