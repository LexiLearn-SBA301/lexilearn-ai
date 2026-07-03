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
from agents.prepare_context import prepare_context


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


def mock_write_essay_node(state: AgentState) -> dict:
    """[MOCK] Node viết bài luận (Tool 3) — sau này thay bằng logic thật."""
    query = state.get("human_message", "")
    ctx = state.get("context")
    summary = ctx.summary if ctx else ""
    
    mock_essay = EssayDraft(
        title=f"Phân tích: {query}",
        full_text=f"[MOCK deep] Bài luận phân tích chi tiết cho: {query}\n\nContext summary: {summary}",
        word_count=30,
    )
    return {
        "essay": mock_essay,
        "current_stage": Stage.WRITE_ESSAY,
        "current_node": "write_essay",
    }
