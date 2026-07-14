"""State KHÔNG được rò rỉ giữa 2 lượt chat cùng thread_id.

Graph chạy trên checkpoint cũ của thread, nên input của init_state phải XÓA được kết
quả lượt trước. Nếu không: retry_counts của câu hỏi trước làm cạn quota retry của câu
sau, và best_attempts của câu trước có thể bị judge trả về làm câu trả lời cho câu sau
(supervisor_judge dùng best["output"] khi hết lượt retry).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from state.agent_state import AgentState, init_state
from state.state_schema import (
    EventType,
    StreamEvent,
    append_events,
    merge_dict,
    take_last,
)


def test_take_last_none_xoa_field():
    assert take_last("cũ", "mới") == "mới"
    assert take_last("cũ", None) is None          # None = xóa, KHÔNG giữ giá trị cũ


def test_merge_dict_none_xoa_sach():
    assert merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dict({"a": 1}, {"a": 9}) == {"a": 9}
    assert merge_dict({"a": 1}, None) == {}       # None = xóa sạch


def test_append_events_none_xoa_sach():
    ev = StreamEvent(seq=1, type=EventType.STATUS, node="n")
    assert append_events([ev], [ev]) == [ev, ev]  # trong 1 lượt: nối tiếp
    assert append_events([ev], None) == []        # None = xóa sạch


def _stub_graph():
    """Graph tối giản mô phỏng 1 lượt chat có retry: ghi vào đúng các field tích lũy."""
    captured: dict = {}

    def probe(state: AgentState) -> dict:
        # chụp state NGAY ĐẦU lượt (sau khi input đã áp lên checkpoint cũ)
        captured.clear()
        captured.update(state)
        return {}

    def run_turn(state: AgentState) -> dict:
        return {
            "context": "CONTEXT-" + state["human_message"],
            "retry_counts": {"prepare_context": 2},          # xài hết quota retry
            "best_attempts": {"write_essay": {"output": "BÀI-" + state["human_message"],
                                              "score": 0.9}},
            "events": [StreamEvent(seq=1, type=EventType.STATUS, node="run_turn")],
        }

    g = StateGraph(AgentState)
    g.add_node("probe", probe)
    g.add_node("run_turn", run_turn)
    g.add_edge(START, "probe")
    g.add_edge("probe", "run_turn")
    g.add_edge("run_turn", END)
    return g.compile(checkpointer=MemorySaver()), captured


def test_luot_sau_khong_dinh_state_luot_truoc():
    app, captured = _stub_graph()
    cfg = {"configurable": {"thread_id": "t1"}}

    async def main():
        await app.ainvoke(init_state("Chí Phèo", "t1", "r1"), config=cfg)
        return await app.ainvoke(init_state("Vợ Nhặt", "t1", "r2"), config=cfg)

    out = asyncio.run(main())

    # Đầu lượt 2: mọi field tích lũy của lượt 1 phải sạch.
    assert not captured["retry_counts"], "quota retry lượt trước dính sang lượt sau"
    assert not captured["best_attempts"], "bài luận câu hỏi trước dính sang câu hỏi sau"
    assert not captured["events"]
    assert captured["context"] is None

    # ...nhưng lịch sử hội thoại thì PHẢI giữ (đó là trí nhớ nhiều lượt).
    assert [m.content for m in out["messages"]] == ["Chí Phèo", "Vợ Nhặt"]
