"""
Test Tool 2 (critics_debate) với LLM GIẢ -> không cần Ollama.

Tool 2 KHÔNG retrieve: đoạn văn bản chung được truyền vào (mô phỏng Tool 1).
Kiểm: 4 critic R1 đọc chung chunks -> bulletin 4 entry -> 4 critic R2 đều có
rebuttal và không tự phản biện mình -> node public ráp DebateState, total=8.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.messages import AIMessage

from agents.critics_debate import (
    CRITIC_ORDER,
    _resolve_arg_id,
    _speak_r1,
    build_debate_subgraph,
    critics_debate,
)
from agents.debate_schemas import CriticR1Out, CriticR2Out, _ArgIn, _RebuttalIn
from state.state_schema import CriticRole, DebateState, PreparedContext, SourceChunk


class _FakeStructured:
    def __init__(self, schema):
        self.schema = schema

    async def ainvoke(self, messages):
        if self.schema is CriticR1Out:
            return CriticR1Out(
                thesis="Luận đề mock",
                arguments=[
                    _ArgIn(point="Điểm 1", support="Dựa trên đoạn c1"),
                    _ArgIn(point="Điểm 2", support="Dựa trên đoạn c2"),
                ],
            )
        if self.schema is CriticR2Out:
            # nhắm tới tất cả role; assembly tự bỏ self -> mỗi critic còn 3 rebuttal.
            # target_point=1 -> server map thành arg_id "{target}-a1" (a1 luôn tồn tại).
            return CriticR2Out(rebuttals=[
                _RebuttalIn(target_critic=r, target_point=1, stance="disagree",
                            reason=f"Phản biện {r.value}")
                for r in CRITIC_ORDER
            ])
        raise AssertionError(f"schema không mong đợi: {self.schema}")


class FakeLLM:
    def with_structured_output(self, schema):
        return _FakeStructured(schema)

    async def ainvoke(self, messages):
        return AIMessage(content="raw fallback")


def _chunks():
    return [
        SourceChunk(chunk_id="c1", text="Đoạn văn mẫu 1 của tác phẩm."),
        SourceChunk(chunk_id="c2", text="Đoạn văn mẫu 2 của tác phẩm."),
    ]


def _sub_input():
    return {
        "work_title": "Tác phẩm X",
        "author": "Tác giả Y",
        "context_summary": "Tóm tắt ngắn.",
        "chunks": _chunks(),
        "round1": {}, "round2": {}, "bulletin": [],
        "consensus_points": [], "contested_points": [],
    }


def test_subgraph_full_debate():
    app = build_debate_subgraph(FakeLLM())
    out = asyncio.run(app.ainvoke(_sub_input()))

    # 4 critic R1, đều parse OK
    assert set(out["round1"].keys()) == set(CRITIC_ORDER)
    for turn in out["round1"].values():
        assert turn.round == 1
        assert turn.parsed_ok is True
        assert turn.thesis
        assert len(turn.arguments) == 2

    # Bulletin đủ 4 entry
    assert len(out["bulletin"]) == 4

    # 4 critic R2, mỗi critic có rebuttal và KHÔNG tự phản biện mình
    assert set(out["round2"].keys()) == set(CRITIC_ORDER)
    for role, turn in out["round2"].items():
        assert turn.round == 2
        assert turn.bulletin_seen is True
        assert turn.rebuttals
        for reb in turn.rebuttals:
            assert reb.target_critic != role
            # target_point=1 -> id THẬT "{target}-a1" (không phải id bịa)
            assert reb.target_arg_id == f"{reb.target_critic.value}-a1"


def test_resolve_arg_id_chong_bia():
    valid = {"tam_ly-a1", "tam_ly-a2", "lich_su-a1"}
    assert _resolve_arg_id(CriticRole.TAM_LY, 2, valid) == "tam_ly-a2"   # hợp lệ
    assert _resolve_arg_id(CriticRole.TAM_LY, 3, valid) is None          # ngoài range -> None
    assert _resolve_arg_id(CriticRole.LICH_SU, 2, valid) is None         # critic có nhưng point vượt
    assert _resolve_arg_id(CriticRole.TAM_LY, None, valid) is None       # model không chọn
    assert _resolve_arg_id(CriticRole.TAM_LY, 0, valid) is None          # số lạ


def test_public_node_reads_context_and_builds_debatestate():
    app = build_debate_subgraph(FakeLLM())
    state = {
        "intent": None,
        "context": PreparedContext(summary="Tóm tắt", chunks=_chunks()),
    }

    delta = asyncio.run(critics_debate(state, subgraph=app))

    debate = delta["debate"]
    assert isinstance(debate, DebateState)
    assert debate.total_invocations == 8            # 4 R1 + 4 R2
    assert len(debate.round1) == 4
    assert len(debate.round2) == 4
    assert delta["current_node"] == "critics_debate"


def test_speak_r1_retries_until_min_arguments():
    """R1 trả <2 luận điểm -> _speak_r1 nhắc lại 1 lần để đạt tối thiểu."""
    class _RetryLLM:
        def __init__(self):
            self.calls = 0

        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            k = 1 if self.calls == 1 else 2      # lần đầu 1 luận điểm, lần 2 đủ 2
            return CriticR1Out(thesis="th", arguments=[_ArgIn(point=f"p{i}") for i in range(k)])

    llm = _RetryLLM()
    turn = asyncio.run(_speak_r1(CriticRole.TAM_LY, {"chunks": []}, llm))
    assert llm.calls == 2                 # đã retry đúng 1 lần
    assert len(turn.arguments) == 2       # cuối cùng đạt >= 2
    assert turn.parsed_ok is True
