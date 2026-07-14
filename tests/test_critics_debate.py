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
    _speak_r1,
    _target_from_arg_id,
    build_debate_subgraph,
    build_r1_prompt,
    build_r2_prompt,
    critics_debate,
)
from agents.debate_schemas import CriticR1Out, CriticR2Out, _ArgIn, _RebuttalIn
from state.state_schema import (
    BulletinEntry,
    CriticRole,
    DebateState,
    PreparedContext,
    SourceChunk,
)


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
            # arg_id "{target}-a1" luôn có thật (mỗi critic R1 nêu 2 luận điểm).
            return CriticR2Out(rebuttals=[
                _RebuttalIn(target_arg_id=f"{r.value}-a1", stance="disagree",
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
            # target_critic PHẢI suy ra từ chính id -> không thể lệch nhau
            assert reb.target_arg_id == f"{reb.target_critic.value}-a1"


def test_target_from_arg_id_chong_bia():
    valid = {"tam_ly-a1", "tam_ly-a2", "lich_su-a1"}
    assert _target_from_arg_id("tam_ly-a2", valid) == CriticRole.TAM_LY  # hợp lệ
    assert _target_from_arg_id("tam_ly-a3", valid) is None               # ngoài range
    assert _target_from_arg_id("lich_su-a2", valid) is None              # critic có, luận điểm không
    assert _target_from_arg_id("", valid) is None                        # model không chọn
    assert _target_from_arg_id("critic_la-a1", valid) is None            # role lạ


def test_target_from_arg_id_chap_nhan_ngoac_vuong():
    """Qwen chép id KÈM ngoặc vuông từ bảng tin ('[lich_su-a1]') -> phải khớp, không bị loại.

    Ca hỏng thật: bảng tin in '[lich_su-a1]', prompt bảo "chép nguyên văn id trong ngoặc
    vuông", Qwen trả '[lich_su-a1]' -> so với arg_ids (không ngoặc) trượt hết -> cả 4 critic
    mất sạch rebuttal -> vòng 2 rỗng, FE trông y như vòng 1.
    """
    valid = {"lich_su-a1", "hinh_thuc-a2"}
    assert _target_from_arg_id("[lich_su-a1]", valid) == CriticRole.LICH_SU
    assert _target_from_arg_id("  [hinh_thuc-a2] ", valid) == CriticRole.HINH_THUC
    assert _target_from_arg_id("LICH_SU-A1", valid) == CriticRole.LICH_SU   # model viết hoa
    assert _target_from_arg_id("[tam_ly-a9]", valid) is None                # vẫn chặn id bịa


def test_rebuttal_khong_the_gan_sai_critic():
    """Hồi quy: reason bắt bẻ luận điểm của Lịch sử thì KHÔNG thể bị gắn nhãn Tiếp nhận.

    Ca hỏng cũ: model điền tách rời (target_critic=tiep_nhan, target_point=2) trong khi
    reason đang nói về luận điểm của Lịch sử -> "tiep_nhan-a2" vẫn tồn tại nên lọt validate,
    FE in "Trả lời Tiếp nhận". Nay attribution suy TỪ id nên hai thứ không thể lệch.
    """
    valid = {"lich_su-a2", "tiep_nhan-a2"}
    assert _target_from_arg_id("lich_su-a2", valid) == CriticRole.LICH_SU


def test_r1_prompt_co_de_bai():
    """R1 phải thấy câu hỏi gốc, nếu không 4 critic phân tích chung chung, không bám đề."""
    p = build_r1_prompt(
        CriticRole.TAM_LY, "Tỏ lòng", "Phạm Ngũ Lão", "Tóm tắt", _chunks(),
        question="Phân tích hình ảnh người tráng sĩ trong Tỏ lòng",
    )
    assert "Phân tích hình ảnh người tráng sĩ trong Tỏ lòng" in p


def test_r2_prompt_co_de_bai_van_ban_goc_va_ly_le():
    """R2 phải có ĐỀ BÀI + VĂN BẢN GỐC + lý lẽ (support) của luận điểm bị nhắm tới.

    Thiếu văn bản gốc -> critic không đối chiếu được dẫn chứng; thiếu support -> luận điểm
    chỉ là khẳng định trần trụi, không có gì để bắt bẻ. Cả hai đều đẩy R2 về chỗ nhắc lại
    luận đề của chính mình thay vì phản biện.
    """
    bulletin = [
        BulletinEntry(
            critic=CriticRole.LICH_SU,
            thesis="Luận đề Lịch sử",
            key_points=["Tác phẩm phản ánh tinh thần tự cường"],
            supports=["Dựa vào hình ảnh ba vạn quân trong đoạn c1"],
            arg_ids=["lich_su-a1"],
        ),
    ]
    p = build_r2_prompt(
        CriticRole.TAM_LY, "Luận đề Tâm lý", bulletin, _chunks(),
        question="Phân tích hình ảnh người tráng sĩ trong Tỏ lòng",
    )
    assert "Phân tích hình ảnh người tráng sĩ trong Tỏ lòng" in p   # đề bài
    assert "Đoạn văn mẫu 1 của tác phẩm." in p                      # văn bản gốc
    assert "[lich_su-a1]" in p                                      # id để chép
    assert "Dựa vào hình ảnh ba vạn quân trong đoạn c1" in p        # lý lẽ để bắt bẻ


def test_r2_prompt_chiu_duoc_bulletin_cu_khong_co_supports():
    """Bulletin trong checkpoint CŨ chưa có field supports -> vẫn phải render đủ luận điểm."""
    bulletin = [
        BulletinEntry(critic=CriticRole.LICH_SU, thesis="Luận đề",
                      key_points=["Điểm A", "Điểm B"],
                      arg_ids=["lich_su-a1", "lich_su-a2"]),   # supports = [] (mặc định)
    ]
    p = build_r2_prompt(CriticRole.TAM_LY, "th", bulletin, _chunks())
    assert "[lich_su-a1] Điểm A" in p
    assert "[lich_su-a2] Điểm B" in p


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
