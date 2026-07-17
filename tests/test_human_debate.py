"""
Test "tranh luận cùng người học" — LLM GIẢ + queue thật, KHÔNG cần Ollama/Redis.

Cách test điểm PAUSE: chạy subgraph trong 1 task, dùng 1 task khác đóng vai FE/BE gọi
debate_session.submit() — đúng y hình dạng thật (2 coroutine, 2 request khác nhau).

Bao các ca:
  - opt-in TẮT  -> 2 node human là no-op, luồng cũ nguyên vẹn (4 critic, bulletin 4).
  - opt-in BẬT  -> người học thành thành viên thứ 5: R1 ra Argument 'human-aN' (bị 4 critic
                   bắt bẻ được), R2 ra Rebuttal.
  - BỎ QUA vòng 1 nhưng VẪN vào được vòng 2 (yêu cầu rõ của người dùng).
  - hết giờ im lặng -> hội đồng tự đi tiếp, không kẹt.
  - validate ở biên: id bịa / stance lạ -> 400; gửi khi không có phiên -> 409.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from langchain_core.messages import AIMessage

from agents.critics_debate import (
    CRITIC_ORDER,
    _turn_content,
    build_debate_subgraph,
    build_r2_prompt,
    critics_debate,
)
from agents.debate_schemas import CriticR1Out, CriticR2Out, _ArgIn, _RebuttalIn
from exceptions import DebateInvalidReply, DebateNotWaiting
from services.agent_service import debate_session
from services.agent_service.debate_session import HumanReply
from state.state_schema import (
    Argument,
    BulletinEntry,
    CriticRole,
    CriticTurn,
    PreparedContext,
    Rebuttal,
    SourceChunk,
)

THREAD = "t-human-1"


class _FakeStructured:
    def __init__(self, schema):
        self.schema = schema

    async def ainvoke(self, messages):
        if self.schema is CriticR1Out:
            return CriticR1Out(
                thesis="Luận đề mock",
                arguments=[_ArgIn(point="Điểm 1", support="Dựa c1"),
                           _ArgIn(point="Điểm 2", support="Dựa c2")],
            )
        if self.schema is CriticR2Out:
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


def _sub_input(human: bool):
    return {
        "question": "Phân tích bi kịch của nhân vật.",
        "work_title": "Tác phẩm X",
        "author": "Tác giả Y",
        "context_summary": "Tóm tắt ngắn.",
        "chunks": [SourceChunk(chunk_id="c1", text="Đoạn 1."),
                   SourceChunk(chunk_id="c2", text="Đoạn 2.")],
        "thread_id": THREAD,
        "human_in_debate": human,
        "round1": {}, "round2": {}, "bulletin": [],
        "consensus_points": [], "contested_points": [],
    }


def _state_for_node():
    """State cho node public critics_debate(). thread_id rỗng -> không có người học."""
    return {
        "thread_id": "",
        "human_message": "Phân tích bi kịch của nhân vật.",
        "intent": None,
        "context": PreparedContext(summary="Tóm tắt", chunks=[
            SourceChunk(chunk_id="c1", text="Đoạn 1."),
            SourceChunk(chunk_id="c2", text="Đoạn 2."),
        ]),
    }


async def _wait_round(n: int, timeout=5.0):
    """Chờ node mở phiên CHO ĐÚNG VÒNG n. Không sleep cứng (node còn phải chạy 4 critic).

    Phải bám theo VÒNG chứ không phải "có phiên hay không": ngay sau khi gửi tín hiệu kết
    thúc vòng 1, phiên vòng 1 vẫn còn mở thêm một nhịp -> chờ kiểu is_waiting() sẽ trả về
    ngay lập tức và tin của vòng 2 rơi vào cái queue vừa bị bỏ, mất hút.
    """
    async with asyncio.timeout(timeout):
        while debate_session.waiting_round(THREAD) != n:
            await asyncio.sleep(0.005)


@pytest.fixture(autouse=True)
def _clean():
    debate_session.close_session(THREAD)
    debate_session.clear_optin(THREAD)
    yield
    debate_session.close_session(THREAD)
    debate_session.clear_optin(THREAD)


# =============================================================================
# Opt-in TẮT -> node human phải trong suốt
# =============================================================================

def test_optin_off_human_nodes_are_noop():
    app = build_debate_subgraph(FakeLLM())
    out = asyncio.run(app.ainvoke(_sub_input(human=False)))

    assert set(out["round1"].keys()) == set(CRITIC_ORDER)   # không có HUMAN
    assert set(out["round2"].keys()) == set(CRITIC_ORDER)
    assert len(out["bulletin"]) == 4
    assert not debate_session.is_waiting(THREAD)            # không mở phiên chờ nào


# =============================================================================
# Opt-in BẬT — người học nói ở CẢ 2 vòng
# =============================================================================

def test_human_joins_both_rounds():
    async def scenario():
        app = build_debate_subgraph(FakeLLM())
        run = asyncio.create_task(app.ainvoke(_sub_input(human=True)))

        # --- Vòng 1: gõ 2 luận điểm rồi bấm "Kết thúc" ---
        await _wait_round(1)
        debate_session.submit(THREAD, HumanReply(message="Chí Phèo là bi kịch tha hoá."))
        debate_session.submit(THREAD, HumanReply(message="Thị Nở là lối về làm người."))
        debate_session.submit(THREAD, None)                 # Kết thúc phản biện

        # --- Vòng 2: reply 1 luận điểm của critic Tâm lý ---
        await _wait_round(2)
        debate_session.submit(THREAD, HumanReply(
            message="Văn bản cho thấy điều ngược lại.",
            target_arg_id="tam_ly-a1", stance="disagree"))
        debate_session.submit(THREAD, None)
        return await run

    out = asyncio.run(scenario())

    # Vòng 1: mỗi tin -> 1 Argument có arg_id 'human-aN' (thứ khiến critic bắt bẻ lại được)
    h1 = out["round1"][CriticRole.HUMAN]
    assert [a.arg_id for a in h1.arguments] == ["human-a1", "human-a2"]
    assert h1.arguments[0].point == "Chí Phèo là bi kịch tha hoá."
    assert h1.thesis == ""            # người học gõ thẳng luận điểm, không có luận đề
    assert h1.parsed_ok is True

    # Người học vào bảng tin -> 5 entry, và 4 critic ĐỌC được id human-*
    assert len(out["bulletin"]) == 5
    human_entry = next(e for e in out["bulletin"] if e.critic == CriticRole.HUMAN)
    assert human_entry.arg_ids == ["human-a1", "human-a2"]

    # Vòng 2: tin -> Rebuttal, target suy TỪ arg_id
    h2 = out["round2"][CriticRole.HUMAN]
    assert len(h2.rebuttals) == 1
    reb = h2.rebuttals[0]
    assert reb.target_critic == CriticRole.TAM_LY
    assert reb.target_arg_id == "tam_ly-a1"
    assert reb.stance == "disagree"
    assert reb.reason == "Văn bản cho thấy điều ngược lại."

    # stance của người học chảy vào contested (collect_node có tính lượt HUMAN)
    assert any("Người học" in line for line in out["contested_points"])


def test_each_human_message_is_emitted_immediately():
    """Mỗi tin phải ra 1 event NGAY lúc nhận, không gom cả lượt rồi bắn một cục ở cuối.

    Trước: người học Enter xong ngồi nhìn màn hình im tới lúc bấm Kết thúc, rồi mọi tin
    mới phọt ra một lượt. Bắt luôn lỗi arg_id: quên `start=` thì mọi tin live đều 'human-a1'.
    """
    seen: list[dict] = []

    async def scenario():
        app = build_debate_subgraph(FakeLLM())
        run = asyncio.create_task(app.ainvoke(_sub_input(human=True),
                                              config={"callbacks": []}))
        await _wait_round(1)
        debate_session.submit(THREAD, HumanReply(message="Tin một"))
        # Chờ event của tin 1 xuất hiện TRƯỚC khi gửi tin 2 -> chứng minh nó không đợi
        # tới cuối lượt.
        async with asyncio.timeout(3):
            while not [e for e in seen if e.get("round") == 1]:
                await asyncio.sleep(0.005)
        debate_session.submit(THREAD, HumanReply(message="Tin hai"))
        async with asyncio.timeout(3):
            while len([e for e in seen if e.get("round") == 1]) < 2:
                await asyncio.sleep(0.005)
        debate_session.submit(THREAD, None)
        await _wait_round(2)
        debate_session.submit(THREAD, None)
        return await run

    # Bắt writer của LangGraph: mỗi _emit_live_turn của người học đi qua đây.
    import agents.critics_debate as cd

    real = cd._emit_live_turn

    def spy(writer, role, round_no, turn):
        if role == CriticRole.HUMAN:
            seen.append({"round": round_no,
                         "arg_ids": [a.arg_id for a in turn.arguments]})
        return real(writer, role, round_no, turn)

    cd._emit_live_turn = spy
    try:
        out = asyncio.run(scenario())
    finally:
        cd._emit_live_turn = real

    r1 = [e for e in seen if e["round"] == 1]
    assert len(r1) == 2                       # 2 tin -> 2 event, KHÔNG phải 1 event gộp
    assert r1[0]["arg_ids"] == ["human-a1"]   # từng tin mang đúng id của mình
    assert r1[1]["arg_ids"] == ["human-a2"]
    # state vẫn giữ bản GỘP đầy đủ (bulletin + bài luận đọc từ đây)
    assert [a.arg_id for a in out["round1"][CriticRole.HUMAN].arguments] == \
        ["human-a1", "human-a2"]


def test_human_skips_round1_but_joins_round2():
    """Vắng vòng 1 KHÔNG được tước quyền vào vòng 2 (yêu cầu rõ của người dùng).

    Bẫy: nếu ai đó implement "bỏ qua" bằng cách xoá cờ opt-in thì pause vòng 2 biến mất.
    """
    async def scenario():
        app = build_debate_subgraph(FakeLLM())
        run = asyncio.create_task(app.ainvoke(_sub_input(human=True)))

        await _wait_round(1)
        debate_session.submit(THREAD, None)                 # Bỏ qua vòng 1

        await _wait_round(2)                                # vòng 2 VẪN phải mời
        debate_session.submit(THREAD, HumanReply(
            message="Tôi nghĩ khác.", target_arg_id="lich_su-a2", stance="qualify"))
        debate_session.submit(THREAD, None)
        return await run

    out = asyncio.run(scenario())

    assert CriticRole.HUMAN not in out["round1"]            # vắng vòng 1
    assert len(out["bulletin"]) == 4                        # -> không có entry human
    assert CriticRole.HUMAN in out["round2"]                # nhưng vẫn nói được vòng 2
    assert out["round2"][CriticRole.HUMAN].rebuttals[0].target_critic == CriticRole.LICH_SU


def test_idle_timeout_lets_council_continue(monkeypatch):
    """Người học không gõ gì -> hết giờ im lặng, hội đồng đi tiếp chứ không kẹt."""
    monkeypatch.setattr(debate_session, "IDLE_TIMEOUT_S", 0.05)

    app = build_debate_subgraph(FakeLLM())
    out = asyncio.run(app.ainvoke(_sub_input(human=True)))

    assert CriticRole.HUMAN not in out["round1"]
    assert CriticRole.HUMAN not in out["round2"]
    assert set(out["round1"].keys()) == set(CRITIC_ORDER)   # luồng vẫn chạy trọn
    assert not debate_session.is_waiting(THREAD)            # phiên được dọn sạch


def test_max_turns_closes_session(monkeypatch):
    """Đủ trần số tin -> node tự đóng phiên, không chờ thêm."""
    monkeypatch.setattr(debate_session, "MAX_HUMAN_TURNS", 2)

    async def scenario():
        app = build_debate_subgraph(FakeLLM())
        run = asyncio.create_task(app.ainvoke(_sub_input(human=True)))
        await _wait_round(1)
        debate_session.submit(THREAD, HumanReply(message="Tin 1"))
        debate_session.submit(THREAD, HumanReply(message="Tin 2"))
        # KHÔNG gửi None: node phải tự thoát vì đã đủ trần.
        await _wait_round(2)                 # sang thẳng vòng 2
        debate_session.submit(THREAD, None)
        return await run

    out = asyncio.run(scenario())
    assert len(out["round1"][CriticRole.HUMAN].arguments) == 2


# =============================================================================
# Validate ở biên (endpoint) — id do FE gửi sai thì báo lỗi THẲNG, khác đường LLM
# (id model bịa thì bị bỏ im lặng).
# =============================================================================

def test_submit_without_session_raises_409():
    with pytest.raises(DebateNotWaiting):
        debate_session.submit(THREAD, HumanReply(message="Không ai chờ tôi"))


def test_round2_rejects_fake_arg_id():
    debate_session.open_session(THREAD, 2, {"tam_ly-a1"})
    with pytest.raises(DebateInvalidReply):
        debate_session.submit(THREAD, HumanReply(
            message="x", target_arg_id="khong_co-a9", stance="agree"))


def test_round2_rejects_bad_stance():
    debate_session.open_session(THREAD, 2, {"tam_ly-a1"})
    with pytest.raises(DebateInvalidReply):
        debate_session.submit(THREAD, HumanReply(
            message="x", target_arg_id="tam_ly-a1", stance="ghet"))


def test_round2_accepts_bracketed_arg_id():
    """Bảng tin in id trong ngoặc vuông -> nhận cả '[tam_ly-a1]' lẫn 'tam_ly-a1'."""
    debate_session.open_session(THREAD, 2, {"tam_ly-a1"})
    debate_session.submit(THREAD, HumanReply(
        message="x", target_arg_id="[tam_ly-a1]", stance="agree"))


def test_round2_rejects_self_target():
    """Người học phản biện chính luận điểm mình vừa nêu = vô nghĩa (4 critic bị chặn ở
    _speak_r2 qua `target == role`; người học đi đường khác nên chặn ở debate_session)."""
    debate_session.open_session(THREAD, 2, {"tam_ly-a1", "human-a1"})
    with pytest.raises(DebateInvalidReply):
        debate_session.submit(THREAD, HumanReply(
            message="x", target_arg_id="human-a1", stance="agree"))


def test_round1_rejects_empty_message():
    debate_session.open_session(THREAD, 1, set())
    with pytest.raises(DebateInvalidReply):
        debate_session.submit(THREAD, HumanReply(message="   "))


def test_optin_is_one_shot():
    """take_optin() phải XOÁ cờ -> không dính sang lượt chat sau của cùng thread."""
    debate_session.mark_optin(THREAD)
    assert debate_session.take_optin(THREAD) is True
    assert debate_session.take_optin(THREAD) is False


# =============================================================================
# Clause ép vòng 2 — điều kiện phải là "bảng tin CÓ human", không phải cờ opt-in
# =============================================================================

def _bulletin(with_human: bool):
    entries = [BulletinEntry(critic=CriticRole.TAM_LY, thesis="Luận đề", key_points=["p"],
                             supports=["s"], arg_ids=["tam_ly-a1"])]
    if with_human:
        entries.append(BulletinEntry(critic=CriticRole.HUMAN, thesis="", key_points=["ý tôi"],
                                     supports=[""], arg_ids=["human-a1"]))
    return entries


def test_r2_prompt_forces_targeting_human_when_present():
    """Clause phải có CẢ cận dưới lẫn cận trên.

    Đo thật với qwen2.5:3b: chỉ ra lệnh "ít nhất 1" thì model dồn 3/3 phản biện vào người
    học và bỏ hẳn 5 luận điểm của critic khác -> tranh luận AI–AI biến mất.
    """
    p = build_r2_prompt(CriticRole.LICH_SU, "luận đề", _bulletin(True), [], "câu hỏi",
                        has_human=True)
    assert "human-" in p
    assert "ĐÚNG 1 phản biện (không nhiều hơn)" in p
    assert "còn lại PHẢI nhắm vào nhà phê bình khác" in p
    assert "NGƯỜI HỌC" in p


def test_r2_prompt_has_no_human_clause_when_absent():
    """Người học bỏ qua vòng 1 -> KHÔNG được ra lệnh nhắm vào id không tồn tại, nếu không
    model buộc phải bịa id -> _target_from_arg_id loại sạch -> vòng 2 trống trơn."""
    p = build_r2_prompt(CriticRole.LICH_SU, "luận đề", _bulletin(False), [], "câu hỏi",
                        has_human=False)
    assert "human-" not in p
    assert "NGƯỜI HỌC" not in p
    assert "các nhà phê bình KHÁC" in p


# =============================================================================
# Bong bóng chính của vòng 2 — KHÔNG được lặp lại luận đề vòng 1
# =============================================================================

def test_round2_content_does_not_repeat_round1_thesis():
    """R2 echo own_thesis -> FE vẽ lại y nguyên bong bóng vòng 1 lần hai (thiếu luận điểm
    con) -> trông như critic phát biểu lại thay vì đang bắt bẻ ai."""
    turn = CriticTurn(
        critic=CriticRole.TAM_LY, round=2, bulletin_seen=True,
        thesis="Luận đề vòng 1 của tôi",     # _speak_r2 vẫn gán own_thesis vào đây
        rebuttals=[Rebuttal(target_critic=CriticRole.LICH_SU, target_arg_id="lich_su-a1",
                            stance="disagree", reason="Văn bản nói ngược lại")],
    )
    assert _turn_content(2, CriticRole.TAM_LY, turn) == ""


def test_round2_without_rebuttals_says_so():
    """Parse lỗi / bị loại sạch id bịa -> nói thẳng, đừng lấy luận đề cũ lấp chỗ trống."""
    turn = CriticTurn(critic=CriticRole.TAM_LY, round=2, bulletin_seen=True,
                      thesis="Luận đề vòng 1 của tôi", rebuttals=[], parsed_ok=False)
    assert _turn_content(2, CriticRole.TAM_LY, turn) == "(không đưa ra phản biện nào)"


def test_round1_content_still_shows_thesis():
    """Vòng 1 KHÔNG đổi: luận đề vẫn là nội dung chính của bong bóng."""
    turn = CriticTurn(critic=CriticRole.TAM_LY, round=1, thesis="Luận đề của tôi",
                      arguments=[Argument(arg_id="tam_ly-a1", point="p", support="s")])
    assert _turn_content(1, CriticRole.TAM_LY, turn) == "Luận đề của tôi"


def test_emitted_r2_events_do_not_echo_r1_content():
    """Chốt ở tầng EVENT, không chỉ ở hàm dựng chuỗi.

    Đây mới là thứ FE nhận và Redis checkpoint: trước đây event r2 mang `content` GIỐNG HỆT
    TỪNG KÝ TỰ với event r1 của cùng critic -> FE vẽ lại bong bóng vòng 1 lần thứ hai.
    """
    app = build_debate_subgraph(FakeLLM())
    delta = asyncio.run(critics_debate(_state_for_node(), subgraph=app))

    evs = [e for e in delta["events"] if e.type.value == "critic_turn"]
    r1 = {e.node: e.content for e in evs if e.node.endswith(":r1")}
    r2 = {e.node: e.content for e in evs if e.node.endswith(":r2")}
    assert len(r1) == 4 and len(r2) == 4

    for role in CRITIC_ORDER:
        assert r1[f"critic:{role.value}:r1"]           # vòng 1 vẫn có luận đề
        assert r2[f"critic:{role.value}:r2"] == ""     # vòng 2 KHÔNG lặp lại nó
    # và không chuỗi r1 nào tái xuất ở r2
    assert not (set(r1.values()) & set(v for v in r2.values() if v))


def test_round1_human_gets_no_parse_error_label():
    """Người học không qua LLM -> thesis rỗng là ĐÚNG, đừng dán nhãn lỗi của đường AI."""
    turn = CriticTurn(critic=CriticRole.HUMAN, round=1, thesis="",
                      arguments=[Argument(arg_id="human-a1", point="ý tôi", support="")])
    assert _turn_content(1, CriticRole.HUMAN, turn) == ""
    assert _turn_content(1, CriticRole.TAM_LY, turn) == "(chưa parse được luận điểm)"


def test_bulletin_omits_empty_thesis_label():
    """Lượt người học không có luận đề -> đừng in nhãn 'Luận đề:' cụt đuôi vào prompt."""
    p = build_r2_prompt(CriticRole.LICH_SU, "x", _bulletin(True), [], "q", has_human=True)
    assert "Luận đề: \n" not in p
    assert "[human-a1] ý tôi" in p
