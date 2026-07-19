"""
Test Supervisor judge (agents/supervisor_judge.py) với Gemini GIẢ -> không cần
GEMINI_API_KEY.

Kiểm: judge_node dùng lại được cho cả 3 stage (khác criteria/nội dung theo
state["current_stage"]), verdict pass/retry ráp đúng JudgeVerdict, retry_counts/
last_feedback cập nhật khi retry, tự chuyển RETRY -> REJECT khi hết lượt retry,
và lỗi rõ ràng nếu thiếu current_stage.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.judge_schemas import CriterionScore, JudgeOut
from agents.supervisor_judge import judge_node
from state.state_schema import (
    CriticRole,
    CriticTurn,
    DebateState,
    JudgeVerdict,
    PreparedContext,
    Stage,
    Verdict,
)


class _FakeResp:
    def __init__(self, parsed: JudgeOut):
        self.parsed = parsed
        self.text = None


class _FakeModels:
    def __init__(self, decision: JudgeOut):
        self.decision = decision

    def generate_content(self, model, contents, config):
        return _FakeResp(self.decision)


class _FakeClient:
    def __init__(self, decision: JudgeOut):
        self.models = _FakeModels(decision)


def _base_state(**overrides) -> dict:
    state = {
        "current_stage": Stage.PREPARE_CONTEXT,
        "context": PreparedContext(summary="Tóm tắt ngắn", chunks=[]),
        "retry_counts": {},
        "retry_limits": {"prepare_context": 2},
    }
    state.update(overrides)
    return state


def test_pass_verdict_no_retry_bookkeeping():
    client = _FakeClient(JudgeOut(
        verdict="pass", scores=[CriterionScore(criterion="coverage", score=0.9)],
        reasoning="Đủ tốt",
    ))
    delta = judge_node(_base_state(), client=client)

    verdict: JudgeVerdict = delta["judges"]["prepare_context"]
    assert verdict.verdict == Verdict.PASS
    assert verdict.stage == Stage.PREPARE_CONTEXT
    assert verdict.scores == {"coverage": 0.9}    # list[CriterionScore] -> dict ráp lại
    assert "retry_counts" not in delta       # pass -> không đụng retry bookkeeping
    assert "last_feedback" not in delta
    assert delta["current_node"] == "judge:prepare_context"


def test_retry_verdict_increments_count_and_sets_feedback():
    client = _FakeClient(JudgeOut(verdict="retry", reasoning="Thiếu", feedback="Thêm entities"))
    delta = judge_node(_base_state(retry_counts={"prepare_context": 0}), client=client)

    assert delta["judges"]["prepare_context"].verdict == Verdict.RETRY
    assert delta["retry_counts"] == {"prepare_context": 1}
    assert delta["last_feedback"] == {"prepare_context": "Thêm entities"}


def test_retry_downgrades_to_reject_when_limit_exhausted():
    client = _FakeClient(JudgeOut(verdict="retry", feedback="Vẫn thiếu"))
    state = _base_state(retry_counts={"prepare_context": 2}, retry_limits={"prepare_context": 2})
    delta = judge_node(state, client=client)

    assert delta["judges"]["prepare_context"].verdict == Verdict.REJECT


def test_missing_client_falls_back_to_pass(monkeypatch):
    # Giả lập thiếu OPENROUTER_API_KEY (không phụ thuộc .env thật của máy chạy test).
    import providers.openrouter_provider as openrouter_provider_module
    monkeypatch.setattr(openrouter_provider_module.openrouter_provider, "get_client", lambda: None)

    delta = judge_node(_base_state(), client=None)
    assert delta["judges"]["prepare_context"].verdict == Verdict.PASS
    assert "[fallback]" in delta["judges"]["prepare_context"].reasoning


def test_judges_debate_stage_with_different_content_and_criteria():
    debate = DebateState(
        round1={CriticRole.TAM_LY: CriticTurn(critic=CriticRole.TAM_LY, round=1, thesis="th")},
        round2={},
    )
    client = _FakeClient(JudgeOut(verdict="pass", reasoning="Substantive"))
    delta = judge_node(
        {"current_stage": Stage.CRITICS_DEBATE, "debate": debate,
         "retry_counts": {}, "retry_limits": {}},
        client=client,
    )

    assert delta["judges"]["critics_debate"].verdict == Verdict.PASS
    assert delta["current_node"] == "judge:critics_debate"


def test_missing_current_stage_raises_clear_error():
    with pytest.raises(ValueError, match="current_stage"):
        judge_node({"retry_counts": {}, "retry_limits": {}})


def test_same_function_reusable_for_all_three_stages(monkeypatch):
    # judge_node là 1 hàm 1-tham-số, đăng ký thẳng được dưới nhiều tên node LangGraph
    # (vd g.add_node("judge_essay", judge_node)) mà không cần factory/closure nào.
    import providers.openrouter_provider as openrouter_provider_module
    client = _FakeClient(JudgeOut(verdict="pass", reasoning="ok"))
    monkeypatch.setattr(openrouter_provider_module.openrouter_provider, "get_client", lambda: client)

    delta = judge_node({"current_stage": Stage.WRITE_ESSAY, "retry_counts": {}, "retry_limits": {}})
    assert delta["judges"]["write_essay"].verdict == Verdict.PASS
    assert delta["current_node"] == "judge:write_essay"


def test_first_attempt_recorded_as_best_without_touching_real_field():
    client = _FakeClient(JudgeOut(
        verdict="retry", scores=[CriterionScore(criterion="coverage", score=0.8)], feedback="Thiếu chút",
    ))
    ctx = PreparedContext(summary="Attempt 1", chunks=[])
    delta = judge_node(
        _base_state(context=ctx, retry_counts={"prepare_context": 0}, retry_limits={"prepare_context": 1}),
        client=client,
    )

    assert delta["best_attempts"]["prepare_context"] == {"output": ctx, "score": 0.8}
    assert "context" not in delta   # chưa hết lượt -> chưa cần phục hồi field thật


def test_retry_exhausted_restores_best_attempt_not_last_attempt():
    good_ctx = PreparedContext(summary="Attempt 1 - tốt", chunks=[])
    bad_ctx = PreparedContext(summary="Attempt 2 - tệ hơn", chunks=[])

    # Lần 1: RETRY, điểm 0.8 -> trở thành best (chưa có best nào trước đó).
    client_1 = _FakeClient(JudgeOut(
        verdict="retry", scores=[CriterionScore(criterion="coverage", score=0.8)], feedback="Thiếu chút",
    ))
    delta_1 = judge_node(
        _base_state(context=good_ctx, retry_counts={"prepare_context": 0}, retry_limits={"prepare_context": 1}),
        client=client_1,
    )
    assert delta_1["best_attempts"]["prepare_context"]["score"] == 0.8

    # Lần 2 (đã hết lượt: retry_counts=1 >= limit=1): điểm 0.3, TỆ HƠN lần 1.
    client_2 = _FakeClient(JudgeOut(
        verdict="retry", scores=[CriterionScore(criterion="coverage", score=0.3)], feedback="Vẫn thiếu",
    ))
    state_2 = _base_state(
        context=bad_ctx,                                    # field thật ĐÃ bị tool ghi đè bằng bản tệ hơn
        retry_counts={"prepare_context": 1},
        retry_limits={"prepare_context": 1},
        best_attempts=delta_1["best_attempts"],              # mô phỏng LangGraph merge delta_1 vào state
    )
    delta_2 = judge_node(state_2, client=client_2)

    assert delta_2["judges"]["prepare_context"].verdict == Verdict.REJECT
    assert delta_2["context"] is good_ctx        # phục hồi bản TỐT NHẤT (lần 1), không phải bản vừa chạy (lần 2)
    assert "best_attempts" not in delta_2        # bản lần 2 tệ hơn -> không ghi đè best


def test_retry_exhausted_keeps_current_when_it_is_the_best():
    ctx_1 = PreparedContext(summary="Attempt 1 - tệ", chunks=[])
    ctx_2 = PreparedContext(summary="Attempt 2 - tốt hơn", chunks=[])

    client_1 = _FakeClient(JudgeOut(
        verdict="retry", scores=[CriterionScore(criterion="coverage", score=0.3)], feedback="Kem",
    ))
    delta_1 = judge_node(
        _base_state(context=ctx_1, retry_counts={"prepare_context": 0}, retry_limits={"prepare_context": 1}),
        client=client_1,
    )

    client_2 = _FakeClient(JudgeOut(
        verdict="retry", scores=[CriterionScore(criterion="coverage", score=0.9)], feedback="Van chua du",
    ))
    state_2 = _base_state(
        context=ctx_2,
        retry_counts={"prepare_context": 1},
        retry_limits={"prepare_context": 1},
        best_attempts=delta_1["best_attempts"],
    )
    delta_2 = judge_node(state_2, client=client_2)

    assert delta_2["judges"]["prepare_context"].verdict == Verdict.REJECT
    assert delta_2["context"] is ctx_2                              # bản vừa chạy đã là bản tốt nhất
    assert delta_2["best_attempts"]["prepare_context"]["score"] == 0.9
