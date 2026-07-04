"""
Supervisor judge — chấm output của từng tool Qwen fine-tuned (Tool 1/2/3) và quyết
định PASS (đi tiếp) hay RETRY (chạy lại tool kèm feedback).

Dùng CHUNG 1 hàm `judge_node(state)` cho cả 3 cổng chấm trong diagram ("Context đủ
chi tiết?", "Debate có substantive?", "Logic + style + depth?"), khác nhau ở 2 chỗ
(đều là DỮ LIỆU, không phải hành vi):
  - tiêu chí chấm       -> config/judge_prompts.py, key theo Stage
  - nội dung cần chấm   -> _CONTENT_RENDERERS[stage], lấy từ state theo Stage

Tránh tách 3 class riêng cho 3 tool vì cả 3 lần chấm có CÙNG shape (system prompt
tiêu chí + human prompt nội dung -> structured output JudgeOut), chỉ khác dữ liệu.
Mirror đúng cách agents/critics_debate.py xử lý 4 critic khác persona: 1 hàm dùng
lại + dict tĩnh keyed theo enum, không phải class per-role.

KHÔNG cần factory/closure bind stage: mỗi tool (prepare_context, critics_debate,
write_essay) đã tự set `state["current_stage"]` trước khi judge chạy (xem
agents/critics_debate.py cuối hàm critics_debate()). judge_node() chỉ ĐỌC lại field
đó để biết đang chấm gì -> không cần ai truyền stage vào từ ngoài.

Nhờ vậy `judge_node` là hàm 1 tham số (`state`) đúng chữ ký node LangGraph, đăng ký
thẳng được, có thể dùng lại y nguyên cho cả 3 vị trí trong graph nếu muốn tên node
riêng để dễ trace/checkpoint:
    g.add_node("judge_context", judge_node)
    g.add_node("judge_debate", judge_node)
    g.add_node("judge_essay", judge_node)
(3 tên node khác nhau, CÙNG 1 hàm bên dưới — không phải 3 hàm khác nhau.)

Dùng Gemini (google-genai) structured output, mirror agents/supervisor.py. Thiếu
GEMINI_API_KEY hoặc lỗi gọi -> fallback verdict=PASS để graph vẫn chạy tiếp (an toàn
hơn RETRY, vì lỗi gọi model không nên khiến pipeline lặp vô hạn).

Hết lượt retry -> KHÔNG hard-fail: mỗi lần chấm, so điểm (trung bình `scores`) với
bản tốt nhất đã thấy trước đó (`state["best_attempts"][stage]`), giữ lại bản điểm
cao hơn. Khi hết lượt mà vẫn RETRY, phục hồi bản TỐT NHẤT (có thể không phải bản vừa
chạy) vào field thật (context/debate/essay) rồi coi verdict là REJECT chỉ để LOG nội
bộ — graph vẫn đi tiếp bình thường (không có lượt retry nào bị "vứt bỏ trắng tay").
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from dotenv import load_dotenv

from agents.judge_schemas import JudgeOut
from config.judge_prompts import get_judge_criteria
from providers.gemini_provider import gemini_provider
from state.agent_state import AgentState
from state.state_schema import (
    DebateState,
    EssayDraft,
    EventEmitter,
    JudgeVerdict,
    PreparedContext,
    Stage,
    Verdict,
    safe_stream_writer,
)

load_dotenv()

logger = logging.getLogger("rag-service.graph.supervisor_judge")

JUDGE_MODEL = os.getenv("GEMINI_SUPERVISOR_MODEL", "gemini-2.5-flash")

DEFAULT_RETRY_LIMIT = 1  # dùng khi state chưa set retry_limits cho stage này


# =============================================================================
# Ráp nội dung cần chấm từ state, theo từng Stage (dữ liệu khác nhau, không phải
# hành vi khác nhau) — giống _render_chunks/_render_bulletin ở critics_debate.py.
# =============================================================================

def _render_context(state: AgentState) -> str:
    ctx: Optional[PreparedContext] = state.get("context")
    if ctx is None:
        return "(chưa có context — Tool 1 chưa chạy)"
    entities = ", ".join(e.name for e in ctx.entities) or "(không có)"
    themes = ", ".join(ctx.themes) or "(không có)"
    return (
        f"Tóm tắt:\n{ctx.summary or '(trống)'}\n\n"
        f"Số đoạn trích (chunks): {len(ctx.chunks)}\n"
        f"Entities: {entities}\n"
        f"Themes: {themes}"
    )


def _render_debate(state: AgentState) -> str:
    debate: Optional[DebateState] = state.get("debate")
    if debate is None:
        return "(chưa có debate — Tool 2 chưa chạy)"
    lines = []
    for role, turn in debate.round1.items():
        lines.append(
            f"[R1] {role.value}: {turn.thesis} "
            f"({len(turn.arguments)} luận điểm, parsed_ok={turn.parsed_ok})"
        )
    for role, turn in debate.round2.items():
        lines.append(
            f"[R2] {role.value}: {len(turn.rebuttals)} phản biện, parsed_ok={turn.parsed_ok}"
        )
    lines.append(
        f"Đồng thuận: {len(debate.consensus_points)} điểm; "
        f"Tranh cãi: {len(debate.contested_points)} điểm"
    )
    return "\n".join(lines)


def _render_essay(state: AgentState) -> str:
    essay: Optional[EssayDraft] = state.get("essay")
    if essay is None:
        return "(chưa có essay — Tool 3 chưa chạy)"
    check = essay.citation_check
    check_line = (
        f"Citation check: {check.verified}/{check.total} verified, passed={check.passed}"
        if check else "Citation check: (chưa chạy)"
    )
    return (
        f"Tiêu đề: {essay.title}\n"
        f"Số đoạn (sections): {len(essay.sections)}, word_count={essay.word_count}\n"
        f"{check_line}\n\n"
        f"Toàn văn:\n{essay.full_text}"
    )


_CONTENT_RENDERERS: dict[Stage, Callable[[AgentState], str]] = {  # trong này chứa các hàm render nội dung cần chấm theo stage
    Stage.PREPARE_CONTEXT: _render_context,
    Stage.CRITICS_DEBATE: _render_debate,
    Stage.WRITE_ESSAY: _render_essay,
}
# Callable[[AgentState], str]  khai báo giống Function<> trong java -> tham số AgentState cần để các hàm render chạy
# Callable[[Kiểu_Tham_Số_1, Kiểu_Tham_Số_2, ...], Kiểu_Trả_Về]

# Field thật trong state chứa output của từng stage -> dùng để đọc bản hiện tại và
# để phục hồi bản tốt nhất khi hết lượt retry (xem judge_node).
_OUTPUT_FIELD: dict[Stage, str] = {
    Stage.PREPARE_CONTEXT: "context",
    Stage.CRITICS_DEBATE: "debate",
    Stage.WRITE_ESSAY: "essay",
}


def _aggregate_score(verdict: Verdict, scores: dict[str, float]) -> float:
    """Điểm tổng hợp để so 'bản nào tốt hơn' giữa các lần retry.

    Ưu tiên trung bìn h các tiêu chí Gemini chấm; nếu Gemini không cho điểm (rỗng,
    hoặc rơi vào nhánh fallback), dùng verdict làm điểm thay thế (PASS > RETRY).
    """
    if scores:
        return sum(scores.values()) / len(scores)
    return 1.0 if verdict == Verdict.PASS else 0.0

# =============================================================================
# Gọi LLM
# =============================================================================

def _judge(stage: Stage, content: str, *, client=None) -> JudgeOut:
    """Gọi Gemini chấm `content` theo tiêu chí của `stage`. Mọi sự cố -> fallback pass."""
    client = client or gemini_provider.get_client()
    if client is None:
        logger.warning("Thiếu GEMINI_API_KEY -> fallback verdict=pass.")
        return JudgeOut(verdict="pass", reasoning="[fallback] chưa cấu hình GEMINI_API_KEY")
    try:
        from google.genai import types
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=content,
            config=types.GenerateContentConfig(
                system_instruction=get_judge_criteria(stage), # lấy prompt tiêu chí chấm
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=JudgeOut,
            ),
        )
        decision = resp.parsed
        if isinstance(decision, JudgeOut):
            return decision
        if resp.text:
            return JudgeOut.model_validate_json(resp.text)
        raise ValueError("Gemini trả về rỗng")
    except Exception as e:
        logger.warning("Judge %s gọi Gemini lỗi (%s) -> fallback verdict=pass.", stage.value, e)
        return JudgeOut(verdict="pass", reasoning=f"[fallback] lỗi gọi Gemini: {e}")


# =============================================================================
# Node public (1 hàm dùng lại cho cả 3 cổng chấm)
# =============================================================================

def judge_node(state: AgentState, *, client=None) -> dict:
    """Chấm output của tool vừa chạy, quyết định PASS/RETRY.

    Đọc `state["current_stage"]` để biết đang chấm tool nào — field này do chính
    tool đó set ngay trước khi judge chạy (vd critics_debate() set
    current_stage=Stage.CRITICS_DEBATE), nên không cần nhận stage từ ngoài.

    Hết lượt retry: KHÔNG hard-fail. Phục hồi bản TỐT NHẤT đã thấy (so bằng điểm
    trung bình `scores`) vào field thật rồi cho graph đi tiếp — verdict lúc đó là
    REJECT chỉ mang ý nghĩa "chưa từng pass sạch", không chặn pipeline.

    client: inject để test (fake Gemini client); None -> dùng gemini_provider thật.
    """
    # Đặt tên label_stage (không phải `stage` trơn) để không lẫn với `state` khi
    # đọc lướt — 2 biến đứng sát nhau suốt hàm này.
    label_stage = state.get("current_stage")
    if label_stage is None:
        raise ValueError(
            "judge_node cần state['current_stage'] đã được tool trước đó set "
            "(vd critics_debate() set Stage.CRITICS_DEBATE)."
        )

    emitter = EventEmitter(state, writer=safe_stream_writer())
    render = _CONTENT_RENDERERS[label_stage]
    out = _judge(label_stage, render(state), client=client)

    retry_counts = state.get("retry_counts") or {}
    retry_limits = state.get("retry_limits") or {}
    count = retry_counts.get(label_stage.value, 0)
    limit = retry_limits.get(label_stage.value, DEFAULT_RETRY_LIMIT)

    verdict = Verdict(out.verdict)
    scores = {s.criterion: s.score for s in out.scores}
    score = _aggregate_score(verdict, scores)

    # So bản hiện tại với bản tốt nhất đã lưu -> giữ bản điểm cao hơn.
    field_name = _OUTPUT_FIELD[label_stage]
    current_output = state.get(field_name)
    best = (state.get("best_attempts") or {}).get(label_stage.value)

    delta: dict = {
        "current_node": f"judge:{label_stage.value}",
    }
    if best is None or score >= best["score"]:
        best = {"output": current_output, "score": score}
        delta["best_attempts"] = {label_stage.value: best}

    if verdict == Verdict.RETRY and count >= limit:
        logger.info(
            "Judge %s: hết lượt retry (%d/%d) -> dùng bản tốt nhất (score=%.2f), không hard-fail",
            label_stage.value, count, limit, best["score"],
        )
        verdict = Verdict.REJECT           # chỉ để log/observability
        delta[field_name] = best["output"]  # field thật = bản TỐT NHẤT, không phải bản vừa chạy

    judge_verdict = JudgeVerdict(
        stage=label_stage,
        verdict=verdict,
        scores=scores,
        reasoning=out.reasoning,
        feedback=out.feedback,
        judged_at=datetime.now(timezone.utc),
    )
    logger.info("Judge %s verdict=%s score=%.2f", label_stage.value, verdict.value, score)

    delta["judges"] = {label_stage.value: judge_verdict}
    node_name = f"judge:{label_stage.value}"
    emitter.judge(node_name, judge_verdict)
    if verdict == Verdict.RETRY:
        delta["retry_counts"] = {label_stage.value: count + 1}
        delta["last_feedback"] = {label_stage.value: out.feedback}
        emitter.retry(
            node_name,
            out.feedback or "Chưa đạt yêu cầu, chạy lại.",
            payload={"stage": label_stage.value, "attempt": count + 1, "limit": limit},
        )
    delta["events"] = emitter.milestones
    delta["event_seq"] = emitter.seq
    return delta
