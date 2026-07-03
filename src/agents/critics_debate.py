"""
Tool 2 — critics_debate: 4 nhà phê bình × 2 vòng, phản biện song song trên CÙNG
một đoạn văn bản.

Đơn giản hóa: Tool 2 KHÔNG tự retrieve. Đoạn văn bản gốc (chunks) do Tool 1
(prepare_context) chuẩn bị và truyền vào qua state["context"]. Cả 4 persona đọc
CHUNG đoạn đó; khác biệt nằm ở góc nhìn (system prompt), không ở dữ liệu.
Lý do: Vector DB chỉ chứa văn bản gốc thô + metadata, không có nhãn phân tích nên
cho từng critic tự truy vấn theo "dimension" (vd 'tâm trạng') là vô nghĩa.

Fan-out được gói trong 1 SUBGRAPH (LangGraph), expose ra ngoài như MỘT node
`critics_debate(state)` -> graph chính (graph/workflow.py) KHÔNG phải sửa; khi
Tool 1 / Mode A xong chỉ cần trỏ 1 edge tới node này.

Luồng subgraph (khớp tool2_critics_debate_internals_detail.svg):
    START ─┬─> {4 critic}_r1 ─┐
           │  (đọc chunks)    ├─> bulletin ─┬─> {4 critic}_r2 ─┐
           └──────────────────┘  (barrier)  │  (đọc bulletin)  ├─> collect ─> END
                                            └───────────────────┘  (barrier)

Phân chia file của Tool 2:
- config/critic_prompts.py : persona (system prompt, chuỗi tĩnh).
- agents/debate_schemas.py : schema I/O của LLM (CriticR1Out / CriticR2Out).
- file này                 : ráp human prompt + DebateSubState + điều phối subgraph + gọi LLM.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.debate_schemas import CriticR1Out, CriticR2Out
from config.critic_prompts import CRITIC_PERSONAS
from state.state_schema import (
    CRITIC_DISPLAY,
    Argument,
    BulletinEntry,
    CriticRole,
    CriticTurn,
    DebateState,
    Rebuttal,
    SourceChunk,
    Stage,
    merge_dict,
)

logger = logging.getLogger("rag-service.agents.critics_debate")

CRITIC_ORDER: list[CriticRole] = [ # difine rõ ràng CRITIC_ORDER là 1 list CriticRole mở khóa khả năng truy xuất
    CriticRole.HINH_THUC,
    CriticRole.LICH_SU,
    CriticRole.TAM_LY,
    CriticRole.TIEP_NHAN,
]

MIN_R1_ARGS = 2   # mỗi critic R1 tối thiểu 2 luận điểm (để R2 có cái phản biện + target_arg_id map được)


# =============================================================================
# Sub-state của subgraph (giữ cạnh build_debate_subgraph; chỉ sống in-memory, không persist)
# =============================================================================

class DebateSubState(TypedDict, total=False):
    work_title: Optional[str]
    author: Optional[str]
    context_summary: str
    chunks: list[SourceChunk]        # đoạn văn bản CHUNG (từ Tool 1) cho cả 4 critic
    judge_feedback: str               # feedback của supervisor_judge lượt RETRY trước (rỗng nếu lần đầu)
    # 4 node ghi song song -> cần reducer merge_dict
    round1: Annotated[dict[CriticRole, CriticTurn], merge_dict]
    round2: Annotated[dict[CriticRole, CriticTurn], merge_dict]
    # các key dưới do 1 node ghi -> không cần reducer
    bulletin: list[BulletinEntry]
    consensus_points: list[str]
    contested_points: list[str]


# =============================================================================
# Helper
# =============================================================================

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_text(content: Any) -> str:
    """ChatOllama có thể trả str hoặc list block -> gộp về chuỗi (dùng cho fallback)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and "text" in b:
                parts.append(str(b["text"]))
        return "".join(parts).strip()
    return str(content).strip()


def _fallback_text(llm, msgs) -> str:
    try:
        return _extract_text(llm.invoke(msgs).content)
    except Exception as e:  # pragma: no cover - đường lỗi kép
        return f"[lỗi gọi model: {e}]"


def _resolve_arg_id(target: CriticRole, point: Optional[int],
                    valid_ids: set[str]) -> Optional[str]:
    """Ghép (critic, SỐ thứ tự luận điểm) -> arg_id; None nếu số không hợp lệ.

    Chống 3B bịa id: model chỉ CHỌN số, server dựng id theo quy tắc "{role}-a{i}"
    rồi kiểm tra id đó có thật trong bulletin không. Sai/thiếu -> None (không tạo id rác).
    """
    if not point or point < 1:
        return None
    candidate = f"{target.value}-a{point}"
    return candidate if candidate in valid_ids else None


# =============================================================================
# Ráp human prompt cho từng vòng (nội suy dữ liệu runtime vào chuỗi).
# System prompt (persona) nằm ở config/critic_prompts.py; đây là phần "nhiệm vụ +
# dữ liệu" nên là logic, không để trong config được.
# =============================================================================

def _render_chunks(chunks: list[SourceChunk]) -> str:
    if not chunks:
        return "(không có đoạn văn bản nào)"
    lines = []
    for c in chunks:
        text = (c.text or "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "…"
        lines.append(f"[{c.chunk_id}] {text}")
    return "\n".join(lines)


def build_r1_prompt(
    role: CriticRole,
    work_title: str | None,
    author: str | None,
    context_summary: str,
    chunks: list[SourceChunk],
    judge_feedback: str = "",
) -> str:
    """Human prompt vòng 1: critic đọc đoạn văn bản chung + tóm tắt Tool 1 rồi nêu luận điểm."""
    display = CRITIC_DISPLAY[role]
    header = f"Tác phẩm: {work_title or '(chưa rõ)'}"
    if author:
        header += f" — Tác giả: {author}"
    ctx = (context_summary or "").strip()
    ctx_block = f"\nTóm tắt ngữ cảnh đã chuẩn bị:\n{ctx}\n" if ctx else ""
    fb = (judge_feedback or "").strip()
    # Feedback là 1 chuỗi CHUNG cho cả debate (không tách theo từng critic) -> phải
    # cảnh báo rõ "chỉ áp dụng nếu đúng với phần của bạn" để 3 critic KHÔNG liên quan
    # không tự suy diễn lỗi hoặc lấn sang chuyên môn khác (tránh ảo giác).
    fb_block = (
        f"\nGóp ý của giám khảo ở lượt debate TRƯỚC (lượt này bị yêu cầu làm lại):\n{fb}\n"
        f"Chỉ điều chỉnh nếu góp ý trên ĐÚNG với phần việc của bạn ({display}); nếu không "
        f"liên quan tới góc nhìn của bạn thì bỏ qua, KHÔNG tự suy diễn thêm lỗi hoặc sửa "
        f"sang chuyên môn khác.\n"
    ) if fb else ""
    return (
        f"{header}\n{ctx_block}{fb_block}\n"
        f"Các đoạn văn bản gốc (dùng làm dẫn chứng):\n"
        f"{_render_chunks(chunks)}\n\n"
        f"Nhiệm vụ ({display}):\n"
        f"- Nêu 1 luận đề (thesis) ngắn gọn thể hiện góc nhìn chuyên môn của bạn.\n"
        f"- BẮT BUỘC đưa ra ÍT NHẤT {MIN_R1_ARGS} luận điểm;"
        f" Mỗi luận điểm  gồm 'point' (khẳng định) và 'support' (diễn giải, bám vào dẫn chứng ở trên). "
        f"KHÔNG để trống danh sách luận điểm.\n"
        f"- Chỉ phân tích trong đúng chuyên môn của bạn, không lấn sang góc khác.\n"
        f"- Viết bằng tiếng Việt."
    )


def _render_bulletin(bulletin: list[BulletinEntry], exclude: CriticRole) -> str:
    blocks = []
    for e in bulletin:
        if e.critic == exclude:
            continue
        pts = "\n".join(f"  {i}. {p}" for i, p in enumerate(e.key_points, 1)) or "  (không có luận điểm)"
        blocks.append(
            f"### {CRITIC_DISPLAY[e.critic]} ({e.critic.value})\n"
            f"Luận đề: {e.thesis}\n{pts}"
        )
    return "\n\n".join(blocks) if blocks else "(bảng tin trống)"


def build_r2_prompt(
    role: CriticRole,
    own_thesis: str,
    bulletin: list[BulletinEntry],
) -> str:
    """Human prompt vòng 2: critic đọc Bulletin chung rồi phản biện chéo 3 critic KHÁC."""
    display = CRITIC_DISPLAY[role]
    others = [e.critic.value for e in bulletin if e.critic != role]
    targets = ", ".join(others) if others else "(không có)"
    return (
        f"Luận đề vòng 1 của chính bạn ({display}):\n{own_thesis or '(trống)'}\n\n"
        f"BẢNG TIN CHUNG — luận đề & luận điểm của các nhà phê bình KHÁC:\n"
        f"{_render_bulletin(bulletin, role)}\n\n"
        f"Nhiệm vụ ({display}):\n"
        f"- Phản biện các nhà phê bình KHÁC (KHÔNG phản biện chính mình).\n"
        f"- Mỗi phản biện gồm: 'target_critic' (chọn trong: {targets}), 'target_point' "
        f"(SỐ THỨ TỰ luận điểm của critic đó mà bạn nhắm tới, theo bảng tin trên), 'stance' "
        f"(agree | disagree | qualify), và 'reason' (lý do ngắn, bám chuyên môn của bạn).\n"
        f"- Đưa ra 2–3 phản biện.\n"
        f"- Viết bằng tiếng Việt."
    )


# =============================================================================
# "Nói" 1 lượt critic (gọi LLM structured + ráp CriticTurn, có fallback)
# =============================================================================

def _speak_r1(role: CriticRole, state: DebateSubState, llm) -> CriticTurn:
    msgs = [
        SystemMessage(content=CRITIC_PERSONAS[role]),
        HumanMessage(content=build_r1_prompt(
            role,
            state.get("work_title"),
            state.get("author"),
            state.get("context_summary", ""),
            state.get("chunks", []),
            state.get("judge_feedback", ""),
        )),
    ]
    try:
        structured = llm.with_structured_output(CriticR1Out)
        out: CriticR1Out = structured.invoke(msgs)
        if len(out.arguments) < MIN_R1_ARGS:
            # temp=0 -> retry y hệt prompt sẽ lặp kết quả; kèm câu nhắc (đổi input) để thử thêm 1 lần.
            logger.info("R1 %s chỉ %d luận điểm -> nhắc lại 1 lần", role.value, len(out.arguments))
            retry = structured.invoke(msgs + [HumanMessage(content=(
                f"Bạn mới nêu {len(out.arguments)} luận điểm. BẮT BUỘC nêu ĐỦ tối thiểu "
                f"{MIN_R1_ARGS} luận điểm, mỗi luận điểm có 'point' và 'support' bám dẫn chứng."
            ))])
            if len(retry.arguments) > len(out.arguments):   # best-effort: giữ bản nhiều luận điểm hơn
                out = retry
        args = [
            Argument(arg_id=f"{role.value}-a{i}", point=a.point, support=a.support)
            for i, a in enumerate(out.arguments, 1)
        ]
        return CriticTurn(
            critic=role, round=1, thesis=out.thesis, arguments=args,
            raw_output=out.model_dump_json(), parsed_ok=True, spoke_at=_now(),
        )
    except Exception as e:
        logger.warning("R1 %s structured lỗi (%s) -> fallback text", role.value, e)
        raw = _fallback_text(llm, msgs)
        return CriticTurn(
            critic=role, round=1, thesis=raw[:500],
            raw_output=raw, parsed_ok=False, spoke_at=_now(),
        )


def _speak_r2(role: CriticRole, own_thesis: str,
              bulletin: list[BulletinEntry], llm) -> CriticTurn:
    msgs = [
        SystemMessage(content=CRITIC_PERSONAS[role]),
        HumanMessage(content=build_r2_prompt(role, own_thesis, bulletin)),
    ]
    try:
        out: CriticR2Out = llm.with_structured_output(CriticR2Out).invoke(msgs)
        valid_arg_ids = {aid for e in bulletin for aid in e.arg_ids}
        rebs = [
            Rebuttal(
                target_critic=r.target_critic,
                target_arg_id=_resolve_arg_id(r.target_critic, r.target_point, valid_arg_ids),
                stance=r.stance, reason=r.reason,
            )
            for r in out.rebuttals
            if r.target_critic != role  # bỏ trường hợp model lỡ tự phản biện mình
        ]
        return CriticTurn(
            critic=role, round=2, bulletin_seen=True, thesis=own_thesis,
            rebuttals=rebs, raw_output=out.model_dump_json(),
            parsed_ok=True, spoke_at=_now(),
        )
    except Exception as e:
        logger.warning("R2 %s structured lỗi (%s) -> fallback text", role.value, e)
        raw = _fallback_text(llm, msgs)
        return CriticTurn(
            critic=role, round=2, bulletin_seen=True, thesis=own_thesis,
            rebuttals=[], raw_output=raw, parsed_ok=False, spoke_at=_now(),
        )


# =============================================================================
# Node của subgraph (factory bắt sẵn role + llm -> tránh late-binding trong loop)
# =============================================================================

def make_r1_node(role: CriticRole, llm):
    def _node(state: DebateSubState) -> dict:
        return {"round1": {role: _speak_r1(role, state, llm)}}
    return _node


def make_r2_node(role: CriticRole, llm):
    def _node(state: DebateSubState) -> dict:
        own = state.get("round1", {}).get(role)
        own_thesis = own.thesis if own else ""
        return {"round2": {role: _speak_r2(role, own_thesis, state.get("bulletin", []), llm)}}
    return _node


def bulletin_node(state: DebateSubState) -> dict:
    """Barrier sau R1: parse 4 lượt R1 thành Bulletin chung cho R2 đọc."""
    r1 = state.get("round1", {})
    bulletin = []
    for role in CRITIC_ORDER:
        turn = r1.get(role)
        if not turn:
            continue
        bulletin.append(BulletinEntry(
            critic=role,
            thesis=turn.thesis,
            key_points=[a.point for a in turn.arguments],
            arg_ids=[a.arg_id for a in turn.arguments],
        ))
    return {"bulletin": bulletin}


def collect_node(state: DebateSubState) -> dict:
    """Barrier sau R2: gom điểm đồng thuận / tranh cãi từ stance của các rebuttal."""
    r2 = state.get("round2", {})
    consensus, contested = [], []
    for role in CRITIC_ORDER:
        turn = r2.get(role)
        if not turn:
            continue
        for reb in turn.rebuttals:
            tgt = CRITIC_DISPLAY.get(reb.target_critic, str(reb.target_critic))
            line = f"{CRITIC_DISPLAY[role]} → {tgt}: {reb.reason}"
            if reb.stance == "agree":
                consensus.append(line)
            elif reb.stance == "disagree":
                contested.append(line)
    return {"consensus_points": consensus, "contested_points": contested}


# =============================================================================
# Dựng subgraph (inject llm) + node public
# =============================================================================

def build_debate_subgraph(llm):
    """Graph này chạy theo cơ chế Superstep (BSP) tuân thủ :
    Các node chạy song song
    Có rào chắn đồng bộ chờ đủ node
    có giai đoạn merge các node
    """
    g = StateGraph(DebateSubState)
    for role in CRITIC_ORDER:
        g.add_node(f"{role.value}_r1", make_r1_node(role, llm))
        # gọi wapper trong waper trả về (return _node) chính là địa chỉ của hàm
        # giống  g.add_node("supervisor", supervisor)
        # make_r1_node(role, llm) ở đây thực chất là 1 lệnh thực thi hàm chứ không phải là lưu địa chỉ cho LangGraph gọi
        # và LangGraph chỉ nhận địa chỉ của hàm có tham số truyền vào duy nhất là State
    g.add_node("bulletin", bulletin_node)
    for role in CRITIC_ORDER:
        g.add_node(f"{role.value}_r2", make_r2_node(role, llm))
    g.add_node("collect", collect_node)

    for role in CRITIC_ORDER: # fan-in
        g.add_edge(START, f"{role.value}_r1")       # 4 R1 chạy song song
        g.add_edge(f"{role.value}_r1", "bulletin")  # 4 node đều trỏ vào bulletin -> rào chắn chờ 4 node hoàn thành
    for role in CRITIC_ORDER:
        g.add_edge("bulletin", f"{role.value}_r2")  # 4 R2 chạy song song
        g.add_edge(f"{role.value}_r2", "collect")   # collect chờ đủ 4 (barrier)
    g.add_edge("collect", END)
    return g.compile()


_PROD_APP = None


def _prod_subgraph():
    """Lazy load, request đầu tiên phải chịu vì cơ chế này ."""
    global _PROD_APP
    if _PROD_APP is None:
        from providers.ollama_provider import ollama_provider
        # temperature > 0 (thay vì 0.0 mặc định) để retry (feedback giám khảo) không
        # sinh ra output y hệt lần trước; chỉ áp dụng ở đây, không đụng factual/chat.
        _PROD_APP = build_debate_subgraph(ollama_provider.get_llm(temperature=0.3))
    return _PROD_APP


def _chunks_from_context(context) -> list[SourceChunk]:
    """Lấy đoạn văn bản chung từ PreparedContext của Tool 1 (chunks, fallback key_passages)."""
    if context is None:
        return []
    chunks = list(getattr(context, "chunks", None) or [])
    # getattr = get attribute | getattr(object, tên_thuộc_tính, giá_trị_mặc_định)
    if not chunks:
        chunks = list(getattr(context, "key_passages", None) or [])
    return chunks


def critics_debate(state, *, subgraph=None) -> dict:
    """Node public (Tool 2). Đọc intent + context (Tool 1), chạy subgraph, ghi state['debate'].

    subgraph: inject để test; None -> dùng subgraph production (lazy).
    """
    intent = state.get("intent")
    context = state.get("context")
    judge_feedback = (state.get("last_feedback") or {}).get(Stage.CRITICS_DEBATE.value, "")
    sub_in: DebateSubState = {
        "work_title": getattr(intent, "work_title", None),
        "author": getattr(intent, "author", None),
        "context_summary": getattr(context, "summary", "") or "",
        "chunks": _chunks_from_context(context),
        "judge_feedback": judge_feedback,
        "round1": {}, "round2": {}, "bulletin": [],
        "consensus_points": [], "contested_points": [],
    }
    app = subgraph or _prod_subgraph()
    result = app.invoke(sub_in)

    r1 = result.get("round1", {})
    r2 = result.get("round2", {})
    debate = DebateState(
        round1=r1,
        bulletin=result.get("bulletin", []),
        round2=r2,
        consensus_points=result.get("consensus_points", []),
        contested_points=result.get("contested_points", []),
        total_invocations=len(r1) + len(r2),
    )
    logger.info("Tool2 critics_debate xong: R1=%d R2=%d", len(r1), len(r2))
    return {
        "debate": debate,
        "current_stage": Stage.CRITICS_DEBATE,
        "current_node": "critics_debate",
    }
