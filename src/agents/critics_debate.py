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
from config.ui_theme import ui_meta
from state.state_schema import (
    CRITIC_DISPLAY,
    Argument,
    BulletinEntry,
    CriticRole,
    CriticTurn,
    DebateState,
    EventEmitter,
    EventType,
    Rebuttal,
    SourceChunk,
    Stage,
    StreamEvent,
    merge_dict,
    safe_stream_writer,
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
    question: str                     # ĐỀ BÀI: câu hỏi gốc của người dùng — cả 2 vòng phải bám
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


async def _fallback_text(llm, msgs) -> str:
    try:
        return _extract_text((await llm.ainvoke(msgs)).content)
    except Exception as e:  # pragma: no cover - đường lỗi kép
        return f"[lỗi gọi model: {e}]"


def _normalize_arg_id(raw: str) -> str:
    """Chuẩn hoá id model trả về trước khi tra bulletin.

    Bảng tin in id trong ngoặc vuông ("[lich_su-a1]") cho dễ đọc, và Qwen chép NGUYÊN CẢ
    NGOẶC. So thẳng với arg_ids (không ngoặc) sẽ trượt 100% -> mọi rebuttal bị loại ->
    vòng 2 rỗng trơn. Nhận cả 2 dạng thay vì bắt model đoán ý.
    """
    return (raw or "").strip().strip("[]").strip().lower()


def _target_from_arg_id(arg_id: str, valid_ids: set[str]) -> Optional[CriticRole]:
    """arg_id ("lich_su-a2" hoặc "[lich_su-a2]") -> critic BỊ phản biện; None nếu id không có thật.

    Attribution suy TỪ id, không tin field riêng của model: khi model điền tách rời
    (target_critic, target_point) thì 2 field có thể lệch nhau mà validate không bắt được
    (số thứ tự luận điểm reset theo từng critic -> "point 2" của critic nào cũng tồn tại),
    sinh ra ca "bắt bẻ luận điểm của Lịch sử nhưng FE in là Trả lời Tiếp nhận".
    """
    aid = _normalize_arg_id(arg_id)
    if aid not in valid_ids:
        return None
    try:
        return CriticRole(aid.rsplit("-a", 1)[0])
    except ValueError:
        return None


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


def _question_block(question: str) -> str:
    """Đề bài của người dùng — đặt ĐẦU prompt ở CẢ 2 vòng.

    Thiếu khối này, 4 critic phân tích tác phẩm chung chung theo chuyên môn của mình
    thay vì trả lời đúng câu người dùng hỏi.
    """
    q = (question or "").strip()
    return f"CÂU HỎI CỦA NGƯỜI DÙNG (bám sát yêu cầu này):\n{q}\n\n" if q else ""


def build_r1_prompt(
    role: CriticRole,
    work_title: str | None,
    author: str | None,
    context_summary: str,
    chunks: list[SourceChunk],
    judge_feedback: str = "",
    question: str = "",
) -> str:
    """Human prompt vòng 1: critic đọc đề bài + đoạn văn bản chung + tóm tắt Tool 1 rồi nêu luận điểm."""
    display = CRITIC_DISPLAY[role]
    header = f"{_question_block(question)}Tác phẩm: {work_title or '(chưa rõ)'}"
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
        f"- Nêu 1 luận đề (thesis) ngắn gọn, trả lời thẳng CÂU HỎI CỦA NGƯỜI DÙNG từ "
        f"góc nhìn chuyên môn của bạn.\n"
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
        # Hiện arg_id (duy nhất toàn bảng tin) thay cho số thứ tự (reset theo từng critic,
        # dễ lẫn giữa các khối) -> model chỉ việc CHÉP LẠI id, không phải tự ghép critic + số.
        # Kèm 'Lý lẽ' (support): critic phải ĐỌC ĐƯỢC lập luận thì mới bắt bẻ được nó; chỉ
        # đưa khẳng định trần trụi -> R2 không có gì để phản biện, quay ra nhắc lại luận đề mình.
        lines = []
        for i, (aid, p) in enumerate(zip(e.arg_ids, e.key_points)):
            # tra supports theo index (không zip 3 chiều): bulletin lưu trong checkpoint CŨ
            # chưa có field supports -> zip sẽ cắt cụt cả bảng tin về rỗng.
            s = e.supports[i] if i < len(e.supports) else ""
            lines.append(f"  [{aid}] {p}")
            if s:
                lines.append(f"      Lý lẽ: {s}")
        pts = "\n".join(lines) or "  (không có luận điểm)"
        blocks.append(
            f"### {CRITIC_DISPLAY[e.critic]} ({e.critic.value})\n"
            f"Luận đề: {e.thesis}\n{pts}"
        )
    return "\n\n".join(blocks) if blocks else "(bảng tin trống)"


def build_r2_prompt(
    role: CriticRole,
    own_thesis: str,
    bulletin: list[BulletinEntry],
    chunks: list[SourceChunk],
    question: str = "",
) -> str:
    """Human prompt vòng 2: critic đọc lại ĐỀ BÀI + VĂN BẢN GỐC + Bulletin rồi phản biện 3 critic KHÁC.

    chunks BẮT BUỘC có mặt: thiếu văn bản gốc, critic không đối chiếu được dẫn chứng nên
    chỉ còn nước phát biểu lại luận đề vòng 1 của mình -> "phản biện" hoá ra là nêu quan
    điểm cá nhân, không bắt bẻ được ai.
    """
    display = CRITIC_DISPLAY[role]
    return (
        f"{_question_block(question)}"
        f"VĂN BẢN GỐC (nguồn dẫn chứng DUY NHẤT — mọi phản biện phải đối chiếu với đây):\n"
        f"{_render_chunks(chunks)}\n\n"
        f"Luận đề vòng 1 của chính bạn ({display}):\n{own_thesis or '(trống)'}\n\n"
        f"BẢNG TIN CHUNG — luận đề & luận điểm của các nhà phê bình KHÁC.\n"
        f"Mỗi luận điểm có một ID trong ngoặc vuông, vd [lich_su-a2]:\n"
        f"{_render_bulletin(bulletin, role)}\n\n"
        f"Nhiệm vụ ({display}) — PHẢN BIỆN, không phải trình bày lại quan điểm của bạn:\n"
        f"- Với mỗi luận điểm bạn nhắm tới, hãy KIỂM TRA nó bằng VĂN BẢN GỐC ở trên: lý lẽ "
        f"của họ có được văn bản chống đỡ không? Họ có suy diễn điều văn bản không nói, "
        f"đọc sai chi tiết, hay bỏ sót chi tiết làm họ sai không?\n"
        f"- 'reason' phải CHỈ RA CHỖ HỎNG trong lập luận CỦA HỌ và trích cụm từ/câu cụ thể "
        f"trong văn bản gốc làm bằng. CẤM chỉ nhắc lại luận đề của bạn rồi coi đó là phản "
        f"biện (vd 'bài thơ thật ra nói về X' mà không đụng tới lý lẽ của họ).\n"
        f"- Nếu lý lẽ của họ ĐỨNG VỮNG trước văn bản, hãy dùng stance 'agree' và nói rõ vì "
        f"sao nó đứng vững — đừng phản đối lấy lệ.\n"
        f"- Mỗi phản biện gồm: 'target_arg_id' (CHÉP NGUYÊN VĂN id trong ngoặc vuông của "
        f"luận điểm bạn nhắm tới — phải là id CÓ THẬT trong bảng tin trên, KHÔNG tự bịa, "
        f"KHÔNG dùng id của chính bạn), 'stance' (agree | disagree | qualify), và 'reason'.\n"
        f"- 'reason' phải nói đúng về luận điểm mang id bạn đã chọn.\n"
        f"- Đưa ra 2–3 phản biện.\n"
        f"- Viết bằng tiếng Việt."
    )


# =============================================================================
# "Nói" 1 lượt critic (gọi LLM structured + ráp CriticTurn, có fallback)
# =============================================================================

async def _speak_r1(role: CriticRole, state: DebateSubState, llm) -> CriticTurn:
    msgs = [
        SystemMessage(content=CRITIC_PERSONAS[role]),
        HumanMessage(content=build_r1_prompt(
            role,
            state.get("work_title"),
            state.get("author"),
            state.get("context_summary", ""),
            state.get("chunks", []),
            state.get("judge_feedback", ""),
            state.get("question", ""),
        )),
    ]
    try:
        structured = llm.with_structured_output(CriticR1Out)
        out: CriticR1Out = await structured.ainvoke(msgs)
        if len(out.arguments) < MIN_R1_ARGS:
            # temp=0 -> retry y hệt prompt sẽ lặp kết quả; kèm câu nhắc (đổi input) để thử thêm 1 lần.
            logger.info("R1 %s chỉ %d luận điểm -> nhắc lại 1 lần", role.value, len(out.arguments))
            retry = await structured.ainvoke(msgs + [HumanMessage(content=(
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
        raw = await _fallback_text(llm, msgs)
        return CriticTurn(
            critic=role, round=1, thesis=raw[:500],
            raw_output=raw, parsed_ok=False, spoke_at=_now(),
        )


async def _speak_r2(role: CriticRole, state: DebateSubState, llm) -> CriticTurn:
    own = state.get("round1", {}).get(role)
    own_thesis = own.thesis if own else ""
    bulletin = state.get("bulletin", [])
    msgs = [
        SystemMessage(content=CRITIC_PERSONAS[role]),
        HumanMessage(content=build_r2_prompt(
            role, own_thesis, bulletin,
            state.get("chunks", []),
            state.get("question", ""),
        )),
    ]
    try:
        out: CriticR2Out = await llm.with_structured_output(CriticR2Out).ainvoke(msgs)
        valid_arg_ids = {aid for e in bulletin for aid in e.arg_ids}
        rebs = []
        for r in out.rebuttals:
            target = _target_from_arg_id(r.target_arg_id, valid_arg_ids)
            # id bịa (không có trong bulletin) hoặc id của chính mình -> bỏ hẳn phản biện đó,
            # KHÔNG đoán bừa critic bị nhắm: đoán sai chính là ca gắn nhầm người ở FE.
            if target is None or target == role:
                logger.warning("R2 %s: BỎ phản biện, target_arg_id=%r không khớp bulletin %s",
                               role.value, r.target_arg_id, sorted(valid_arg_ids))
                continue
            rebs.append(Rebuttal(
                target_critic=target, target_arg_id=_normalize_arg_id(r.target_arg_id),
                stance=r.stance, reason=r.reason,
            ))
        if out.rebuttals and not rebs:
            # Cả lượt bị loại sạch -> vòng 2 sẽ rỗng trơn trên FE (trông y hệt vòng 1).
            # Log to để lần sau thấy ngay thay vì phải soi UI mới biết.
            logger.warning("R2 %s: LOẠI SẠCH %d phản biện -> lượt này trống.",
                           role.value, len(out.rebuttals))
        return CriticTurn(
            critic=role, round=2, bulletin_seen=True, thesis=own_thesis,
            rebuttals=rebs, raw_output=out.model_dump_json(),
            parsed_ok=True, spoke_at=_now(),
        )
    except Exception as e:
        logger.warning("R2 %s structured lỗi (%s) -> fallback text", role.value, e)
        raw = await _fallback_text(llm, msgs)
        return CriticTurn(
            critic=role, round=2, bulletin_seen=True, thesis=own_thesis,
            rebuttals=[], raw_output=raw, parsed_ok=False, spoke_at=_now(),
        )


# =============================================================================
# Emit LIVE ra stream ngay khi 1 lượt critic xong (Phase 3 — tiến trình thật).
# Node subgraph gọi safe_stream_writer() để lấy writer của run CHA (contextvar tự
# truyền qua subgraph.invoke; cần astream(subgraphs=True) — xác nhận ở Phase 0 spike).
# seq ở đây là BEST-EFFORT (R1: 1..4, bulletin: 5, R2: 11..14) — FE sắp LIVE theo `ts`.
# Bản PERSIST (seq tuần tự nối tiếp state cha) do node public critics_debate() dựng lại.
# =============================================================================

def _turn_payload(round_no: int, turn: CriticTurn) -> dict:
    """Data nghiệp vụ của 1 lượt critic -> payload (dùng chung cho live & persist)."""
    return {
        "round": round_no,
        "parsed_ok": turn.parsed_ok,
        "arguments": [{"arg_id": a.arg_id, "point": a.point, "support": a.support}
                      for a in turn.arguments],
        "rebuttals": [{"target_critic": r.target_critic.value, "stance": r.stance,
                       "reason": r.reason} for r in turn.rebuttals],
    }


def _emit_live_turn(writer, role: CriticRole, round_no: int, turn: CriticTurn) -> None:
    if writer is None:
        return
    payload = _turn_payload(round_no, turn)
    payload["ui"] = ui_meta(EventType.CRITIC_TURN, role=role)
    ev = StreamEvent(
        seq=(0 if round_no == 1 else 10) + CRITIC_ORDER.index(role) + 1,
        type=EventType.CRITIC_TURN,
        node=f"critic:{role.value}:r{round_no}",
        actor=CRITIC_DISPLAY[role],
        content=turn.thesis or "(chưa parse được luận điểm)",
        payload=payload,
        ts=_now(),
    )
    writer(ev.model_dump(mode="json"))


def _emit_live_bulletin(writer, bulletin: list[BulletinEntry]) -> None:
    if writer is None:
        return
    ev = StreamEvent(
        seq=5, type=EventType.BULLETIN, node="debate:bulletin",
        content="Bảng tin chung đã sẵn sàng.",
        payload={
            "entries": [{"critic": e.critic.value, "thesis": e.thesis,
                         "key_points": e.key_points} for e in bulletin],
            "ui": ui_meta(EventType.BULLETIN),
        },
        ts=_now(),
    )
    writer(ev.model_dump(mode="json"))


# =============================================================================
# Node của subgraph (factory bắt sẵn role + llm -> tránh late-binding trong loop)
# =============================================================================

def make_r1_node(role: CriticRole, llm):
    async def _node(state: DebateSubState) -> dict:
        turn = await _speak_r1(role, state, llm)
        _emit_live_turn(safe_stream_writer(), role, 1, turn)   # bắn LIVE ngay khi lượt xong
        return {"round1": {role: turn}}
    return _node


def make_r2_node(role: CriticRole, llm):
    async def _node(state: DebateSubState) -> dict:
        turn = await _speak_r2(role, state, llm)
        _emit_live_turn(safe_stream_writer(), role, 2, turn)
        return {"round2": {role: turn}}
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
            supports=[a.support for a in turn.arguments],
            arg_ids=[a.arg_id for a in turn.arguments],
        ))
    _emit_live_bulletin(safe_stream_writer(), bulletin)
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


async def critics_debate(state, *, subgraph=None) -> dict:
    """Node public (Tool 2). Đọc intent + context (Tool 1), chạy subgraph, ghi state['debate'].

    subgraph: inject để test; None -> dùng subgraph production (lazy).

    async: 8 lượt critic là phần chiếm Ollama lâu nhất; chạy async thì client ngắt kết nối
    (F5) sẽ huỷ được các request Ollama đang bay, không để chúng cày tiếp và chẹn lượt chat sau.
    """
    intent = state.get("intent")
    context = state.get("context")
    judge_feedback = (state.get("last_feedback") or {}).get(Stage.CRITICS_DEBATE.value, "")
    # Đề bài = human_message (câu người dùng gõ), KHÔNG dùng intent.raw_query: api/debate_router.py
    # gán raw_query = work_title, lấy nhầm sẽ in "CÂU HỎI CỦA NGƯỜI DÙNG: Tỏ lòng".
    # Cùng nguồn với write_essay.py và factual_node.py.
    sub_in: DebateSubState = {
        "question": state.get("human_message", "") or "",
        "work_title": getattr(intent, "work_title", None),
        "author": getattr(intent, "author", None),
        "context_summary": getattr(context, "summary", "") or "",
        "chunks": _chunks_from_context(context),
        "judge_feedback": judge_feedback,
        "round1": {}, "round2": {}, "bulletin": [],
        "consensus_points": [], "contested_points": [],
    }
    app = subgraph or _prod_subgraph()
    result = await app.ainvoke(sub_in)

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

    # không bắn SEE mục đích ở hàm này là lưu lại presist milestone vào state.events
    emitter = EventEmitter(state, writer=None)
    for role in CRITIC_ORDER:
        turn = r1.get(role)
        if turn:
            emitter.critic_turn(f"critic:{role.value}:r1", role,
                                turn.thesis or "(chưa parse được luận điểm)",
                                actor=CRITIC_DISPLAY[role], payload=_turn_payload(1, turn))
    emitter.bulletin(
        "debate:bulletin", "Bảng tin chung đã sẵn sàng.",
        payload={"entries": [{"critic": e.critic.value, "thesis": e.thesis,
                              "key_points": e.key_points} for e in debate.bulletin]},
    )
    for role in CRITIC_ORDER:
        turn = r2.get(role)
        if turn:
            emitter.critic_turn(f"critic:{role.value}:r2", role,
                                turn.thesis or "(phản biện)",
                                actor=CRITIC_DISPLAY[role], payload=_turn_payload(2, turn))
    return {
        "debate": debate,
        "current_stage": Stage.CRITICS_DEBATE,
        "current_node": "critics_debate",
        "events": emitter.milestones,
        "event_seq": emitter.seq,
    }
