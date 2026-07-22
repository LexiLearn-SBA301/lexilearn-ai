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

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.debate_schemas import CriticR1Out, CriticR2Out
from config.critic_prompts import CRITIC_PERSONAS
from config.ui_theme import ui_meta
from services.agent_service import debate_session
from services.agent_service.debate_session import HumanReply, normalize_arg_id
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
    # 2 field dưới phục vụ "tranh luận cùng người học": node human_r1/human_r2 dùng
    # thread_id làm khoá phiên chờ (debate_session). human_in_debate ĐỌC 1 LẦN ở node
    # public rồi truyền xuống -> hai node human đọc CÙNG một giá trị, không tự tra lại
    # cờ (tra lại = bỏ qua vòng 1 xong là mất luôn quyền vào vòng 2).
    thread_id: str
    human_in_debate: bool
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


def _target_from_arg_id(arg_id: str, valid_ids: set[str]) -> Optional[CriticRole]:
    """arg_id ("lich_su-a2" hoặc "[lich_su-a2]") -> critic BỊ phản biện; None nếu id không có thật.

    Attribution suy TỪ id, không tin field riêng của model: khi model điền tách rời
    (target_critic, target_point) thì 2 field có thể lệch nhau mà validate không bắt được
    (số thứ tự luận điểm reset theo từng critic -> "point 2" của critic nào cũng tồn tại),
    sinh ra ca "bắt bẻ luận điểm của Lịch sử nhưng FE in là Trả lời Tiếp nhận".
    """
    aid = normalize_arg_id(arg_id)
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
        # Bỏ hẳn dòng "Luận đề" khi trống thay vì in nhãn cụt. Lượt của NGƯỜI HỌC không có
        # luận đề (họ gõ thẳng từng luận điểm, không qua LLM tóm ý) -> in "Luận đề: " rỗng
        # trông như dữ liệu hỏng và khiến model tưởng họ chẳng có quan điểm gì.
        head = f"### {CRITIC_DISPLAY[e.critic]} ({e.critic.value})"
        if e.thesis:
            head += f"\nLuận đề: {e.thesis}"
        blocks.append(f"{head}\n{pts}")
    return "\n\n".join(blocks) if blocks else "(bảng tin trống)"


def build_r2_prompt(
    role: CriticRole,
    own_thesis: str,
    bulletin: list[BulletinEntry],
    chunks: list[SourceChunk],
    question: str = "",
    has_human: bool = False,
) -> str:
    """Human prompt vòng 2: critic đọc lại ĐỀ BÀI + VĂN BẢN GỐC + Bulletin rồi phản biện những
    người tham gia KHÁC (3 critic còn lại, cộng NGƯỜI HỌC nếu họ có phát biểu ở vòng 1).

    chunks BẮT BUỘC có mặt: thiếu văn bản gốc, critic không đối chiếu được dẫn chứng nên
    chỉ còn nước phát biểu lại luận đề vòng 1 của mình -> "phản biện" hoá ra là nêu quan
    điểm cá nhân, không bắt bẻ được ai.

    has_human: BẮT BUỘC suy từ bulletin CÓ THẬT entry của người học hay không (xem _speak_r2),
    KHÔNG phải từ cờ opt-in. Người học bật nút rồi bấm "Bỏ qua" -> opt-in vẫn True nhưng bảng
    tin KHÔNG có id 'human-...'; ra lệnh nhắm vào id không tồn tại thì model buộc phải BỊA id
    -> _target_from_arg_id() loại sạch -> vòng 2 của critic đó trống trơn.
    """
    display = CRITIC_DISPLAY[role]
    others = "những người tham gia KHÁC (gồm cả NGƯỜI HỌC)" if has_human else "các nhà phê bình KHÁC"
    # Tiêu chí ĐẾM ĐƯỢC ("ĐÚNG 1 phản biện nhắm id human-...") thay vì lời khuyên thái độ
    # ("nhớ để ý người học"): lời khuyên thì model gật rồi bỏ qua, còn cái này verify được
    # bằng code y như cách _speak_r2 lọc target_arg_id.
    #
    # Phải có CẬN TRÊN, không chỉ cận dưới. Đo thật với qwen2.5:3b: ra lệnh "ít nhất 1" thì
    # nó dồn 3/3 phản biện vào người học và bỏ hẳn 5 luận điểm của critic khác. Nhân với 4
    # critic = người học lãnh cả chục phản biện còn tranh luận AI–AI biến mất — mà chính
    # phần đó mới là nguyên liệu cho bài luận.
    human_clause = (
        f"- Trong bảng tin có luận điểm của NGƯỜI HỌC (id bắt đầu bằng 'human-'). BẮT BUỘC "
        f"dành ÍT NHẤT 1 phản biện nhắm vào id 'human-...'; những phản biện "
        f"còn lại PHẢI nhắm vào nhà phê bình khác. Đối xử với người học đúng như một đồng "
        f"nghiệp trong hội đồng: đọc kỹ lý lẽ, đối chiếu VĂN BẢN GỐC, công nhận phần đứng "
        f"vững và chỉ thẳng phần chưa vững. KHÔNG khen lấy lệ, cũng KHÔNG hạ thấp.\n"
        f"- Nếu luận điểm 'human-...' dựa vào chi tiết KHÔNG có trong VĂN BẢN GỐC, hãy nói "
        f"THẲNG điều đó trong 'reason' (stance 'disagree' hoặc 'qualify') và chỉ ra văn bản "
        f"thực sự cho thấy gì. Đó là phản biện hữu ích NHẤT cho người học — tuyệt đối KHÔNG "
        f"được bỏ qua họ chỉ vì khó tìm cụm từ đối chiếu.\n"
    ) if has_human else ""
    return (
        f"{_question_block(question)}"
        f"VĂN BẢN GỐC (nguồn dẫn chứng DUY NHẤT — mọi phản biện phải đối chiếu với đây):\n"
        f"{_render_chunks(chunks)}\n\n"
        f"Luận đề vòng 1 của chính bạn ({display}):\n{own_thesis or '(trống)'}\n\n"
        f"BẢNG TIN CHUNG — luận đề & luận điểm của {others}.\n"
        f"Mỗi luận điểm có một ID trong ngoặc vuông, vd [lich_su-a2]:\n"
        f"{_render_bulletin(bulletin, role)}\n\n"
        f"Nhiệm vụ ({display}) — ĐỐI THOẠI với lập luận của họ, không trình bày lại quan điểm "
        f"của bạn, cũng không phủ định lấy lệ:\n"
        f"- Với mỗi luận điểm bạn nhắm tới, hãy ĐỌC LÝ LẼ của họ rồi đối chiếu VĂN BẢN GỐC.\n"
        f"- Chọn stance TRUNG THỰC theo những gì văn bản cho thấy:\n"
        f"    'agree'    — lý lẽ của họ đứng vững; nói rõ chi tiết nào trong văn bản chống đỡ nó.\n"
        f"    'qualify'  — họ đúng một phần; chỉ ra phần đúng và phần cần giới hạn/bổ sung.\n"
        f"    'disagree' — văn bản MÂU THUẪN với họ; phải trích cụm từ cụ thể chứng minh.\n"
        f"- CẤM dùng khuôn 'văn bản không đề cập/không chứng minh' cho mọi luận điểm. Đó là né "
        f"tranh luận. Nếu văn bản có chi tiết liên quan, hãy bàn về chi tiết ấy.\n"
        f"- CẤM chỉ nhắc lại luận đề của bạn rồi coi đó là phản biện (vd 'tác phẩm thật ra nói "
        f"về X') mà không đụng tới lý lẽ của họ.\n"
        f"- 'reason' phải trích cụm từ/câu cụ thể trong VĂN BẢN GỐC làm bằng, dù stance nào.\n"
        f"- Mỗi phản biện gồm: 'target_arg_id' (CHÉP NGUYÊN VĂN id trong ngoặc vuông của "
        f"luận điểm bạn nhắm tới — phải là id CÓ THẬT trong bảng tin trên, KHÔNG tự bịa, "
        f"KHÔNG dùng id của chính bạn), 'stance' (agree | disagree | qualify), và 'reason'.\n"
        f"- 'reason' phải nói đúng về luận điểm mang id bạn đã chọn.\n"
        f"{human_clause}"
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
    # Suy TỪ bulletin, không từ cờ opt-in: bật nút rồi bỏ qua vòng 1 = không có entry human.
    has_human = any(e.critic == CriticRole.HUMAN for e in bulletin)
    msgs = [
        SystemMessage(content=CRITIC_PERSONAS[role]),
        HumanMessage(content=build_r2_prompt(
            role, own_thesis, bulletin,
            state.get("chunks", []),
            state.get("question", ""),
            has_human=has_human,
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
                target_critic=target, target_arg_id=normalize_arg_id(r.target_arg_id),
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


def _live_seq(role: CriticRole, round_no: int) -> int:
    """
    vòng 1: 4 5 6
    vòng 2: 14 15 16
    """
    idx = CRITIC_ORDER.index(role) if role in CRITIC_ORDER else len(CRITIC_ORDER)
    return (0 if round_no == 1 else 10) + idx + 1


def _turn_content(round_no: int, role: CriticRole, turn: CriticTurn) -> str:
    """
    xử lý chuỗi an toàn không trả về luận đề lúc r2 nữa
    và trả null nếu role là người đọc ( theo thiết kế người đọc không có luận điểm )
    """
    if round_no == 2:
        return "" if turn.rebuttals else "(không đưa ra phản biện nào)"
    if turn.thesis:
        return turn.thesis
    return "" if role == CriticRole.HUMAN else "(chưa parse được luận điểm)"


def _emit_live_turn(writer, role: CriticRole, round_no: int, turn: CriticTurn) -> None:
    if writer is None:
        return
    payload = _turn_payload(round_no, turn)
    payload["ui"] = ui_meta(EventType.CRITIC_TURN, role=role)
    ev = StreamEvent(
        seq=_live_seq(role, round_no),
        type=EventType.CRITIC_TURN,
        node=f"critic:{role.value}:r{round_no}",
        actor=CRITIC_DISPLAY[role],
        content=_turn_content(round_no, role, turn),
        payload=payload,
        ts=_now(),
    )
    writer(ev.model_dump(mode="json"))


def _emit_await_human(writer, round_no: int, valid_arg_ids: set[str]) -> None:
    """Báo FE mở ô nhập. CHỈ stream, KHÔNG persist (trạng thái UI nhất thời, replay vô nghĩa).

    payload.round quyết định FE hiện giao diện nào: vòng 1 = ô text tự do (nêu luận điểm
    của mình), vòng 2 = chọn luận điểm để reply + stance. valid_arg_ids để FE chỉ cho chọn
    id CÓ THẬT — sai id thì /chat/debate/reply trả 400.
    """
    if writer is None:
        return
    ev = StreamEvent(
        seq=_live_seq(CriticRole.HUMAN, round_no),
        type=EventType.AWAIT_HUMAN,
        node=f"debate:await_human:r{round_no}",
        actor=CRITIC_DISPLAY[CriticRole.HUMAN],
        content=("Mời bạn nêu luận điểm của mình về câu hỏi này."
                 if round_no == 1 else
                 "Mời bạn phản biện các nhà phê bình."),
        payload={
            "round": round_no,
            "max_turns": debate_session.MAX_HUMAN_TURNS,
            "idle_timeout_s": debate_session.IDLE_TIMEOUT_S,
            "valid_arg_ids": sorted(valid_arg_ids),
            "ui": ui_meta(EventType.AWAIT_HUMAN),
        },
        ts=_now(),
    )
    writer(ev.model_dump(mode="json"))


def _emit_human_closed(writer, round_no: int, reason: str, content: str) -> None:
    """Đóng ô nhập ở FE (hết giờ / đã kết thúc / đã đủ số tin). Chỉ stream, không persist."""
    if writer is None:
        return
    ev = StreamEvent(
        seq=_live_seq(CriticRole.HUMAN, round_no),
        type=EventType.AWAIT_HUMAN,
        node=f"debate:await_human:r{round_no}:closed",
        actor=CRITIC_DISPLAY[CriticRole.HUMAN],
        content=content,
        payload={"round": round_no, "closed": True, "reason": reason,
                 "ui": ui_meta(EventType.AWAIT_HUMAN)},
        ts=_now(),
    )
    writer(ev.model_dump(mode="json"))


def _emit_live_bulletin(writer, bulletin: list[BulletinEntry]) -> None:
    if writer is None:
        return
    ev = StreamEvent(
        seq=6, type=EventType.BULLETIN, node="debate:bulletin",
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


# =============================================================================
# Node của NGƯỜI HỌC — điểm PAUSE. Nói SAU 4 critic mỗi vòng (đọc hết ý kiến hội đồng
# rồi mới phát biểu), nên đứng giữa fan-in R1 và bulletin (vòng 1), giữa fan-in R2 và
# collect (vòng 2).
# =============================================================================

async def _collect_human(thread_id: str, round_no: int, valid_arg_ids: set[str],
                         writer) -> list[HumanReply]:
    """PAUSE tại đây tới khi người học gửi đủ / bấm kết thúc / hết giờ im lặng.

       1. get() thấy queue rỗng → tạo một Future, để lại số điện thoại trong self._getters, rồi ngủ. Coroutine bị gỡ khỏi event loop
      hoàn toàn, không tốn CPU.
      2. put_nowait() đẩy item vào, rồi gọi lại theo số điện thoại đó (waiter.set_result(None)).
      3. Event loop thấy Future xong → xếp coroutine kia vào hàng chạy lại → get() tỉnh dậy, loop lại, thấy queue không rỗng nữa →
      get_nowait() lấy item ra.
    """
    p = debate_session.open_session(thread_id, round_no, valid_arg_ids)

    _emit_await_human(writer, round_no, valid_arg_ids)
    replies: list[HumanReply] = []
    reason, note = "ended", "Bạn đã kết thúc phần phát biểu."
    try:
        while len(replies) < debate_session.MAX_HUMAN_TURNS:
            try:
                item = await asyncio.wait_for(p.queue.get(),
                                              timeout=debate_session.IDLE_TIMEOUT_S)
            except asyncio.TimeoutError:
                reason, note = "timeout", "Hết thời gian chờ — hội đồng tiếp tục."
                logger.info("human_r%d: im lặng quá %.0fs -> đi tiếp (%d tin).",
                            round_no, debate_session.IDLE_TIMEOUT_S, len(replies))
                break
            if item is None:          # Bỏ qua / Kết thúc phản biện — cùng 1 tín hiệu
                break
            replies.append(item)
            # Bắn NGAY từng tin thay vì gom cả lượt rồi bắn một cục ở cuối: người học Enter
            # xong phải thấy tin của mình hiện lên liền, chứ không ngồi nhìn màn hình im
            _emit_live_turn(writer, CriticRole.HUMAN, round_no,
                            _human_turn(round_no, [item], valid_arg_ids,
                                        start=len(replies)))
        else:
            reason, note = "max_turns", f"Đã đủ {debate_session.MAX_HUMAN_TURNS} lượt phát biểu."
    finally:
        # Đóng phiên kể cả khi bị huỷ (người dùng bấm Dừng -> CancelledError) -> không để
        # lại phiên ma khiến submit() sau đó treo vào queue không ai đọc.
        debate_session.close_session(thread_id)
    _emit_human_closed(writer, round_no, reason, note)
    logger.info("human_r%d: nhận %d tin (%s).", round_no, len(replies), reason)
    return replies


def _human_turn(round_no: int, replies: list[HumanReply],
                valid_arg_ids: set[str], *, start: int = 1) -> CriticTurn:
    """Gói các tin của người học thành CriticTurn — CÙNG cấu trúc critic AI dùng.

    Vòng 1 -> Argument có arg_id 'human-aN': đó là thứ DUY NHẤT khiến 4 critic bắt bẻ
    ngược lại được ở vòng 2 (Rebuttal không có id nên không thể bị nhắm tới).
    thesis để TRỐNG: người học gõ thẳng từng luận điểm, không qua LLM tóm ý; gán tin đầu
    làm luận đề thì nó bị in 2 lần (vừa luận đề vừa human-a1) ở cả prompt lẫn UI.

    Vòng 2 -> Rebuttal: target_arg_id + stance đã được submit() validate rồi.

    start: số thứ tự arg_id của tin ĐẦU trong `replies`. Cần vì hàm này được gọi 2 kiểu —
    gói CẢ lượt (start=1, dựng state) và gói ĐÚNG 1 tin vừa nhận để bắn live ngay
    (start=vị trí thật). Thiếu nó thì mọi tin bắn live đều mang id 'human-a1'.
    """
    if round_no == 1:
        args = [Argument(arg_id=f"human-a{i}", point=r.message, support="")
                for i, r in enumerate(replies, start)]
        return CriticTurn(critic=CriticRole.HUMAN, round=1, thesis="", arguments=args,
                          parsed_ok=True, spoke_at=_now())

    rebs = []
    for r in replies:
        target = _target_from_arg_id(r.target_arg_id or "", valid_arg_ids)
        if target is None:      # submit() đã chặn -> tới đây coi như không xảy ra
            logger.warning("human_r2: bỏ phản biện, target_arg_id=%r lạ", r.target_arg_id)
            continue
        rebs.append(Rebuttal(
            target_critic=target, target_arg_id=normalize_arg_id(r.target_arg_id),
            stance=r.stance, reason=r.message,
        ))
    return CriticTurn(critic=CriticRole.HUMAN, round=2, bulletin_seen=True, thesis="",
                      rebuttals=rebs, parsed_ok=True, spoke_at=_now())


def make_human_node(round_no: int):
    """Node PAUSE của người học cho 1 vòng.

    Opt-in tắt -> return {} NGAY: node thành no-op thuần. Nhờ vậy KHÔNG cần conditional
    edge hay dựng 2 biến thể subgraph — luồng cũ chạy y nguyên xuyên qua node này, và
    mọi test cũ (assert round1 đúng 4 key) vẫn xanh.
    """
    async def _node(state: DebateSubState) -> dict:
        if not state.get("human_in_debate"):
            return {}
        thread_id = state.get("thread_id") or ""
        if not thread_id:
            # Không có khoá phiên thì người học không gửi tin tới đâu được -> đừng pause
            logger.warning("human_r%d: thiếu thread_id -> bỏ lượt người học.", round_no)
            return {}

        # Vòng 1 chưa có gì để nhắm tới; vòng 2 chỉ được nhắm id CÓ THẬT trong bảng tin.
        valid = {aid for e in state.get("bulletin", []) for aid in e.arg_ids}
        writer = safe_stream_writer()
        replies = await _collect_human(thread_id, round_no, valid, writer)
        if not replies:
            return {}                       # bỏ qua / hết giờ -> hội đồng đi tiếp

        # KHÔNG bắn live ở đây: _collect_human đã bắn từng tin lúc nhận rồi, bắn thêm bản
        # gộp nữa là FE vẽ lại y hệt lần hai. Bản gộp chỉ dùng để dựng STATE (bulletin,
        # bài luận) và để critics_debate() ghi milestone persist — cả hai đều không live.
        turn = _human_turn(round_no, replies, valid)
        return {f"round{round_no}": {CriticRole.HUMAN: turn}}
    return _node


def bulletin_node(state: DebateSubState) -> dict:
    """Barrier sau R1: parse các lượt R1 (4 critic + người học nếu có) thành Bulletin cho R2."""
    r1 = state.get("round1", {})
    bulletin = []
    for role in CRITIC_ORDER + [CriticRole.HUMAN]:
        turn = r1.get(role)
        if not turn:
            # Guard sẵn có này nuốt luôn cả ca người học vắng mặt (bỏ qua / hết giờ / không
            # bật opt-in) -> không cần nhánh riêng nào cho "không có human".
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
    """Barrier sau R2: gom điểm đồng thuận / tranh cãi từ stance của các rebuttal.

    Tính cả phản biện của NGƯỜI HỌC: stance của họ là stance thật (họ tự chọn), nên
    đóng góp của họ vào consensus/contested có giá trị y như của critic.
    """
    r2 = state.get("round2", {})
    consensus, contested = [], []
    for role in CRITIC_ORDER + [CriticRole.HUMAN]:
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
    g.add_node("human_r1", make_human_node(1))
    g.add_node("bulletin", bulletin_node)
    for role in CRITIC_ORDER:
        g.add_node(f"{role.value}_r2", make_r2_node(role, llm))
    g.add_node("human_r2", make_human_node(2))
    g.add_node("collect", collect_node)

    # human_r1 thay bulletin làm BARRIER của vòng 1: người học phải đọc xong ý kiến của cả
    # 4 critic rồi mới phát biểu, và phát biểu đó phải kịp vào bulletin để vòng 2 bắt bẻ.
    # Opt-in tắt -> human_r1/human_r2 là no-op -> chuỗi hệt như cũ, chỉ thêm 1 nhịp rỗng.
    for role in CRITIC_ORDER: # fan-in
        g.add_edge(START, f"{role.value}_r1")       # 4 R1 chạy song song
        g.add_edge(f"{role.value}_r1", "human_r1")  # 4 node đều trỏ vào -> rào chắn chờ đủ 4
    g.add_edge("human_r1", "bulletin")
    for role in CRITIC_ORDER:
        g.add_edge("bulletin", f"{role.value}_r2")  # 4 R2 chạy song song
        g.add_edge(f"{role.value}_r2", "human_r2")  # barrier chờ đủ 4 rồi mới tới lượt người học
    g.add_edge("human_r2", "collect")
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
    thread_id = state.get("thread_id") or ""
    # check có đăng ký phản biện trong thread này? và xóa để phục vụ chat sau
    human_in_debate = debate_session.has_optin(thread_id) if thread_id else False
    logger.info("critics_debate vào node: thread=%s human_in_debate=%s retry_feedback=%s",
                thread_id or "(rỗng)", human_in_debate, bool(judge_feedback))
    if human_in_debate:
        # Debate bắt đầu -> khoá nút "Tranh luận cùng AI" trên FE (bấm thêm cũng vô nghĩa:
        # cờ vừa bị lấy đi rồi).
        writer = safe_stream_writer()
        if writer is not None:
            writer(StreamEvent(
                seq=state.get("event_seq", 0), type=EventType.DEBATE_LOCK,
                node="debate:lock", content="Hội đồng bắt đầu tranh luận — bạn sẽ được mời phát biểu.",
                payload={"locked": True, "ui": ui_meta(EventType.DEBATE_LOCK)}, ts=_now(),
            ).model_dump(mode="json"))

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
        "thread_id": thread_id,
        "human_in_debate": human_in_debate,
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
    for role in CRITIC_ORDER + [CriticRole.HUMAN]:
        turn = r1.get(role)
        if turn:
            emitter.critic_turn(f"critic:{role.value}:r1", role,
                                _turn_content(1, role, turn),
                                actor=CRITIC_DISPLAY[role], payload=_turn_payload(1, turn))
    emitter.bulletin(
        "debate:bulletin", "Bảng tin chung đã sẵn sàng.",
        payload={"entries": [{"critic": e.critic.value, "thesis": e.thesis,
                              "key_points": e.key_points} for e in debate.bulletin]},
    )
    for role in CRITIC_ORDER + [CriticRole.HUMAN]:
        turn = r2.get(role)
        if turn:
            emitter.critic_turn(f"critic:{role.value}:r2", role,
                                _turn_content(2, role, turn),
                                actor=CRITIC_DISPLAY[role], payload=_turn_payload(2, turn))
    delta = {
        "debate": debate,
        "current_stage": Stage.CRITICS_DEBATE,
        "current_node": "critics_debate",
        "events": emitter.milestones,
        "event_seq": emitter.seq,
    }
    return delta
