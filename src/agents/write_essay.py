import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from state.agent_state import AgentState
from state.state_schema import Stage, EssayDraft, EssaySection, CRITIC_DISPLAY, EventEmitter, safe_stream_writer
from agents.essay_schemas import EssayLLMOutput
from config.essay_prompts import ESSAY_SYSTEM_PROMPT, ESSAY_USER_PROMPT_TEMPLATE
from providers.ollama_provider import ollama_provider, FINE_TUNED_OLLAMA_LLM_MODEL

logger = logging.getLogger("rag-service.agents.write_essay")


def _render_debate_data(debate_state: Any) -> str:
    """Format kết quả debate thành chuỗi dễ đọc cho prompt.

    KHÔNG in consensus_points/contested_points: collect_node dựng 2 list đó TỪ CHÍNH `reason`
    của các rebuttal, nên in chúng rồi in tiếp "CHI TIẾT PHẢN BIỆN CHÉO" là lặp nguyên văn
    cùng một nội dung 2 lần -> prompt phồng gấp đôi mà không thêm thông tin nào cho Qwen-3B.
    """
    if not debate_state:
        return "(Không có dữ liệu tranh luận)"

    parts = []

    # Vòng 1: Luận đề & Luận điểm
    parts.append("### CHI TIẾT LUẬN ĐIỂM TỪ CÁC GÓC NHÌN:")
    for role, turn in debate_state.round1.items():
        role_name = CRITIC_DISPLAY.get(role, role.value)
        parts.append(f"**{role_name}**")
        # Bỏ nhãn khi rỗng thay vì in "Luận đề: " / "+ điểm: " cụt đuôi. Lượt của NGƯỜI HỌC
        # không có luận đề lẫn lý lẽ (họ gõ thẳng luận điểm, không qua LLM) -> in nhãn rỗng
        # thì Qwen-3B tưởng dữ liệu hỏng hoặc tưởng họ không có quan điểm.
        if turn.thesis:
            parts.append(f"Luận đề: {turn.thesis}")
        for arg in turn.arguments:
            parts.append(f"  + {arg.point}: {arg.support}" if arg.support else f"  + {arg.point}")
        parts.append("")
        
    # Vòng 2: Phản biện
    parts.append("### CHI TIẾT PHẢN BIỆN CHÉO:")
    has_rebuttals = False
    for role, turn in debate_state.round2.items():
        if not turn.rebuttals:
            continue
        has_rebuttals = True
        role_name = CRITIC_DISPLAY.get(role, role.value)
        parts.append(f"**{role_name}** phản biện:")
        for reb in turn.rebuttals:
            target_name = CRITIC_DISPLAY.get(reb.target_critic, reb.target_critic.value)
            parts.append(f"  + Nhắm tới {target_name} ({reb.stance}): {reb.reason}")
    
    if not has_rebuttals:
        parts.append("(Không có phản biện đáng kể)")

    return "\n".join(parts)


def _render_nguyen_lieu(context: Any) -> str:
    """Văn bản gốc, đánh số (1) (2) (3) — ĐÚNG format [Ngữ liệu] model được fine-tune (s4.jsonl).

    Trước đây node này chỉ truyền summary + entities, chunks KHÔNG BAO GIỜ tới tay model.
    Mà fine-tune dạy nó "mọi dẫn chứng phải trích nguyên văn từ ngữ liệu" -> không có câu chữ
    nào để trích, nó buộc phải bịa dẫn chứng (citation_check fail).
    """
    chunks = getattr(context, "chunks", None) if context else None
    if not chunks:
        return "(không có ngữ liệu)"
    return "\n\n".join(f"({i}) {(c.text or '').strip()}" for i, c in enumerate(chunks, 1))


def _stream_prose(emitter: EventEmitter, node: str, text: str) -> None:
    """Cách 2 — chảy chữ "giả": tua lại prose ĐÃ sinh xong ra kênh token (is_partial,
    KHÔNG persist) để FE có hiệu ứng gõ chữ. KHÔNG đụng bước sinh structured của Tool 3.
    Text vào đây đã sạch (chỉ heading + body, không có `thinking`) nên đẩy thẳng an toàn;
    answer chuẩn vẫn nằm ở done.payload.answer (finalize) nên token chỉ để hiển thị."""
    if not text:
        return
    for word in text.split(" "):
        emitter.token(node, word + " ")


async def write_essay(state: AgentState) -> dict:
    """Node public (Tool 3). Đọc debate + context, gọi LLM sinh bài luận.

    async: sinh bài luận là lượt gọi Ollama dài nhất -> phải huỷ được khi client ngắt (F5),
    nếu không nó vẫn sinh nốt và chiếm hàng đợi Ollama của lượt chat kế tiếp.
    """
    emitter = EventEmitter(state, writer=safe_stream_writer())
    query = state.get("human_message", "")
    context = state.get("context")
    debate = state.get("debate")
    
    # Ráp template
    nguyen_lieu = _render_nguyen_lieu(context)
    debate_str = _render_debate_data(debate)

    # Nếu có feedback từ judge (với Tool 3, có thể tương lai thêm judge, cấu trúc sẵn)
    fb = (state.get("last_feedback") or {}).get(Stage.WRITE_ESSAY.value, "")
    fb_block = f"\n\n--- GÓP Ý TỪ GIÁM KHẢO CHO LƯỢT TRƯỚC ---\n{fb}\nHãy sửa lại bài viết theo góp ý trên, giữ nguyên những điểm đã tốt." if fb else ""

    user_prompt = ESSAY_USER_PROMPT_TEMPLATE.format(
        nguyen_lieu=nguyen_lieu,
        query=query,
        debate_data=debate_str,
        feedback_block=fb_block
    )

    msgs = [
        SystemMessage(content=ESSAY_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    llm = ollama_provider.get_llm(model=FINE_TUNED_OLLAMA_LLM_MODEL, temperature=0.3)
    structured_llm = llm.with_structured_output(EssayLLMOutput)
    
    logger.info("Tool3 write_essay bắt đầu sinh bài (sẽ tốn thời gian vì sinh text dài)...")
    emitter.status("write_essay", "Đang viết bài luận…")
    try:
        # Gọi mô hình để lấy structured output
        out: EssayLLMOutput = await structured_llm.ainvoke(msgs)
        
        # Chuyển đổi EssayLLMOutput -> EssayDraft (bỏ thinking)
        sections = []
        full_text_parts = []
        word_count = 0
        
        for sec in out.sections:
            sections.append(EssaySection(heading=sec.heading, body=sec.body))
            full_text_parts.append(f"## {sec.heading}\n{sec.body}")
            word_count += len(sec.body.split())
            
        full_text = "\n\n".join(full_text_parts)
        
        essay = EssayDraft(
            title=out.title,
            sections=sections,
            full_text=full_text,
            word_count=word_count,
        )
        logger.info("Tool3 write_essay xong: title='%s', words=%d", essay.title, essay.word_count)
        # Cách 2: tua prose sạch ra kênh token cho FE gõ dần, rồi chốt milestone hoàn thành.
        _stream_prose(emitter, "write_essay", full_text)
        emitter.status(
            "write_essay",
            f"Đã hoàn thành bản thảo: {essay.title} ({essay.word_count} từ)",
            payload={"title": essay.title, "word_count": essay.word_count},
        )

    except Exception as e:
        logger.error("Tool3 write_essay structured lỗi (%s) -> trả fallback", e)
        # Fallback nếu model parse lỗi
        essay = EssayDraft(
            title="Lỗi sinh bài luận",
            full_text=f"[LỖI TOOL 3] Không thể sinh bài luận từ dữ liệu tranh luận. Lỗi: {e}",
            word_count=0
        )
        emitter.status("write_essay", "Lỗi sinh bài luận, dùng bản dự phòng.")

    return {
        "essay": essay,
        "current_stage": Stage.WRITE_ESSAY,
        "current_node": "write_essay",
        "events": emitter.milestones,
        "event_seq": emitter.seq,
    }
