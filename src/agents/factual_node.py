import os
import logging
from typing import Optional

from state.agent_state import AgentState
from state.state_schema import EventEmitter, FactualResult, Stage, SourceChunk, safe_stream_writer
from services.rag_service import RAGService
from providers.ollama_provider import FINE_TUNED_OLLAMA_LLM_MODEL
from config.prompt_template import OFF_TOPIC_REPLY

logger = logging.getLogger("rag-service.agents.factual_node")


def factual_node(state: AgentState, rag_service: Optional[RAGService] = None) -> dict:
    """Node xử lý câu hỏi Factual (Mode A), gọi RAGService để trả lời."""
    emitter = EventEmitter(state, writer=safe_stream_writer())
    intent = state.get("intent")
    human_message = state.get("human_message", "")
    # Mode A: retrieve bằng câu đã resolve tham chiếu (supervisor); rỗng -> câu gốc.
    query = (getattr(intent, "retrieval_query", "") or "").strip() or human_message
    # Lịch sử = messages (các lượt đã xong, ≤10 do finalize giữ window) -> qwen trả lời có mạch.
    history = state.get("messages", []) or []
    # Câu ngoài văn học (toán/code/khoa học...) -> trả lời từ chối CỐ ĐỊNH, không gọi
    # qwen (3B hay lộ đáp án ngoài lề) và không tra cứu.
    if not getattr(intent, "on_topic", True):
        logger.info("Factual node: off-topic -> trả lời cố định, bỏ qua retrieve/LLM.")
        return {
            "factual": FactualResult(answer=OFF_TOPIC_REPLY, chunks_used=[], model="static"),
            "current_stage": Stage.FACTUAL,
            "current_node": "factual",
            "events": emitter.milestones,
            "event_seq": emitter.seq,
        }
    # Yêu cầu phân tích sâu nhưng supervisor chưa chốt được tác phẩm -> hỏi lại tác phẩm nào,
    # KHÔNG retrieve/không gọi model. Đoán bừa 1 tác phẩm rồi chạy cả pipeline chỉ ra bài sai
    # (judge chấm accuracy=0 ở cuối). Câu hỏi lại do supervisor (Gemini) soạn sẵn.
    if getattr(intent, "needs_clarification", False):
        question = getattr(intent, "clarification_question", "") or "Bạn muốn mình phân tích tác phẩm nào ạ?"
        logger.info("Factual node: cần hỏi lại tác phẩm -> trả câu hỏi lại, bỏ qua retrieve/LLM.")
        return {
            "factual": FactualResult(answer=question, chunks_used=[], model="static"),
            "current_stage": Stage.FACTUAL,
            "current_node": "factual",
            "events": emitter.milestones,
            "event_seq": emitter.seq,
        }
    # Supervisor quyết định có tra cứu không (chào hỏi/tán gẫu -> khỏi retrieve).
    need_retrieval = getattr(intent, "need_retrieval", True)
    # != None -> supervisor giao lệnh SỬA bài văn lượt trước: vẫn retrieve (cần ngữ liệu để lấy
    # dẫn chứng) nhưng rag_service đổi sang REFINE_PROMPT; bài cũ model đọc từ `history`.
    refine_instruction = getattr(intent, "refine_instruction", None)

    if rag_service:
        logger.info(f"Factual node querying RAGService for: {query} (retrieve={need_retrieval}, refine={bool(refine_instruction)})")
        if need_retrieval:
            emitter.status(
                "factual",
                "Đang tra cứu lại tác phẩm để sửa bài…" if refine_instruction else "Đang tra cứu tài liệu…",
            )

        filters = state.get("filters", {})
        # Dùng model fine-tune như yêu cầu của user
        result = rag_service.query(query=query, filters=filters, model_name=FINE_TUNED_OLLAMA_LLM_MODEL, retrieve=need_retrieval, history=history, refine_instruction=refine_instruction)

        answer = result.get("answer", "Không có câu trả lời.")
        raw_sources = result.get("sources", [])

        # Chuyển đổi thành SourceChunk
        chunks_used = []
        for src in raw_sources:
            try:
                chunks_used.append(SourceChunk.model_validate(src))
            except Exception as e:
                logger.warning(f"Lỗi parse SourceChunk trong factual_node: {e}")

        model_name = FINE_TUNED_OLLAMA_LLM_MODEL
        if need_retrieval:
            works = sorted({c.metadata.get("work_title") for c in chunks_used if c.metadata.get("work_title")})
            emitter.retrieval(
                "factual",
                f"Đã tra cứu {len(chunks_used)} đoạn trích" + (f" từ: {', '.join(works)}" if works else "."),
                payload={"count": len(chunks_used), "works": works},
            )
    else:
        logger.error("RAGService is None trong factual_node")
        answer = f"[LỖI] Không có RAGService để trả lời cho: {query}"
        chunks_used = []
        model_name = "unknown"
        emitter.status("factual", "Không có RAGService để tra cứu.")

    return {
        "factual": FactualResult(
            answer=answer,
            chunks_used=chunks_used,
            model=model_name,
        ),
        "current_stage": Stage.FACTUAL,
        "current_node": "factual",
        "events": emitter.milestones,
        "event_seq": emitter.seq,
    }
