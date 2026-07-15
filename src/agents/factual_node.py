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
    query = state.get("human_message", "")
    intent = state.get("intent")
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
    # Supervisor quyết định có tra cứu không (chào hỏi/tán gẫu -> khỏi retrieve).
    need_retrieval = getattr(intent, "need_retrieval", True)

    if rag_service:
        logger.info(f"Factual node querying RAGService for: {query} (retrieve={need_retrieval})")
        if need_retrieval:
            emitter.status("factual", "Đang tra cứu tài liệu…")

        filters = state.get("filters", {})
        # Dùng model fine-tune như yêu cầu của user
        result = rag_service.query(query=query, filters=filters, model_name=FINE_TUNED_OLLAMA_LLM_MODEL, retrieve=need_retrieval)

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
            works = sorted({c.metadata.get("ten_tac_pham") for c in chunks_used if c.metadata.get("ten_tac_pham")})
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
