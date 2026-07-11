import logging
import json
import re
import os
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage

from state.agent_state import AgentState
from state.state_schema import (
    Entity,
    EventEmitter,
    PreparedContext,
    SourceChunk,
    Stage,
    safe_stream_writer,
)
from config.prepare_context_prompt import PREPARE_CONTEXT_SYSTEM, PREPARE_CONTEXT_USER
from providers.ollama_provider import ollama_provider
from services.rag_service import RAGService

logger = logging.getLogger("rag-service.agents.prepare_context")

_DEEP_RETRIEVAL_LIMIT = 10
CONTEXT_PARSER_MODEL = os.getenv("CONTEXT_PARSER_MODEL", os.getenv("FINE_TUNED_OLLAMA_LLM_MODEL", "qwen2.5:3b"))

def _parse_llm_response(text: str) -> dict:
    """Fallback 3-tier parsing for JSON output."""
    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Attempt 2: regex extract JSON block
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
        
    # Attempt 3: Return empty default
    logger.warning("Failed to parse LLM output as JSON. Falling back to empty summary.")
    return {"summary": "", "entities": [], "themes": []}


def _build_retrieval_query(intent) -> str:
    """Dựng query retrieve GỌN từ keyword supervisor (Gemini) đã tách sẵn thay vì nhồi
    nguyên câu hỏi (giảm nhiễu embedding -> trúng hơn). Chỉ lấy thực thể CỤ THỂ
    (work_title/author/entities), KHÔNG lấy requested_dimensions vì đó là khái niệm
    trừu tượng (vd 'tâm lý') mà DB văn bản thô không truy hồi theo được.
    Trả "" nếu supervisor chưa tách được gì -> caller fallback về câu hỏi gốc.
    """
    if intent is None:
        return ""
    parts = []
    if getattr(intent, "work_title", None):
        parts.append(intent.work_title)
    if getattr(intent, "author", None):
        parts.append(intent.author)
    parts.extend(getattr(intent, "detected_entities", None) or [])
    seen, out = set(), []
    for p in parts:
        p = (p or "").strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return " ".join(out)


def _ci_match(value: str) -> dict:
    """Khớp metadata KHÔNG phân biệt hoa/thường: DB lưu tên tác phẩm CHỮ HOA (vd
    'VIỆT BẮC') còn supervisor tách title-case ('Việt Bắc') -> exact match trượt.
    Anchored ^...$ để vẫn là khớp TRỌN chuỗi (chỉ nới hoa/thường), không thành khớp
    một phần. Mongo $options='i' fold được cả dấu tiếng Việt (Ệ↔ệ, Ắ↔ắ)."""
    return {"$regex": f"^{re.escape(value.strip())}$", "$options": "i"}


def prepare_context(state: AgentState, rag_service: RAGService) -> dict:
    emitter = EventEmitter(state, writer=safe_stream_writer())
    intent = state.get("intent")
    raw_query = intent.raw_query if intent else state.get("human_message", "")
    # Ưu tiên keyword đã tách (work_title + author + entities); rỗng -> fallback câu hỏi gốc.
    query = _build_retrieval_query(intent) or raw_query

    filters = state.get("filters", {}) or {}
    if intent:
        if intent.work_title:
            filters["ten_tac_pham"] = _ci_match(intent.work_title)   # khớp bất kể hoa/thường
        if intent.author:
            filters["tac_gia"] = _ci_match(intent.author)

    # Retrieve chunks
    raw_chunks = rag_service.hybrid_search(query, filters=filters, limit=_DEEP_RETRIEVAL_LIMIT)
    # Metadata trong DB có thể BẨN (ten_tac_pham sai nhãn) -> filter exact-match trượt hết.
    # Nếu rỗng mà có filter -> thử lại KHÔNG filter để vector+keyword tự tìm theo nội dung.
    if not raw_chunks and filters:
        logger.info("prepare_context: 0 chunk với filter %s -> thử lại KHÔNG filter.", filters)
        raw_chunks = rag_service.hybrid_search(query, filters=None, limit=_DEEP_RETRIEVAL_LIMIT)
    chunks = [SourceChunk.model_validate(c) for c in raw_chunks]

    works = sorted({c.metadata.get("ten_tac_pham") for c in chunks if c.metadata.get("ten_tac_pham")})
    emitter.retrieval(
        "prepare_context",
        f"Đã truy hồi {len(chunks)} đoạn trích" + (f" từ: {', '.join(works)}" if works else "."),
        payload={"count": len(chunks), "works": works},
    )

    if not chunks:
        # No context found
        logger.warning("prepare_context found no chunks for query: %s", query)
        return {
            "context": PreparedContext(
                retrieval_query=query,
                chunks=[],
                summary="Không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu.",
                entities=[],
                themes=[],
                retrieved_at=datetime.now(timezone.utc),
            ),
            "current_stage": Stage.PREPARE_CONTEXT,
            "current_node": "prepare_context",
            "events": emitter.milestones,
            "event_seq": emitter.seq,
        }

    # Format chunks text
    chunks_text_parts = []
    for idx, c in enumerate(chunks):
        title = c.metadata.get("ten_tac_pham", "Không rõ")
        author = c.metadata.get("tac_gia", "Không rõ")
        chunks_text_parts.append(f"--- Chunk {idx+1} (Tác phẩm: {title}, Tác giả: {author}) ---\n{c.text}")
    chunks_text = "\n\n".join(chunks_text_parts)
    
    # Run Ollama
    system_prompt = PREPARE_CONTEXT_SYSTEM
    user_prompt = PREPARE_CONTEXT_USER.format(query=query, chunks_text=chunks_text)
    
    llm = ollama_provider.get_llm(model=CONTEXT_PARSER_MODEL)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    emitter.status("prepare_context", "Đang phân tích & tóm tắt ngữ cảnh…")
    response_text = ""
    try:
        response = llm.invoke(messages)
        response_text = str(response.content) if not isinstance(response.content, str) else response.content
    except Exception as e:
        logger.error("Ollama call failed in prepare_context: %s", e)
        emitter.status("prepare_context", "Không tóm tắt được ngữ cảnh (lỗi mô hình).")
        return {
            "context": PreparedContext(
                retrieval_query=query,
                chunks=chunks,
                summary=f"[LỖI HỆ THỐNG] Không thể gọi mô hình ngôn ngữ (Ollama). Chi tiết: {e}",
                entities=[],
                themes=[],
                retrieved_at=datetime.now(timezone.utc),
            ),
            "current_stage": Stage.PREPARE_CONTEXT,
            "current_node": "prepare_context",
            "events": emitter.milestones,
            "event_seq": emitter.seq,
        }
        
    parsed = _parse_llm_response(response_text)
    
    # Parse entities
    entities_data = parsed.get("entities", [])
    entities = []
    seen_entity_names = set()
    
    for e in entities_data:
        name = str(e.get("name", "")).strip()
        if not name or name.lower() in seen_entity_names:
            continue
        seen_entity_names.add(name.lower())
        entities.append(Entity(
            name=name,
            type=str(e.get("type", "other")),
            description=str(e.get("description", ""))
        ))
        
    themes = [str(t) for t in parsed.get("themes", [])]
    summary = str(parsed.get("summary", ""))

    context_obj = PreparedContext(
        retrieval_query=query,
        chunks=chunks,
        summary=summary,
        entities=entities,
        themes=themes,
        retrieved_at=datetime.now(timezone.utc),
    )
    
    logger.info("prepare_context completed for query: %s", query)
    emitter.status(
        "prepare_context",
        f"Đã dựng ngữ cảnh: {len(entities)} thực thể"
        + (f", chủ đề: {', '.join(themes)}" if themes else ""),
        payload={"themes": themes, "entities": [e.name for e in entities]},
    )
    return {
        "context": context_obj,
        "current_stage": Stage.PREPARE_CONTEXT,
        "current_node": "prepare_context",
        "events": emitter.milestones,
        "event_seq": emitter.seq,
    }
