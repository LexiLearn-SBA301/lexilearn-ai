import asyncio
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


def _dedupe_chunks(chunks: list) -> list:
    """Bỏ chunk trùng/nằm gọn trong chunk khác (rác do cửa sổ trượt lúc ingest chồng lấn).

    Ca thật (Đăm Săn): 10 chunk lấy về thì 6 chunk là CÙNG một khúc "chày mòn... chuồng lợn...
    chuồng trâu..." lặp lại -> prompt Tool 2/3 phồng gấp đôi và Qwen-3B phải đọc lại 6 lần
    cùng một đoạn. Giữ chunk DÀI trước, rồi bỏ chunk nào là chuỗi con của chunk đã giữ.
    """
    kept: list = []
    for c in sorted(chunks, key=lambda c: len(c.text or ""), reverse=True):
        text = " ".join((c.text or "").split())     # chuẩn hoá khoảng trắng để so chuỗi con
        if not text:
            continue
        if any(text in " ".join((k.text or "").split()) for k in kept):
            continue
        kept.append(c)
    # trả về đúng thứ tự retrieval ban đầu (điểm liên quan giảm dần), không phải thứ tự độ dài
    order = {id(c): i for i, c in enumerate(chunks)}
    kept.sort(key=lambda c: order[id(c)])
    if len(kept) < len(chunks):
        logger.info("prepare_context: dedupe chunk %d -> %d (bỏ đoạn trùng/lồng nhau).",
                    len(chunks), len(kept))
    return kept


def _ci_match(value: str) -> dict:
    """Khớp metadata KHÔNG phân biệt hoa/thường: DB lưu tên tác phẩm CHỮ HOA (vd
    'VIỆT BẮC') còn supervisor tách title-case ('Việt Bắc') -> exact match trượt.
    Anchored ^...$ để vẫn là khớp TRỌN chuỗi (chỉ nới hoa/thường), không thành khớp
    một phần. Mongo $options='i' fold được cả dấu tiếng Việt (Ệ↔ệ, Ắ↔ắ)."""
    return {"$regex": f"^{re.escape(value.strip())}$", "$options": "i"}


async def prepare_context(state: AgentState, rag_service: RAGService) -> dict:
    """Node async: client F5/đóng tab -> CancelledError huỷ được lượt gọi Ollama đang chạy
    (node sync chạy trong thread thì không ngắt được, request vẫn chiếm hàng đợi Ollama).
    Phần retrieve vẫn là I/O blocking (pymongo) -> đẩy sang thread để không chẹn event loop."""
    emitter = EventEmitter(state, writer=safe_stream_writer())
    intent = state.get("intent")
    raw_query = intent.raw_query if intent else state.get("human_message", "")
    # Supervisor quyết định có tra cứu không. False = user đã dán sẵn đoạn thơ/văn
    # ngay trong câu hỏi -> phân tích trực tiếp, tra cứu thêm chỉ gây nhiễu ngữ cảnh.
    need_retrieval = getattr(intent, "need_retrieval", True)

    if not need_retrieval:
        # Dùng thẳng văn bản người dùng cung cấp làm ngữ cảnh (1 chunk). Gắn nhãn
        # tác phẩm/tác giả từ intent để judge relevance khớp câu hỏi, không REJECT.
        query = raw_query
        meta: dict = {}
        if intent and getattr(intent, "work_title", None):
            meta["ten_tac_pham"] = intent.work_title
        if intent and getattr(intent, "author", None):
            meta["tac_gia"] = intent.author
        chunks = [SourceChunk(chunk_id="user-input", text=raw_query, metadata=meta)]
        emitter.status("prepare_context", "Phân tích trực tiếp văn bản bạn cung cấp (không tra cứu thư viện).")
    else:
        # Ưu tiên keyword đã tách (work_title + author + entities); rỗng -> câu đã resolve tham
        # chiếu (retrieval_query) tốt hơn raw_query khi câu hỏi kiểu "phân tích tác phẩm đó".
        resolved = (getattr(intent, "retrieval_query", "") or "").strip() if intent else ""
        query = _build_retrieval_query(intent) or resolved or raw_query

        filters = state.get("filters", {}) or {}
        if intent:
            if intent.work_title:
                filters["ten_tac_pham"] = _ci_match(intent.work_title)   # khớp bất kể hoa/thường
            if intent.author:
                filters["tac_gia"] = _ci_match(intent.author)

        # Retrieve chunks
        raw_chunks = await asyncio.to_thread(
            rag_service.hybrid_search, query, filters=filters, limit=_DEEP_RETRIEVAL_LIMIT)
        # Metadata trong DB có thể BẨN (ten_tac_pham sai nhãn) -> filter exact-match trượt hết.
        # Nếu rỗng mà có filter -> thử lại KHÔNG filter để vector+keyword tự tìm theo nội dung.
        if not raw_chunks and filters:
            logger.info("prepare_context: 0 chunk với filter %s -> thử lại KHÔNG filter.", filters)
            raw_chunks = await asyncio.to_thread(
                rag_service.hybrid_search, query, filters=None, limit=_DEEP_RETRIEVAL_LIMIT)
        chunks = _dedupe_chunks([SourceChunk.model_validate(c) for c in raw_chunks])

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
        response = await llm.ainvoke(messages)
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
