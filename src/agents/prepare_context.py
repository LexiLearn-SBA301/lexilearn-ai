import logging
import json
import re
import os
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, HumanMessage

from state.agent_state import AgentState
from state.state_schema import PreparedContext, SourceChunk, Entity, Stage
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

def prepare_context(state: AgentState, rag_service: RAGService) -> dict:
    intent = state.get("intent")
    query = intent.raw_query if intent else state.get("human_message", "")
    
    filters = {}
    if intent:
        if intent.work_title:
            filters["ten_tac_pham"] = intent.work_title
        if intent.author:
            filters["tac_gia"] = intent.author
            
    # Retrieve chunks
    raw_chunks = rag_service.hybrid_search(query, filters=filters, limit=_DEEP_RETRIEVAL_LIMIT)
    chunks = [SourceChunk.model_validate(c) for c in raw_chunks]
    
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
    
    response_text = ""
    try:
        response = llm.invoke(messages)
        response_text = str(response.content) if not isinstance(response.content, str) else response.content
    except Exception as e:
        logger.error("Ollama call failed in prepare_context: %s", e)
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
    return {
        "context": context_obj,
        "current_stage": Stage.PREPARE_CONTEXT,
        "current_node": "prepare_context",
    }
