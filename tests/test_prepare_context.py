import asyncio
import os
import sys
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.prepare_context import (
    _build_retrieval_query, _ci_match, _dedupe_chunks, prepare_context, _parse_llm_response,
)
from state.state_schema import Stage, IntentAnalysis, Route, CriticRole, SourceChunk


def test_dedupe_chunks_bo_doan_chong_lan():
    """Chunk chồng lấn (cửa sổ trượt lúc ingest) -> cùng 1 khúc văn lặp nhiều lần trong prompt.

    Ca thật (Đăm Săn): 10 chunk lấy về thì 6 chunk là cùng khúc 'chày mòn... chuồng lợn...'
    -> prompt Tool 2/3 phồng gấp đôi, Qwen-3B đọc lại 6 lần cùng một đoạn.
    """
    full = "Đăm Săn bừng tỉnh, chớp ngay một cái chày mòn, ném trúng vành tai kẻ địch. Mtao Mxây tháo chạy."
    chunks = [
        SourceChunk(chunk_id="c1", text="Đăm Săn bừng tỉnh, chớp ngay một cái chày mòn"),  # con của c2
        SourceChunk(chunk_id="c2", text=full),
        SourceChunk(chunk_id="c3", text="Nhà Mtao Mxây đầu sàn hiên đẽo hình mặt trăng."),  # đoạn khác
        SourceChunk(chunk_id="c4", text="Mtao Mxây tháo chạy."),                            # con của c2
    ]
    out = _dedupe_chunks(chunks)
    assert [c.chunk_id for c in out] == ["c2", "c3"]   # giữ bản dài + đoạn khác, đúng thứ tự retrieval

class MockRAGService:
    def __init__(self, chunks=None):
        self._chunks = chunks or []

    def hybrid_search(self, query, filters=None, limit=10):
        # Trả về dummy chunks giống dictionary như lúc query database thật
        return self._chunks


def test_build_retrieval_query():
    intent = IntentAnalysis(
        raw_query="Phân tích Thúy Kiều trong Truyện Kiều của Nguyễn Du",
        route=Route.DEEP,

        work_title="Truyện Kiều",
        author="Nguyễn Du",
        detected_entities=["Thúy Kiều"],
        requested_dimensions=[CriticRole.TAM_LY],
        time_period=""
    )
    q = _build_retrieval_query(intent)
    # Hàm này ưu tiên gộp work_title + author + entities để tạo query sạch
    assert "Truyện Kiều" in q
    assert "Nguyễn Du" in q
    assert "Thúy Kiều" in q
    assert "tâm lý" not in q  # Không chứa abstract dimensions


def test_ci_match():
    res = _ci_match("Việt Bắc")
    assert res["$regex"] == r"^Việt\ Bắc$"
    assert res["$options"] == "i"


def test_parse_llm_response():
    # Test fallback parse JSON
    valid_json = '{"summary": "test", "entities": [{"name": "A"}], "themes": ["T"]}'
    parsed1 = _parse_llm_response(valid_json)
    assert parsed1["summary"] == "test"

    dirty_json = '```json\n{"summary": "test2"}\n```'
    parsed2 = _parse_llm_response(dirty_json)
    assert parsed2["summary"] == "test2"

    invalid_text = "Tôi không thể tạo JSON"
    parsed3 = _parse_llm_response(invalid_text)
    assert parsed3["summary"] == ""


def test_prepare_context_success():
    chunks = [
        {"chunk_id": "c1", "text": "Đoạn 1", "metadata": {"ten_tac_pham": "T1"}},
        {"chunk_id": "c2", "text": "Đoạn 2", "metadata": {"ten_tac_pham": "T1"}}
    ]
    rag = MockRAGService(chunks=chunks)
    
    mock_llm = MagicMock()
    # Giả lập response từ Ollama trả về content dạng JSON
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
        content='{"summary": "Tóm tắt hay", "entities": [{"name": "Nhân vật A", "type": "character", "description": "desc"}], "themes": ["theme1"]}'
    ))

    state = {
        "intent": IntentAnalysis(
            raw_query="query",
            route=Route.DEEP,
    
            work_title="T1",
            author="Tác giả",
            detected_entities=[],
            requested_dimensions=[]
        ),
        "human_message": "query"
    }

    with patch("agents.prepare_context.ollama_provider.get_llm", return_value=mock_llm):
        out = asyncio.run(prepare_context(state, rag))
        
    assert out["current_stage"] == Stage.PREPARE_CONTEXT
    ctx = out["context"]
    assert len(ctx.chunks) == 2
    assert ctx.summary == "Tóm tắt hay"
    assert len(ctx.entities) == 1
    assert ctx.entities[0].name == "Nhân vật A"
    assert ctx.themes == ["theme1"]
