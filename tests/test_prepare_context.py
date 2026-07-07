import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.prepare_context import _build_retrieval_query, _ci_match, prepare_context, _parse_llm_response
from state.state_schema import Stage, IntentAnalysis, Route, CriticRole

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
    mock_llm.invoke.return_value.content = '{"summary": "Tóm tắt hay", "entities": [{"name": "Nhân vật A", "type": "character", "description": "desc"}], "themes": ["theme1"]}'

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
        out = prepare_context(state, rag)
        
    assert out["current_stage"] == Stage.PREPARE_CONTEXT
    ctx = out["context"]
    assert len(ctx.chunks) == 2
    assert ctx.summary == "Tóm tắt hay"
    assert len(ctx.entities) == 1
    assert ctx.entities[0].name == "Nhân vật A"
    assert ctx.themes == ["theme1"]
