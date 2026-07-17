import asyncio
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from state.agent_state import init_state
from state.state_schema import Route, Stage
from graph.workflow import build_graph, _route_from_state
from agents.supervisor import _Decision


def test_route_from_state():
    assert _route_from_state({"route": Route.DEEP}) == "prepare_context"
    assert _route_from_state({"route": Route.FACTUAL}) == "factual"
    assert _route_from_state({}) == "factual"  # mặc định khi chưa có route


class MockRAGService:
    def __init__(self):
        self.last_query_kwargs = {}
    def hybrid_search(self, query, filters=None, limit=5, k=60):
        return [{"chunk_id": "c1", "text": "Mock content", "metadata": {"ten_tac_pham": "mock"}}]
    def query(self, query, filters=None, limit=5, model_name=None, retrieve=True, history=None, refine_instruction=None):
        self.last_query_kwargs = {"query": query, "retrieve": retrieve, "refine_instruction": refine_instruction}
        return {"answer": "Mock factual answer", "sources": []}


def _run_with_decision(decision: _Decision, rag: "MockRAGService | None" = None) -> dict:
    """Ép supervisor ra 1 quyết định cố định rồi chạy graph (không cần Gemini key).

    ainvoke: graph có node async (prepare_context / debate / write_essay) nên chạy
    bằng invoke() sync sẽ vỡ.
    """
    with patch("agents.supervisor._classify", return_value=decision):
        app = build_graph(rag_service=rag or MockRAGService())  # Inject mock RAG
        return asyncio.run(app.ainvoke(init_state("câu hỏi test", "t1", "r1")))


def _run_with_route(route: Route) -> dict:
    return _run_with_decision(_Decision(route=route, reasoning="test"))


def test_graph_routes_to_factual():
    out = _run_with_route(Route.FACTUAL)
    assert out["route"] == Route.FACTUAL
    assert out.get("factual") is not None
    assert out.get("essay") is None
    assert out["current_stage"] == Stage.DONE


def test_graph_routes_to_deep():
    out = _run_with_route(Route.DEEP)
    assert out["route"] == Route.DEEP
    assert out.get("essay") is not None
    assert out.get("factual") is None
    assert out["current_stage"] == Stage.DONE


def test_refine_di_qua_mode_a_ke_ca_khi_gemini_chon_deep():
    """User đòi sửa bài lượt trước -> Mode A thi hành, KHÔNG chạy deep pipeline.

    Ép _Decision route=DEEP (Gemini hay chọn nhầm vì "thân bài chưa sâu" nghe hệt yêu cầu
    phân tích sâu) + refine_instruction -> supervisor phải ép về FACTUAL, và mệnh lệnh phải
    tới được rag_service để nó đổi sang REFINE_PROMPT.
    """
    rag = MockRAGService()
    lenh = "Viết lại phần thân bài dài hơn và sâu hơn, giữ nguyên mở bài và kết bài."
    out = _run_with_decision(
        _Decision(route=Route.DEEP, refine_instruction=lenh, reasoning="test"), rag=rag)

    assert out["route"] == Route.FACTUAL           # ép route, không nghe theo DEEP
    assert out.get("essay") is None                # không đụng debate/write_essay
    assert out["intent"].refine_instruction == lenh
    assert rag.last_query_kwargs["refine_instruction"] == lenh
    assert rag.last_query_kwargs["retrieve"] is True   # vẫn retrieve để lấy dẫn chứng


def test_deep_chua_ro_tac_pham_thi_hoi_lai_khong_chay_pipeline():
    """Yêu cầu phân tích sâu nhưng chưa chốt được tác phẩm -> hỏi lại, KHÔNG chạy deep pipeline.

    Đây là ca bug: "Phân tích nhân vật" khi lịch sử có nhiều tác phẩm -> trước đây đoán bừa 1
    tác phẩm rồi chạy cả pipeline ra bài sai (judge accuracy=0). Ép _Decision route=DEEP +
    needs_clarification -> supervisor ép về FACTUAL, factual_node trả thẳng câu hỏi lại (static),
    không retrieve/không gọi model, không đụng debate/essay.
    """
    rag = MockRAGService()
    cauhoi = "Bạn muốn mình phân tích tác phẩm nào ạ? Tấm Cám hay Chiến thắng Mtao Mxay?"
    out = _run_with_decision(
        _Decision(route=Route.DEEP, needs_clarification=True,
                  clarification_question=cauhoi, reasoning="test"), rag=rag)

    assert out["route"] == Route.FACTUAL              # ép route, không nghe theo DEEP
    assert out.get("essay") is None                   # không chạy debate/write_essay
    assert out["intent"].needs_clarification is True
    assert out["final_ai_response"] == cauhoi          # trả thẳng câu hỏi lại supervisor soạn
    assert rag.last_query_kwargs == {}                 # KHÔNG gọi rag.query -> không retrieve/LLM


def test_deep_da_ro_tac_pham_khong_bi_ep_hoi_lai():
    """_Decision deep bình thường (needs_clarification mặc định False) -> supervisor GIỮ route=DEEP.

    Test ở tầng supervisor (không chạy cả pipeline) để không phụ thuộc Ollama: chỉ kiểm tra
    quyết định route không bị cờ hỏi-lại kích hoạt nhầm khi Gemini không bật cờ.
    """
    from agents.supervisor import supervisor
    with patch("agents.supervisor._classify", return_value=_Decision(route=Route.DEEP, reasoning="test")):
        out = supervisor(init_state("phân tích nhân vật Tràng trong Vợ Nhặt", "t1", "r1"))
    assert out["route"] == Route.DEEP
    assert out["intent"].needs_clarification is False


def test_clarification_chi_kich_hoat_khi_route_deep():
    """Cờ needs_clarification chỉ nhận khi Gemini định tuyến DEEP. Nếu Gemini ra FACTUAL mà lỡ
    bật needs_clarification thì supervisor BỎ QUA cờ (đúng phạm vi 'deep flow')."""
    from agents.supervisor import supervisor
    with patch("agents.supervisor._classify", return_value=_Decision(
            route=Route.FACTUAL, needs_clarification=True,
            clarification_question="tác phẩm nào?", reasoning="test")):
        out = supervisor(init_state("Vợ Nhặt của ai?", "t1", "r1"))
    assert out["intent"].needs_clarification is False


def test_cau_hoi_thuong_khong_dinh_refine_instruction():
    """Câu hỏi bình thường -> refine_instruction=None, Mode A dùng SYSTEM_PROMPT như cũ."""
    rag = MockRAGService()
    out = _run_with_decision(_Decision(route=Route.FACTUAL, reasoning="test"), rag=rag)
    assert out["intent"].refine_instruction is None
    assert rag.last_query_kwargs["refine_instruction"] is None


def test_supervisor_fallback_without_api_key(monkeypatch):
    """Thiếu GEMINI_API_KEY -> supervisor fallback route=factual, không crash."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from agents import supervisor as sup
    sup._client = None  # reset cache client
    out = sup.supervisor(init_state("Vợ Nhặt của ai?", "t1", "r1"))
    assert out["route"] == Route.FACTUAL
    assert out["intent"].raw_query == "Vợ Nhặt của ai?"