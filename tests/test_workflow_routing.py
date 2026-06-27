import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from state.agent_state import init_state
from state.state_schema import Route, Stage
from graph.workflow import build_graph, _route_from_state
from agents.supervisor import _Decision


def test_route_from_state():
    assert _route_from_state({"route": Route.DEEP}) == "deep"
    assert _route_from_state({"route": Route.FACTUAL}) == "factual"
    assert _route_from_state({}) == "factual"  # mặc định khi chưa có route


def _run_with_route(route: Route) -> dict:
    """Ép supervisor ra 1 route cố định rồi chạy graph (không cần Gemini key)."""
    with patch("agents.supervisor._classify",
               return_value=_Decision(route=route, reasoning="test")):
        app = build_graph()  # không checkpointer -> đủ test routing
        return app.invoke(init_state("câu hỏi test", "t1", "r1"))


def test_graph_routes_to_factual():
    out = _run_with_route(Route.FACTUAL)
    assert out["route"] == Route.FACTUAL
    assert out.get("factual") is not None
    assert out.get("essay") is None
    assert out["current_stage"] == Stage.FACTUAL


def test_graph_routes_to_deep():
    out = _run_with_route(Route.DEEP)
    assert out["route"] == Route.DEEP
    assert out.get("essay") is not None
    assert out.get("factual") is None
    assert out["current_stage"] == Stage.DONE


def test_supervisor_fallback_without_api_key(monkeypatch):
    """Thiếu GEMINI_API_KEY -> supervisor fallback route=factual, không crash."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from agents import supervisor as sup
    sup._client = None  # reset cache client
    out = sup.supervisor(init_state("Vợ Nhặt của ai?", "t1", "r1"))
    assert out["route"] == Route.FACTUAL
    assert out["intent"].raw_query == "Vợ Nhặt của ai?"