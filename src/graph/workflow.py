"""
Workflow — dựng LangGraph StateGraph cho luồng:
    START -> supervisor -> (factual | deep) -> END

- build_graph(checkpointer): wiring + compile. checkpointer inject từ ngoài.
- get_checkpointer(): tạo Redis checkpointer (persist/resume theo thread_id).
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agents.mocks import deep_node, factual_node
from agents.supervisor import supervisor
from state.agent_state import AgentState
from state.state_schema import Route

logger = logging.getLogger("rag-service.graph.workflow")


def _route_from_state(state: AgentState) -> str:
    """Đọc route do supervisor set -> tên node đích cho conditional edge."""
    return "deep" if state.get("route") == Route.DEEP else "factual"


def build_graph(checkpointer=None):
    """Dựng & compile graph. checkpointer=None -> chạy được nhưng không persist."""
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor)
    g.add_node("factual", factual_node)
    g.add_node("deep", deep_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route_from_state,
        {"factual": "factual", "deep": "deep"},
    )
    g.add_edge("factual", END)
    g.add_edge("deep", END)

    return g.compile(checkpointer=checkpointer)