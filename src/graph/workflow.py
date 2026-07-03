"""
Workflow — dựng LangGraph StateGraph cho luồng:
    START -> supervisor -> (factual | deep) -> END

- build_graph(checkpointer): wiring + compile. checkpointer inject từ ngoài.
- get_checkpointer(): tạo Redis checkpointer (persist/resume theo thread_id).
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agents.critics_debate import critics_debate
from agents.supervisor_judge import judge_node
from functools import partial
from graph.finalize import finalize
from agents.mocks import deep_node, factual_node
from agents.supervisor import supervisor
from state.agent_state import AgentState
from state.state_schema import Route, Stage, Verdict

logger = logging.getLogger("rag-service.graph.workflow")


def _route_from_state(state: AgentState) -> str:
    """Đọc route do supervisor set -> tên node đích cho conditional edge."""
    return "deep" if state.get("route") == Route.DEEP else "factual"
def _route_supervisor_judge_debate(state: AgentState) -> str:
    """Đọc route do supervisor set -> tên node đích cho conditional edge."""
    verdict = state.get("judges", {}).get(Stage.CRITICS_DEBATE.value)
    return "debate" if verdict is not None and verdict.verdict == Verdict.RETRY else "next"
    # .verdict vì bây gió nó là object chứ không phải dict
def build_graph(checkpointer=None, rag_service=None):
    """Dựng & compile graph. checkpointer=None -> chạy được nhưng không persist."""
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor)
    g.add_node("factual", factual_node)
    g.add_node("deep", partial(deep_node, rag_service=rag_service))
    g.add_node("finalize", finalize)
    g.add_node("debate", critics_debate)
    g.add_node("supervisor_judge_debate", judge_node)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route_from_state,
        {"factual": "factual", "deep": "deep"},
    )
    g.add_edge("factual", "finalize")
    #g.add_edge("deep", "query")
    #g.add_edge("query", "supervisor_judge_query")
    # g.add_conditional_edges(
    #     "supervisor_judge_query",
    #     _route_supervisor_judge_query,
    #     {"query": "query", "next": "debate"},
    # )
    g.add_edge("deep", "debate")   # MOCK TẠM thay Tool 1 -> nối thẳng "deep" -> "debate" khi Tool 1 thật xong
    g.add_edge("debate", "supervisor_judge_debate")
    g.add_conditional_edges(
        "supervisor_judge_debate",
        _route_supervisor_judge_debate,
        {"debate": "debate", "next": "finalize"},
    )
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)