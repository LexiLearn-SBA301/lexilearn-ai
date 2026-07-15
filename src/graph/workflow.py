"""
Workflow — dựng LangGraph StateGraph cho luồng:
    START -> supervisor -> (factual | deep_analysis) -> END

- build_graph(checkpointer): wiring + compile. checkpointer inject từ ngoài.
- get_checkpointer(): tạo Redis checkpointer (persist/resume theo thread_id).
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from functools import partial

from agents.critics_debate import critics_debate
from agents.supervisor_judge import judge_node
from graph.finalize import finalize
from agents.factual_node import factual_node
from agents.supervisor import supervisor
from agents.prepare_context import prepare_context
from agents.write_essay import write_essay
from state.agent_state import AgentState
from state.state_schema import Route, Stage, Verdict

logger = logging.getLogger("rag-service.graph.workflow")


def _route_from_state(state: AgentState) -> str:
    """Đọc route do supervisor set -> tên node đích cho conditional edge."""
    return "prepare_context" if state.get("route") == Route.DEEP else "factual"

def _route_supervisor_judge_debate(state: AgentState) -> str:
    """Đọc verdict từ judge để quyết định cho retry hay đi tiếp."""
    verdict = state.get("judges", {}).get(Stage.CRITICS_DEBATE.value)
    return "debate" if verdict is not None and verdict.verdict == Verdict.RETRY else "write_essay"

def _route_supervisor_judge_context(state: AgentState) -> str:
    """Đọc verdict từ judge để quyết định context đã đủ chi tiết chưa.

    REJECT = hết lượt retry mà context VẪN không dùng được (vd: kho tài liệu không có
    tác phẩm được hỏi) -> đi thẳng finalize. Cho chạy tiếp debate/essay trên context
    sai tác phẩm chỉ tạo ra bài luận bịa.
    """
    verdict = state.get("judges", {}).get(Stage.PREPARE_CONTEXT.value)
    if verdict is None:
        return "debate"
    if verdict.verdict == Verdict.RETRY:
        return "prepare_context"
    if verdict.verdict == Verdict.REJECT:
        return "finalize"
    return "debate"

def _route_supervisor_judge_essay(state: AgentState) -> str:
    """Đọc verdict từ judge để quyết định bài luận đã đạt chưa."""
    verdict = state.get("judges", {}).get(Stage.WRITE_ESSAY.value)
    return "write_essay" if verdict is not None and verdict.verdict == Verdict.RETRY else "finalize"

def build_graph(checkpointer=None, rag_service=None):
    """Dựng & compile graph. checkpointer=None -> chạy được nhưng không persist."""
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor)
    g.add_node("factual", partial(factual_node, rag_service=rag_service))
    
    # 1. Prepare Context (Tool 1) + Judge
    g.add_node("prepare_context", partial(prepare_context, rag_service=rag_service))
    g.add_node("supervisor_judge_context", judge_node)
    
    # 2. Critics Debate (Tool 2) + Judge
    g.add_node("debate", critics_debate)
    g.add_node("supervisor_judge_debate", judge_node)
    
    # 3. Write Essay (Tool 3) + Judge
    g.add_node("write_essay", write_essay)
    g.add_node("supervisor_judge_essay", judge_node)
    
    g.add_node("finalize", finalize)

#=============================================================

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route_from_state,
        {"factual": "factual", "prepare_context": "prepare_context"},
    )
    g.add_edge("factual", "finalize")
    
    # Luồng Deep Analysis:
    # Tool 1 -> Judge context -> (retry | debate)
    g.add_edge("prepare_context", "supervisor_judge_context")
    g.add_conditional_edges(
        "supervisor_judge_context",
        _route_supervisor_judge_context,
        {"prepare_context": "prepare_context", "debate": "debate", "finalize": "finalize"},
    )
    
    # Tool 2 -> Judge debate -> (retry | write_essay)
    g.add_edge("debate", "supervisor_judge_debate")
    g.add_conditional_edges(
        "supervisor_judge_debate",
        _route_supervisor_judge_debate,
        {"debate": "debate", "write_essay": "write_essay"},
    )
    
    # Tool 3 -> Judge essay -> (retry | finalize)
    g.add_edge("write_essay", "supervisor_judge_essay")
    g.add_conditional_edges(
        "supervisor_judge_essay",
        _route_supervisor_judge_essay,
        {"write_essay": "write_essay", "finalize": "finalize"},
    )
    
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)