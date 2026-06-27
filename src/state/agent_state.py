"""
AgentState — bản HỢP NHẤT của:
  (A) GraphState  (state_schema.py)  -> lifecycle, observability, retry trần, output object
  (B) AgentState  (template của bạn) -> messages(add_messages), HITL, tool-tracking

Nguyên tắc gộp:
  - Field chỉ 1 bên có  -> giữ.
  - Field 2 bên trùng nghĩa -> gộp làm 1, chọn bản giàu hơn (ghi chú "<- ... | ...").
  - Mọi field nhiều node ghi song song -> khai báo reducer (merge_dict / operator.add / take_last).

Quan hệ cần nhớ:
  - messages  = lịch sử hội thoại ĐƯA VÀO LLM   (add_messages)
  - events    = luồng thinking HIỂN THỊ CHO USER (milestone; token đi kênh riêng)
  -> hai thứ KHÁC nhau, cùng tồn tại.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional

from typing_extensions import TypedDict  # Pydantic <3.12 yêu cầu (Docker chạy Python 3.11)

from langgraph.graph.message import add_messages

# Tái dùng toàn bộ sub-model / enum / reducer đã định nghĩa
from state.state_schema import (
    Route,
    Stage,
    IntentAnalysis,
    FactualResult,
    PreparedContext,
    DebateState,
    EssayDraft,
    JudgeVerdict,
    FinalOutput,
    StreamEvent,
    merge_dict,
    take_last,
    DEFAULT_RETRY_LIMITS,
)


class AgentState(TypedDict, total=False):
    # total=False: không bắt buộc tất cả field phải đầy đủ
    # TypedDict: Cấu trúc dữ liệu rõ ràng
    # Annotated[x,y]  bật chế độ reducer y cho field x, để khi lưu state vào Redis hoặc DB, các field này sẽ được merge theo cách định nghĩa.
    # có thể là ghi đè hoặc nối tiếp vào list (operator.add) tùy vào reducer y
    # ===== 1. Định danh & vòng đời  (từ GraphState) =====
    thread_id: str                                      # định danh 1 cuộc trò chuyện
    run_id: str                                         # 1 câu chat trong cuộc hội thoại
    status: Annotated[
        Literal["running", "awaiting_human", "awaiting_retry", "done", "failed"],
        take_last]  # trạng thái xử lý của câu chat giao tiếp giữa BE và UI để biết AI xử lý tới đâu cần gì
    current_stage: Annotated[Stage, take_last]          # đang ở state nào để Supervisor biết
    current_node: Annotated[Optional[str], take_last]   # dựa vào stage -> biết node nào -> retry đúng chỗ + gán actor cho stream

    # ===== 2. Input / hội thoại =====
    human_message: str                                  # user_query
    messages: Annotated[list, add_messages]             # template: lịch sử cho LLM (KHÁC events)

    # ===== 3. Routing / intent  (Supervisor) =====
    intent: Annotated[Optional[IntentAnalysis], take_last]
    route: Annotated[Optional[Route], take_last]

    # ===== 4. Kết quả nhánh / tool  (object, không phải float|str) =====
    factual: Annotated[Optional[FactualResult], take_last]   # <- RAG_answer | giàu hơn: giữ citation
    context: Annotated[Optional[PreparedContext], take_last] # <- agent1_result (Tool 1)
    debate:  Annotated[Optional[DebateState], take_last]     # <- agent2_result (Tool 2, 8 critic)
    essay:   Annotated[Optional[EssayDraft], take_last]      # <- agent3_result (Tool 3)

    # ===== 5. Judge / review + điều khiển retry =====
    judges:        Annotated[dict[str, JudgeVerdict], merge_dict]   # keyed theo stage
    retry_counts:  Annotated[dict[str, int], merge_dict]
    retry_limits:  dict[str, int]                                  # GraphState: trần chống loop vô hạn, Static Config không có merge func
    last_feedback: Annotated[dict[str, str], merge_dict]           # judge -> tool khi retry (retry có hướng)

    # ===== 6. Tool tracking  (từ template) =====
    last_tool_called: Annotated[Optional[str], take_last]
    tool_input:       Annotated[Optional[dict], take_last]
    tool_result:      Annotated[Optional[Any], take_last]   # debug/audit; kết quả "đẹp" nằm ở context/debate/essay

    # ===== 7. Human-in-the-loop  (từ template; để None nếu supervisor tự duyệt hết) =====
    #awaiting_human: Annotated[Optional[bool], take_last]
    #human_decision: Annotated[Optional[str], take_last]

    # ===== 8. Output  (đáp ứng cả 2 bên) =====
    final_ai_response: Annotated[str, take_last]               # template: text thuần, FE render nhanh
    final_output: Annotated[Optional[FinalOutput], take_last]  # GraphState: object đầy đủ citation + sources

    # ===== 9. Thinking realtime ra UI  (KHÁC messages) =====
    events: Annotated[list[StreamEvent], operator.add]         # chỉ milestone; token đi kênh stream riêng
    event_seq: Annotated[int, take_last]


def init_state(human_message: str, thread_id: str, run_id: str) -> AgentState:
    """State khởi đầu cho 1 lần chạy."""
    return AgentState(
        thread_id=thread_id,
        run_id=run_id,
        status="running",
        current_stage=Stage.INTENT,
        current_node="supervisor:intent",
        human_message=human_message,
        messages=[{"role": "user", "content": human_message}],
        intent=None,
        route=None,
        factual=None,
        context=None,
        debate=None,
        essay=None,
        judges={},
        retry_counts={},
        retry_limits=dict(DEFAULT_RETRY_LIMITS),   # {"prepare_context":2,"critics_debate":2,"write_essay":1}
        last_feedback={},
        last_tool_called=None,
        tool_input=None,
        tool_result=None,
        #awaiting_human=False,
        #human_decision=None,
        final_ai_response="",
        final_output=None,
        events=[],
        event_seq=0,
    )
