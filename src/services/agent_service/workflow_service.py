"""
WorkflowService — runner GIỮ graph đã compile (build 1 lần) và chạy nó.

Tách vai trò rõ với graph/workflow.py:
  - workflow.build_graph() : factory, BIẾT cách dựng graph (stateless).
  - WorkflowService        : ÔM self.app + expose invoke() (stateful runner).
checkpointer được inject từ ngoài (None = không persist). Bước Redis sau này
chỉ cần truyền get_checkpointer() vào lúc khởi tạo service ở composition root
(FastAPI lifespan) — KHÔNG phải sửa class này.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator, Optional

from graph.workflow import build_graph
from state.agent_state import AgentState, init_state
from services.rag_service import RAGService

logger = logging.getLogger("rag-service.services.workflow")


class WorkflowService:
    """Ôm 1 graph compiled, tái dùng cho mọi request."""

    def __init__(self, checkpointer: Optional[Any] = None) -> None:
        # build 1 lần lúc khởi tạo, KHÔNG compile lại mỗi request
        self.rag_service = RAGService()
        self.app = build_graph(checkpointer, rag_service=self.rag_service)
        logger.info("WorkflowService sẵn sàng (persist=%s).", checkpointer is not None)

    async def invoke(self, human_message: str, thread_id: str,
               run_id: Optional[str] = None, filters: Optional[dict] = None) -> AgentState:
        """Chạy graph 1 lượt -> trả state cuối.

        thread_id: định danh cuộc hội thoại (để checkpoint/resume khi có Redis).
        run_id: 1 lượt chat; tự sinh nếu không truyền.
        Lịch sử lấy từ checkpoint tích lũy; nạp lại qua seed_history() khi checkpoint nguội.
        """
        run_id = run_id or uuid.uuid4().hex
        state = init_state(human_message, thread_id=thread_id, run_id=run_id, filters=filters)
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("Invoke workflow thread=%s run=%s", thread_id, run_id)
        try:
            return await self.app.ainvoke(state, config=config)
        except Exception as e:
            logger.exception("Workflow failed thread=%s run=%s", thread_id, run_id)
            state["status"] = "failed"
            raise

    async def astream(self, human_message: str, thread_id: str,
                      run_id: Optional[str] = None, filters: Optional[dict] = None) -> AsyncIterator[dict]:
        """Chạy graph 1 lượt, YIELD từng StreamEvent (dict) realtime.

        Dùng stream_mode=["custom"] + subgraphs=True (BẮT BUỘC subgraphs=True, nếu không
        custom event phát từ trong subgraph debate sẽ bị nuốt — xác nhận ở Phase 0 spike).
        Với cấu hình này mỗi chunk là 3-tuple (namespace, mode, data):
          - namespace == ()            -> event từ graph gốc
          - namespace == ('debate:..') -> event từ subgraph debate
        Node tự đính payload['ui']; ở đây chỉ yield `data` (dict đã model_dump).
        """
        run_id = run_id or uuid.uuid4().hex
        state = init_state(human_message, thread_id=thread_id, run_id=run_id, filters=filters)
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("Astream workflow thread=%s run=%s", thread_id, run_id)
        async for namespace, mode, data in self.app.astream(
            state, config=config, stream_mode=["custom"], subgraphs=True
        ):
            if mode == "custom":
                yield data

    async def thread_exists(self, thread_id: str) -> bool:
        """Checkpoint của thread_id còn lịch sử không (BE gọi để quyết seed hay không).

        Thread là all-or-nothing: còn thì có đủ messages, hết TTL thì mất sạch -> chỉ cần
        xem messages có rỗng không."""
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self.app.aget_state(config)
        msgs = (getattr(snapshot, "values", None) or {}).get("messages")
        return bool(msgs)

    async def seed_history(self, thread_id: str, history: Optional[list] = None) -> int:
        """Ghi lịch sử (BE gửi) vào checkpoint mà KHÔNG chạy graph — dùng khi mở lại chat cũ.

        Đè sạch messages cũ (RemoveMessage REMOVE_ALL) rồi nạp history. Cap 10 (5 cặp) để giữ
        đúng window. Trả số message đã seed."""
        from langgraph.graph.message import RemoveMessage, REMOVE_ALL_MESSAGES

        msgs: list = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        for turn in (history or [])[-10:]:            # tối đa 10 message = 5 cặp (đúng window)
            content = (turn.get("content") or "").strip() if isinstance(turn, dict) else ""
            if not content:
                continue
            role = turn.get("role")
            msgs.append({"role": "assistant" if role == "assistant" else "user", "content": content})

        config = {"configurable": {"thread_id": thread_id}}
        # as_node="finalize": gán update như thể node cuối vừa chạy -> next=END -> lượt chat
        # sau chạy MỚI từ START (không resume node dở). Graph nhiều node nên phải chỉ định.
        await self.app.aupdate_state(config, {"messages": msgs}, as_node="finalize")
        logger.info("Seed history thread=%s: %d message.", thread_id, len(msgs) - 1)
        return len(msgs) - 1