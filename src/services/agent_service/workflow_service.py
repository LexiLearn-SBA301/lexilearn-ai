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
               run_id: Optional[str] = None) -> AgentState:
        """Chạy graph 1 lượt -> trả state cuối.

        thread_id: định danh cuộc hội thoại (để checkpoint/resume khi có Redis).
        run_id: 1 lượt chat; tự sinh nếu không truyền.
        """
        run_id = run_id or uuid.uuid4().hex
        state = init_state(human_message, thread_id=thread_id, run_id=run_id)
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("Invoke workflow thread=%s run=%s", thread_id, run_id)
        try:
            return await self.app.ainvoke(state, config=config)
        except Exception as e:
            logger.exception("Workflow failed thread=%s run=%s", thread_id, run_id)
            state["status"] = "failed"
            raise

    async def astream(self, human_message: str, thread_id: str,
                      run_id: Optional[str] = None) -> AsyncIterator[dict]:
        """Chạy graph 1 lượt, YIELD từng StreamEvent (dict) realtime.

        Dùng stream_mode=["custom"] + subgraphs=True (BẮT BUỘC subgraphs=True, nếu không
        custom event phát từ trong subgraph debate sẽ bị nuốt — xác nhận ở Phase 0 spike).
        Với cấu hình này mỗi chunk là 3-tuple (namespace, mode, data):
          - namespace == ()            -> event từ graph gốc
          - namespace == ('debate:..') -> event từ subgraph debate
        Node tự đính payload['ui']; ở đây chỉ yield `data` (dict đã model_dump).
        """
        run_id = run_id or uuid.uuid4().hex
        state = init_state(human_message, thread_id=thread_id, run_id=run_id)
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("Astream workflow thread=%s run=%s", thread_id, run_id)
        async for namespace, mode, data in self.app.astream(
            state, config=config, stream_mode=["custom"], subgraphs=True
        ):
            if mode == "custom":
                yield data