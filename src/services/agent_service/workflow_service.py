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
from typing import Any, Optional

from graph.workflow import build_graph
from state.agent_state import AgentState, init_state

logger = logging.getLogger("rag-service.services.workflow")


class WorkflowService:
    """Ôm 1 graph compiled, tái dùng cho mọi request."""

    def __init__(self, checkpointer: Optional[Any] = None) -> None:
        # build 1 lần lúc khởi tạo, KHÔNG compile lại mỗi request
        self.app = build_graph(checkpointer)
        logger.info("WorkflowService sẵn sàng (persist=%s).", checkpointer is not None)

    def invoke(self, human_message: str, thread_id: str,
               run_id: Optional[str] = None) -> AgentState:
        """Chạy graph 1 lượt -> trả state cuối.

        thread_id: định danh cuộc hội thoại (để checkpoint/resume khi có Redis).
        run_id: 1 lượt chat; tự sinh nếu không truyền.
        """
        run_id = run_id or uuid.uuid4().hex
        state = init_state(human_message, thread_id=thread_id, run_id=run_id)
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("Invoke workflow thread=%s run=%s", thread_id, run_id)
        return self.app.invoke(state, config=config)