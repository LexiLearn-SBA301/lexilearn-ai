"""
Dependencies (DI providers) cho tầng API — tương đương @Configuration của Spring.

Gom các hàm provider dùng với Depends(...) tại đây để router chỉ còn khai báo
endpoint. Khi scale thêm service (RAG, retriever, ...) chỉ cần thêm provider mới
vào file này, không phải sửa router.
"""
from fastapi import Request

from services.agent_service.chat_service import OllamaChatService
from services.agent_service.workflow_service import WorkflowService


def get_workflow(request: Request) -> WorkflowService:  # trả về instance cho Controller sài
    """Lấy WorkflowService dựng sẵn ở lifespan (Redis-backed).

    KHÔNG init ở module-level để tránh kết nối Redis ngay lúc import
    (sẽ vỡ khi chạy CLI 'python main.py --ingest' hoặc test lúc Redis chưa bật).
    """
    return request.app.state.workflow
def get_chat_svc(request: Request) -> OllamaChatService:
    return request.app.state.chat_svc