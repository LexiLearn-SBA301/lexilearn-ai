"""
Exception handlers tập trung

Service raise domain exception -> ở đây map sang HTTP response.
register_exception_handlers dựng sẵn bảng lúc startup;
lúc có lỗi, ExceptionMiddleware tra bảng đó để tìm handler → tạo response
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from exceptions import LLMServiceError

logger = logging.getLogger("rag-service.api.errors")


async def _llm_service_error(request: Request, exc: LLMServiceError) -> JSONResponse:
    """LLM (Ollama) không gọi được -> 503 Service Unavailable."""
    return JSONResponse(
        status_code=503,
        content={"detail": "Không gọi được mô hình ngôn ngữ (LLM). Vui lòng thử lại sau."},
    )
async def _global_fallback_error(request: Request, exc: Exception) -> JSONResponse:
    # LƯỚI VÉT: Lỗi không xác định (Bugs, Null, Out of Memory...) -> 500
    # Ghi log chi tiết lỗi để Dev sửa, nhưng chỉ báo user là "Lỗi hệ thống"
    logger.error(f"Lỗi hệ thống chưa được kiểm soát: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Hệ thống đang gặp sự cố bất ngờ."})

def register_exception_handlers(app: FastAPI) -> None:
    """Đăng ký toàn bộ exception handler lên app. Gọi 1 lần khi khởi tạo FastAPI."""
    app.add_exception_handler(LLMServiceError, _llm_service_error) # -> 503
    #app.add_exception_handler(RagServiceError, _rag_service_error)   # -> 502
    app.add_exception_handler(Exception, _global_fallback_error) # -> 500