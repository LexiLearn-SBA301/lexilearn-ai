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


def register_exception_handlers(app: FastAPI) -> None:
    """Đăng ký toàn bộ exception handler lên app. Gọi 1 lần khi khởi tạo FastAPI."""
    app.add_exception_handler(LLMServiceError, _llm_service_error) # -> 503
    #app.add_exception_handler(RagServiceError, _rag_service_error)   # -> 502