"""
Domain exceptions toàn app.

Service `raise`, tầng API (exception_handlers) map sang HTTP.
Module này KHÔNG phụ thuộc gì -> mọi tầng import an toàn, không tạo vòng.
"""


class AppException(Exception):
    """Base cho mọi domain exception của app.

    Đăng ký 1 handler cho AppException là bắt được mọi class con (theo MRO).
    """


class LLMServiceError(AppException):
    """Gọi LLM (Ollama) thất bại — tầng API map thành HTTP 503."""