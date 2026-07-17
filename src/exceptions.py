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


class DebateNotWaiting(AppException):
    """Gửi lời tranh luận nhưng KHÔNG có phiên nào đang chờ — tầng API map thành HTTP 409.

    Ca thật: người học bấm gửi đúng lúc hết giờ chờ / đã bấm Kết thúc / đã đủ 10 tin,
    hoặc gửi nhầm thread. Không phải lỗi hệ thống -> đừng để rơi vào lưới vét 500.
    """


class DebateInvalidReply(AppException):
    """Lời tranh luận sai định dạng (thiếu nội dung, stance lạ, target_arg_id không có
    thật trong bảng tin) — tầng API map thành HTTP 400.

    Khác hẳn đường LLM: id do model bịa thì bị BỎ IM LẶNG (xem _speak_r2), còn id do FE
    gửi thì báo lỗi thẳng — FE chọn từ danh sách có sẵn nên sai = bug, phải thấy ngay.
    """