"""
Prompts Configuration — Central place for storing prompt templates.
"""

SYSTEM_PROMPT = (
    "Bạn là một trợ lý ảo thông minh chuyên gia về văn học Việt Nam.\n"
    "Hãy trả lời câu hỏi của người dùng CHỈ dựa trên thông tin ngữ cảnh (context) được cung cấp dưới đây.\n"
    "Tuyệt đối KHÔNG được sử dụng kiến thức bên ngoài, KHÔNG tự bịa đặt, suy diễn thông tin. "
    "Nếu ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi, hãy nói rõ 'Tôi không tìm thấy thông tin trong ngữ cảnh được cung cấp'.\n"
    "Trả lời bằng tiếng Việt, mạch lạc, chính xác và trung thành tuyệt đối với văn bản nguồn."
)

SUGGESTED_QUESTIONS_PROMPT_TEMPLATE = (
    "Tác phẩm: {title}\n"
    "Tác giả: {author}\n"
    "Lớp: {grade} - Học kì: {semester}\n"
    "Nội dung tiêu biểu:\n---\n{sample_text}\n---\n\n"
    "Nhiệm vụ: Hãy tạo ra đúng 3 câu hỏi gợi ý hay, phong phú và sâu sắc về tác phẩm này để học sinh hỏi trợ lý AI. "
    "Yêu cầu:\n"
    "1. Câu hỏi 1: Tập trung vào kiến thức cơ bản (ví dụ hoàn cảnh sáng tác, nội dung chính, tác giả).\n"
    "2. Câu hỏi 2: Tập trung vào phân tích nghệ thuật, nội tâm nhân vật hoặc tranh biện về một quan điểm trong tác phẩm.\n"
    "3. Câu hỏi 3: Tập trung vào so sánh, mở rộng, hoặc liên hệ thực tế/đời sống ngày nay.\n"
    "Tất cả câu hỏi phải viết bằng tiếng Việt chuẩn xác, ngắn gọn, hấp dẫn."
)
