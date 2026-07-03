PREPARE_CONTEXT_SYSTEM = """Bạn là trợ lý chuyên phân tích văn học Việt Nam.
Nhiệm vụ: đọc các đoạn trích văn bản được cung cấp, sau đó tóm tắt và trích xuất thông tin."""

PREPARE_CONTEXT_USER = """Dựa vào các đoạn trích văn bản bên dưới, hãy thực hiện:

1. **Tóm tắt** nội dung chính liên quan đến câu hỏi (200–400 từ).
   - Nếu các đoạn trích không chứa đủ thông tin hoặc không liên quan đến câu hỏi,
     hãy ghi rõ trong phần tóm tắt rằng dữ liệu không đủ, và tóm tắt những gì có được.

2. **Trích xuất thực thể** nổi bật: nhân vật, địa điểm, chủ đề, hình ảnh nghệ thuật,
   sự kiện, hoặc bất kỳ yếu tố đáng chú ý nào khác.
   - Mỗi thực thể chỉ liệt kê MỘT LẦN (không trùng lặp).
   - Ghi rõ loại thực thể (ví dụ: nhân vật, địa điểm, chủ đề...) và mô tả ngắn.

3. **Xác định chủ đề / motif** chính được đề cập.

Câu hỏi gốc: {query}

Các đoạn trích:
{chunks_text}

Sau khi phân tích, hãy trả kết quả dưới dạng JSON với cấu trúc:
{{
  "summary": "bản tóm tắt 200–400 từ",
  "entities": [
    {{"name": "Tên thực thể", "type": "loại", "description": "mô tả ngắn"}}
  ],
  "themes": ["chủ đề 1", "chủ đề 2"]
}}
"""
