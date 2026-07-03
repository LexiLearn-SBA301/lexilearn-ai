import os
import json
import logging
import time
from typing import List

# Giả sử PDFReader và ExtractedElement có thể import từ core.pdf_reader
from core.pdf_reader import ExtractedElement

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger("rag-service.gemini-corrector")

class GeminiCorrector:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        if not genai:
            logger.warning("google-genai package is not installed. GeminiCorrector will do nothing.")
            self.client = None
            return
            
        # Kiểm tra API Key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY environment variable not found. GeminiCorrector will do nothing.")
            self.client = None
            return
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_CORRECTOR_MODEL", model_name)
        self.system_prompt = (
            "Bạn là chuyên gia phục hồi văn bản tiếng Việt bị lỗi font (TCVN3, VNI) và lỗi OCR "
            "từ sách giáo khoa Ngữ Văn. Nhiệm vụ:\n"
            "1. Sửa các ký tự bị sai font (ví dụ: ¢ → Â, Õ → Ế, ÷ → Ư)\n"
            "2. Sửa lỗi dính chữ (ví dụ: 'vànhà' → 'và nhà')\n"
            "3. Sửa ký tự rác/nhiễu OCR\n"
            "4. TUYỆT ĐỐI KHÔNG thay đổi nội dung, văn phong, hoặc thêm bớt từ\n"
            "5. TUYỆT ĐỐI KHÔNG sửa tên riêng đặc biệt (Mtao Mxây, Đăm Săn, diêng, v.v.)\n\n"
            "Đầu vào: JSON array [{'id': 0, 'text': '...'}, ...]\n"
            "Đầu ra: JSON array cùng cấu trúc với text đã sửa. Trả về đúng mảng JSON, không bọc trong markdown tick mark nếu không cần thiết."
        )

    def _is_suspicious(self, text: str) -> bool:
        """Kiểm tra nhanh xem text có dấu hiệu bị lỗi font/OCR nặng không."""
        if not text or len(text) < 5:
            return False
            
        # Tìm các ký tự đặc biệt bất thường thường thấy khi lỗi font (không thuộc bảng chữ cái tiếng Việt / ASCII)
        # Các ký tự TCVN3/VNI bị lỗi thường rớt vào các mã ASCII mở rộng hoặc unicode lạ
        suspicious_chars = set("¢Õòè÷µ¸¶·¹¨»¾¼½Æ©ÇË®ÐÎÏÑªÖ×ØÜÞ§£¤¥¦")
        suspicious_count = sum(1 for c in text if c in suspicious_chars)
        
        # Nếu có từ 2 ký tự đáng ngờ trở lên, hoặc tỷ lệ ký tự đáng ngờ > 2% thì gửi đi sửa
        if suspicious_count >= 2 or (suspicious_count / len(text)) > 0.02:
            return True
            
        # Có thể thêm logic detect lỗi dính chữ bằng cách đếm từ quá dài
        words = text.split()
        long_words = sum(1 for w in words if len(w) > 10 and not w.startswith("http"))
        if long_words > max(1, len(words) * 0.1): # Hơn 10% số từ là từ quá dài -> có thể dính chữ
            return True
            
        return False

    def correct(self, elements: List[ExtractedElement], force_all: bool = False) -> List[ExtractedElement]:
        if not self.client or not elements:
            return elements

        logger.info(f"Bắt đầu quy trình kiểm tra và sửa lỗi OCR cho {len(elements)} elements...")
        
        # Lọc ra những element cần sửa để tiết kiệm API call
        suspicious_indices = []
        for i, el in enumerate(elements):
            if force_all or self._is_suspicious(el.raw_text):
                suspicious_indices.append(i)
                
        if not suspicious_indices:
            logger.info("Không phát hiện element nào đáng ngờ. Bỏ qua Gemini Corrector.")
            return elements
            
        logger.info(f"Phát hiện {len(suspicious_indices)}/{len(elements)} elements cần xử lý. Đang gửi lên Gemini...")

        batch_size = int(os.getenv("GEMINI_CORRECTOR_BATCH_SIZE", "30"))
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
        delay_between_batches = float(os.getenv("GEMINI_DELAY_BETWEEN_BATCHES", "15.0"))
        max_batches = int(os.getenv("DEBUG_MAX_BATCHES", "0"))
        
        for i in range(0, len(suspicious_indices), batch_size):
            if max_batches > 0 and (i // batch_size) >= max_batches:
                logger.info(f"Đã đạt giới hạn DEBUG_MAX_BATCHES={max_batches}. Ngừng gọi API cho phần còn lại.")
                break
                
            batch_indices = suspicious_indices[i:i+batch_size]
            
            # Chuẩn bị payload
            payload = [{"id": idx, "text": elements[real_idx].raw_text} for idx, real_idx in enumerate(batch_indices)]
            payload_str = json.dumps(payload, ensure_ascii=False)
            
            logger.info(f"Gửi batch {i//batch_size + 1}/{(len(suspicious_indices)+batch_size-1)//batch_size} lên Gemini...")
            
            try:
                for attempt in range(max_retries):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=payload_str,
                            config=types.GenerateContentConfig(
                                system_instruction=self.system_prompt,
                                response_mime_type="application/json",
                                temperature=0.0
                            )
                        )
                        break # Success
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = delay_between_batches
                            if "429" in str(e) or "Quota" in str(e) or "exhausted" in str(e).lower():
                                wait_time = 35.0  # Chờ 35s nếu bị Rate Limit
                            logger.warning(f"Lỗi API (thử lại sau {wait_time}s): {e}")
                            time.sleep(wait_time)
                        else:
                            raise e
                
                # Parse response
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith("```"):
                    response_text = response_text[3:-3].strip()
                    
                corrected_data = json.loads(response_text)
                
                # Map lại vào elements gốc
                for idx, real_idx in enumerate(batch_indices):
                    if idx < len(corrected_data):
                        new_text = corrected_data[idx].get("text", elements[real_idx].raw_text)
                        elements[real_idx].raw_text = new_text
                        
            except Exception as e:
                logger.error(f"Lỗi khi gọi Gemini API cho batch {i//batch_size + 1}: {e}")
                # Nếu lỗi thì giữ nguyên text cũ (fallback an toàn)
            
            # Delay để tránh rate limit
            if i + batch_size < len(suspicious_indices):
                time.sleep(delay_between_batches)
                    
        return elements
