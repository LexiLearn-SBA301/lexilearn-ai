import os
import json
import logging
import time
from typing import List

from core.semantic_chunker import SemanticChunk

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger("rag-service.gemini-analyzer")

class GeminiAnalyzer:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        if not genai:
            logger.warning("google-genai package is not installed. GeminiAnalyzer will do nothing.")
            self.client = None
            return
            
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY environment variable not found. GeminiAnalyzer will do nothing.")
            self.client = None
            return
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_ANALYZER_MODEL", model_name)
        
        self.system_prompt = (
            "Bạn là chuyên gia phân tích văn bản sách giáo khoa Ngữ Văn. Nhiệm vụ của bạn là đọc các đoạn văn (chunk) và trích xuất Metadata.\n\n"
            "QUY TẮC PHÂN TÍCH:\n"
            "1. ten_tac_pham: Tên tác phẩm / Đề bài học lớn chứa nội dung này.\n"
            "   - Bạn được cung cấp 'structural_context' là ĐƯỜNG DẪN PHÂN CẤP của bài học trong sách (VD: 'PHẦN HAI > Khái quát văn học > I. Bối cảnh').\n"
            "   - Hãy ƯU TIÊN lấy tên tác phẩm/bài học từ cấp cao nhất có ý nghĩa trong structural_context.\n"
            "   - TUYỆT ĐỐI KHÔNG lấy tên các bài thơ/văn TRÍCH DẪN nhỏ nằm bên trong nội dung chunk để làm ten_tac_pham.\n"
            "   - VD: structural_context = 'Khái quát văn học > I. Bối cảnh' và trong text có trích thơ 'Từ ấy' → ten_tac_pham = 'Khái quát văn học', KHÔNG ĐƯỢC đổi thành 'Từ ấy'.\n"
            "2. tac_gia: Tên tác giả.\n"
            "   - Nếu nội dung thuộc bài giảng kiến thức chung (Tổng kết, Luyện tập, Khái quát văn học, Tiếng Việt, Làm văn, Hướng dẫn đọc thêm...) → điền 'Bộ Giáo Dục và Đào Tạo'.\n"
            "   - Nếu là tác phẩm văn học cụ thể → điền đúng tên tác giả.\n"
            "3. is_biography: Nội dung chunk có phải phần 'Tiểu dẫn' hoặc 'Giới thiệu tiểu sử tác giả' không? (True/False).\n"
            "4. nam_sang_tac: Năm sáng tác của tác phẩm (nếu biết hoặc có thể suy ra từ nội dung/kiến thức nền). Trả về số nguyên (VD: 1948). Nếu không rõ, trả về null.\n\n"
            "Input: JSON array [{'chunk_id': '...', 'text': '...', 'structural_context': '...'}, ...]\n"
            "Output: JSON array [{'chunk_id': '...', 'ten_tac_pham': '...', 'tac_gia': '...', 'is_biography': true/false, 'nam_sang_tac': 1948 hoặc null}, ...].\n"
            "CHỈ TRẢ VỀ JSON HỢP LỆ, không kèm theo text giải thích nào khác."
        )

    @staticmethod
    def _build_structural_context(chunk: SemanticChunk) -> str:
        """
        Build a hierarchical context string from a chunk's section metadata.
        Example output: "PHẦN HAI - LỊCH SỬ VĂN HỌC > Khái quát văn học Việt Nam > I. Bối cảnh"
        """
        parts = []
        if chunk.parent_section:
            parts.append(chunk.parent_section)
        if chunk.section_title and chunk.section_title != chunk.parent_section:
            parts.append(chunk.section_title)
        if chunk.subsection_title and chunk.subsection_title != chunk.section_title:
            parts.append(chunk.subsection_title)
        
        # If we have nothing meaningful, fall back to title
        if not parts:
            return chunk.title or ""
        
        return " > ".join(parts)

    def analyze(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        if not self.client or not chunks:
            return chunks

        logger.info(f"Bắt đầu quy trình Gemini Analyzer trích xuất metadata cho {len(chunks)} chunks...")

        batch_size = int(os.getenv("GEMINI_ANALYZER_BATCH_SIZE", "5")) 
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
        delay_between_batches = float(os.getenv("GEMINI_DELAY_BETWEEN_BATCHES", "15.0"))
        max_batches = int(os.getenv("DEBUG_MAX_BATCHES", "0"))
        
        for i in range(0, len(chunks), batch_size):
            if max_batches > 0 and (i // batch_size) >= max_batches:
                logger.info(f"Đã đạt giới hạn DEBUG_MAX_BATCHES={max_batches}. Ngừng gọi API cho phần còn lại.")
                break
                
            batch_chunks = chunks[i:i+batch_size]
            
            payload = [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.content,
                    "structural_context": self._build_structural_context(c)
                }
                for c in batch_chunks
            ]
            payload_str = json.dumps(payload, ensure_ascii=False)
            
            logger.info(f"Gửi batch {i//batch_size + 1}/{(len(chunks)+batch_size-1)//batch_size} lên Gemini Analyzer...")
            
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
                
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith("```"):
                    response_text = response_text[3:-3].strip()
                    
                analyzed_data = json.loads(response_text)
                
                # Map lại vào chunks
                data_dict = {item["chunk_id"]: item for item in analyzed_data if "chunk_id" in item}
                
                for c in batch_chunks:
                    if c.chunk_id in data_dict:
                        info = data_dict[c.chunk_id]
                        c.ten_tac_pham = info.get("ten_tac_pham")
                        c.tac_gia = info.get("tac_gia")
                        c.is_biography = bool(info.get("is_biography", False))
                        # nam_sang_tac: store as attribute for IngestService to pick up
                        raw_year = info.get("nam_sang_tac")
                        c.nam_sang_tac = int(raw_year) if raw_year is not None else None
                        
            except Exception as e:
                logger.error(f"Lỗi khi gọi Gemini Analyzer cho batch {i//batch_size + 1}: {e}")
                # Nếu lỗi thì giữ nguyên giá trị None/False mặc định
            
            # Delay để tránh rate limit
            if i + batch_size < len(chunks):
                time.sleep(delay_between_batches)
                    
        return chunks

