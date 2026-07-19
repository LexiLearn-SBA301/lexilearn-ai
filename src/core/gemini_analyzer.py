import os
import json
import logging
import time
import unicodedata
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
        # Load genre taxonomy ALWAYS (needed for local normalization even if API is missing)
        taxonomy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "genre_taxonomy.json")
        taxonomy_path = os.path.normpath(taxonomy_path)
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy = json.load(f)
        
        # Build reverse alias lookup
        self._genre_alias_map = {}
        self._valid_genres = set()
        self._valid_sub_genres = set()
        for genre_slug, genre_info in self.taxonomy["genres"].items():
            self._valid_genres.add(genre_slug)
            for alias in genre_info.get("aliases", []):
                self._genre_alias_map[unicodedata.normalize('NFC', alias.strip().lower())] = genre_slug
            for sg in genre_info.get("sub_genres", []):
                self._valid_sub_genres.add(sg)
                
        # Build valid genre list string for the prompt
        genre_examples = ", ".join(f"'{g}'" for g in self._valid_genres)
        sub_genre_examples = ", ".join(f"'{sg}'" for sg in list(self._valid_sub_genres)[:8])

        if not genai:
            logger.warning("google-genai package is not installed. GeminiAnalyzer will not do AI extraction.")
            self.client = None
            return
            
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY environment variable not found. GeminiAnalyzer will not do AI extraction.")
            self.client = None
            return
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_ANALYZER_MODEL", model_name)
        

        self.system_prompt = (
            "Bạn là chuyên gia phân tích văn bản sách giáo khoa Ngữ Văn. Nhiệm vụ của bạn là đọc các đoạn văn (chunk) và trích xuất Metadata.\n\n"
            "QUY TẮC PHÂN TÍCH:\n"
            "1. work_title: Tên tác phẩm CHÍNH đang được học. Nếu là một văn bản trích dẫn nhỏ trong bài, hãy lấy tên bài học lớn chứa nó (nếu có trong structural_context).\n"
            "   - Lưu ý: KHÔNG dùng các từ như 'Phần 1', 'Tiểu dẫn'. Hãy trích xuất chính xác tên tác phẩm (VD: Tỏ Lòng, Vợ Nhặt, Bình ngô đại cáo).\n"
            "2. author_name: Tên tác giả của tác phẩm. NẾU phát hiện tác phẩm đó do ai viết thì điền chính xác tên tác giả (VD: Phạm Ngũ Lão, Kim Lân).\n"
            "   - Nếu nội dung thuộc bài giảng kiến thức chung, sách giáo khoa tổng kết → 'Bộ Giáo Dục và Đào Tạo'.\n"
            "3. is_biography: Nội dung chunk có phải phần 'Tiểu dẫn' hoặc 'Giới thiệu tiểu sử tác giả' không? (True/False).\n"
            "4. author_period: Thời đại của tác giả (dan_gian, trung_dai, can_dai, hien_dai). Dựa vào năm sinh/thế kỷ để nhận diện.\n"
            "5. work_period: Thời kỳ của tác phẩm (dan_gian, trung_dai, can_dai, hien_dai). Dựa vào năm sáng tác/thế kỷ để nhận diện.\n"
            f"6. genre: Thể loại chính. CHỈ ĐƯỢC CHỌN MỘT TRONG CÁC GIÁ TRỊ SAU: [{genre_examples}]. Nếu chưa rõ thì để 'van_hoc'.\n"
            f"7. sub_genre: Thể loại con cụ thể, dạng snake_case (VD: {sub_genre_examples}...). Nếu không rõ, hãy để là null.\n"
            "8. publish_year: Lấy chính xác từ \"Năm sáng tác\" hoặc từ khóa tương tự trong văn bản. Trả về số nguyên (VD: 1948). Nếu không có hoặc không rõ, trả về null.\n\n"
            "Input: JSON array [{'chunk_id': '...', 'text': '...', 'structural_context': '...'}, ...]\n"
            "Output: JSON array [{'chunk_id': '...', 'work_title': '...', 'author_name': '...', 'is_biography': true/false, 'author_period': '...', 'work_period': '...', 'genre': '...', 'sub_genre': '...', 'publish_year': 1948 hoặc null}, ...].\n"
            "CHỈ TRẢ VỀ JSON HỢP LỆ, không kèm theo text giải thích nào khác."
        )

    def normalize_genre(self, raw_genre: str | None) -> str:
        """Normalize a raw genre value from AI into a valid slug. Falls back to 'van_hoc'."""
        if not raw_genre:
            return "van_hoc"
        key = unicodedata.normalize('NFC', raw_genre.strip().lower())
        # Direct match (already a valid slug)
        if key in self._valid_genres:
            return key
        # Alias lookup
        if key in self._genre_alias_map:
            return self._genre_alias_map[key]
        # Try removing accents as last resort
        stripped = self._strip_accents(key)
        for alias_key, genre_slug in self._genre_alias_map.items():
            if self._strip_accents(alias_key) == stripped:
                return genre_slug
        logger.warning(f"Genre '{raw_genre}' not found in taxonomy, falling back to 'van_hoc'")
        return "van_hoc"
    
    def normalize_sub_genre(self, raw_sub_genre: str | None) -> str | None:
        """Normalize a raw sub_genre value. Returns None if invalid or empty."""
        if not raw_sub_genre:
            return None
        key = unicodedata.normalize('NFC', raw_sub_genre.strip().lower().replace(" ", "_"))
        if key in self._valid_sub_genres:
            return key
        logger.warning(f"Sub-genre '{raw_sub_genre}' not found in taxonomy, setting to None")
        return None
    
    @staticmethod
    def _strip_accents(text: str) -> str:
        """Remove Vietnamese diacritics for fallback comparison."""
        nfkd = unicodedata.normalize('NFKD', text)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).replace('đ', 'd').replace('Đ', 'D')

    @staticmethod
    def _build_structural_context(chunk: SemanticChunk) -> str:
        """
        Build a hierarchical context string from a chunk's section metadata.
        Example output: "PHẦN HAI - LỊCH SỬ VĂN HỌC > Khái quát văn học Việt Nam > I. Bối cảnh"
        """
        parts = []
        if chunk.section_title:
            parts.append(chunk.section_title)
        if chunk.section_slug and chunk.section_slug != chunk.section_title:
            parts.append(chunk.section_slug)
        
        # If we have nothing meaningful, fall back to work_title
        if not parts:
            return chunk.work_title or ""
        
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
                        c.work_title = info.get("work_title")
                        c.author_name = info.get("author_name")
                        c.author_period = info.get("author_period")
                        c.work_period = info.get("work_period")
                        c.genre = self.normalize_genre(info.get("genre"))
                        c.sub_genre = self.normalize_sub_genre(info.get("sub_genre"))
                        # publish_year: store as attribute for IngestService to pick up
                        raw_year = info.get("publish_year")
                        c.publish_year = int(raw_year) if raw_year is not None else None
                        
            except Exception as e:
                logger.error(f"Lỗi khi gọi Gemini Analyzer cho batch {i//batch_size + 1}: {e}")
                # Nếu lỗi thì giữ nguyên giá trị None/False mặc định
            
            # Delay để tránh rate limit
            if i + batch_size < len(chunks):
                time.sleep(delay_between_batches)
                    
        return chunks

