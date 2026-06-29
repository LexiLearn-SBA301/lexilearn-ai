import os
import re
import io
import time
import logging
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


logger = logging.getLogger("rag-service.pdf-reader")

import vietnamese

TCVN3_SIGNATURE_CHARS = "\u00b5\u00b8\u00b6\u00b7\u00b9\u00a8\u00bb\u00be\u00bc\u00bd\u00c6\u00a9\u00c7\u00cb\u00ae\u00d0\u00ce\u00cf\u00d1\u00aa\u00d6\u00d7\u00d8\u00dc\u00de\u00a7\u00a3\u00a4\u00a5\u00a6\u2212\u03bc\uf02d\u00a2\u00d5\u00f2\u00e8\u00f7"

from config.gemini_ocr_system_prompt import GEMINI_OCR_SYSTEM_PROMPT


@dataclass
class ExtractedElement:
    page: int
    type: str
    raw_text: str
    source_file: str

class PDFReader:
    """
    PDF Reader designed to ingest Vietnamese Literature textbook PDFs.
    Uses pdfplumber as the primary extraction engine with tabular data extraction support,
    and falls back to PyPDF2 if pdfplumber fails.
    """

    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load heading keywords from config
        config = self._load_config()
        self.heading_keywords = set(config["heading_keywords"])
        self._GEMINI_OCR_SYSTEM_PROMPT = GEMINI_OCR_SYSTEM_PROMPT

    def read(self, file_path: str) -> List[ExtractedElement]:
        """
        Reads a PDF file and extracts a list of structured elements.
        Falls back to PyPDF2 and DeepDoc + VietOCR if primary extraction fails or returns no content.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        source_file = os.path.basename(file_path)
        logger.info(f"Starting extraction for PDF file: {source_file}")

        # Tầng 1: pdfplumber (nhanh nhất, hỗ trợ bảng)
        elements = self._try_extract(self._extract_with_pdfplumber, file_path, source_file, "pdfplumber")
        if elements:
            return elements

        # Tầng 2: PyPDF2 (fallback cơ bản)
        elements = self._try_extract(self._extract_with_pypdf, file_path, source_file, "PyPDF2")
        if elements:
            return elements

        # Tầng 3: Gemini Vision OCR (cho PDF quét ảnh — dùng LLM multimodal)
        elements = self._try_extract(self._extract_with_gemini_ocr, file_path, source_file, "Gemini Vision OCR")
        if elements:
            return elements

        logger.error(f"Không thể trích xuất văn bản từ '{source_file}' bằng bất kỳ phương pháp nào.")
        return []

    def _try_extract(self, extract_func, file_path: str, source_file: str, method_name: str) -> List[ExtractedElement]:
        """
        Helper to run an extraction method safely. Returns elements if successful and not empty.
        """
        try:
            logger.info(f"Attempting extraction using {method_name}...")
            elements = extract_func(file_path, source_file)
            if elements:
                logger.info(f"Successfully extracted {len(elements)} elements using {method_name}.")
                return elements
            else:
                logger.warning(f"{method_name} returned 0 elements for '{source_file}'.")
                return []
        except Exception as e:
            logger.warning(f"Extraction with {method_name} failed for '{source_file}' due to: {e}")
            return []

    def _extract_with_pdfplumber(self, file_path: str, source_file: str) -> List[ExtractedElement]:
        """
        Extracts structured elements using pdfplumber. Handles tables and filters
        table text from general text extraction to avoid duplicates.
        """
        if pdfplumber is None:
            raise ImportError("pdfplumber is not installed. Please run: pip install pdfplumber")

        elements: List[ExtractedElement] = []

        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages): #Cơ chế: Đọc từng phong bì được lấy từ pdf.pages và dán nhãn cho phong bì đó - trang đầu sẽ là 0 nên là 0 + 1
                page_num = page_idx + 1
                try:
                    # Detect page-level TCVN3 encoding
                    page_raw_text = page.extract_text() or ""
                    page_sig_count = sum(1 for c in page_raw_text if c in TCVN3_SIGNATURE_CHARS)
                    self.current_page_is_tcvn3 = page_sig_count >= 5

                    tables = page.find_tables()
                    table_elements: List[ExtractedElement] = []
                    
                    for table in tables:
                        table_data = table.extract()#Dữ liệu của bảng và trả ra theo dạng list - list chứa các hàng - mỗi hàng là 1 list
                        if not table_data:
                            continue
                        
                        formatted_rows = []
                        for row in table_data:
                            cleaned_row = [self._clean_text(cell or "") for cell in row]
                            if any(cleaned_row):
                                formatted_rows.append(" | ".join(cleaned_row))
                        
                        if formatted_rows:
                            table_text = "\n".join(formatted_rows)
                            table_elements.append(
                                ExtractedElement(
                                    page=page_num,
                                    type="table",
                                    raw_text=table_text,
                                    source_file=source_file
                                )
                            )

                    if tables:
                        try:
                            filtered_page = page.filter(
                                lambda obj: not (
                                    obj.get("object_type") == "char"
                                    and any(self._is_in_bbox(obj, t.bbox) for t in tables)
                                )
                            ) #Lọc ra các dữ liệu không có trong bảng cùng trong 1 trang pdf -> tránh lặp dữ liệu
                            page_text = filtered_page.extract_text()
                        except Exception as filter_err:
                            logger.warning(
                                f"Failed to filter table characters on page {page_num} "
                                f"of {source_file}: {filter_err}. Proceeding with standard text extraction."
                            )
                            page_text = page.extract_text() #Lấy tất cả dữ liệu trong trang kể cả bảng luôn nhưng mà ta đã lọc ở trên rồi nên là nó sẽ lấy từ trang đã được lọc
                    else:
                        page_text = page.extract_text() #Không có bảng thì lấy tất cả dữ liệu trong trang

                    parsed_elements = self._parse_text_layout(page_text, page_num, source_file)
                    elements.extend(parsed_elements)
                    elements.extend(table_elements)
                    #Nếu dùng append -> [[Đoạn_văn_1, Đoạn_văn_2], Bảng_1] -> danh sách bị lồng nhau, 
                    # còn dùng extend -> [Đoạn_văn_1, Đoạn_văn_2, Bảng_1] -> danh sách phẳng

                except Exception as page_err:
                    logger.warning(
                        f"Error extracting page {page_num} from '{source_file}' using pdfplumber: {page_err}. "
                        "Skipping page."
                    )
                    continue

        return elements

    def _extract_with_pypdf(self, file_path: str, source_file: str) -> List[ExtractedElement]:
        """
        Fallback parser using PyPDF2. Treats all content as text layout (no table isolation).
        """
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is not installed. Please run: pip install PyPDF2")

        elements: List[ExtractedElement] = []

        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                try:
                    page_text = page.extract_text()
                    # Detect page-level TCVN3 encoding
                    page_raw_text = page_text or ""
                    page_sig_count = sum(1 for c in page_raw_text if c in TCVN3_SIGNATURE_CHARS)
                    self.current_page_is_tcvn3 = page_sig_count >= 5

                    parsed_elements = self._parse_text_layout(page_text, page_num, source_file)
                    elements.extend(parsed_elements)
                except Exception as page_err:
                    logger.warning(
                        f"Error extracting page {page_num} from '{source_file}' using PyPDF2: {page_err}. "
                        "Skipping page."
                    )
                    continue

        return elements

    # ── Gemini Vision OCR (Tầng 3) ────────────────────────────────────

    def _extract_with_gemini_ocr(self, file_path: str, source_file: str) -> List[ExtractedElement]:
        """
        Tầng 3: Dùng Gemini Vision (multimodal LLM) để đọc trực tiếp ảnh trang PDF.
        Render PDF thành ảnh JPEG, gom batch, gửi lên Gemini 2.0 Flash.
        """
        if genai is None:
            raise ImportError("google-genai is not installed. Run: pip install google-genai")

        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env. Gemini Vision OCR is disabled.")

        ocr_model = os.getenv("GEMINI_OCR_MODEL", "gemini-2.0-flash")
        pages_per_batch = int(os.getenv("OCR_PAGES_PER_BATCH", "10"))
        dpi = int(os.getenv("OCR_DPI", "150"))

        client = genai.Client(api_key=api_key)

        # 1. Render tất cả trang PDF thành ảnh JPEG
        logger.info(f"Rendering '{source_file}' to images at {dpi} DPI for Gemini Vision OCR...")
        page_images = self._render_pdf_to_images(file_path, dpi=dpi)
        total_pages = len(page_images)
        logger.info(f"Total pages to OCR: {total_pages}")

        if not page_images:
            return []

        # 2. Chia batch và gửi lên Gemini
        elements: List[ExtractedElement] = []
        batches = [page_images[i:i + pages_per_batch] for i in range(0, total_pages, pages_per_batch)]

        delay_between_batches = float(os.getenv("OCR_DELAY_BETWEEN_BATCHES", "4"))

        for batch_idx, batch in enumerate(batches):
            page_start = batch[0][0]
            page_end = batch[-1][0]
            logger.info(f"Gemini Vision OCR batch {batch_idx + 1}/{len(batches)} (pages {page_start}-{page_end})...")

            ocr_text = self._call_gemini_ocr(client, ocr_model, batch)
            if not ocr_text:
                logger.warning(f"Gemini Vision OCR returned empty for batch {batch_idx + 1}. Skipping.")
                continue

            # 3. Parse response theo marker === TRANG N ===
            page_texts = self._parse_gemini_ocr_response(ocr_text, [p[0] for p in batch])

            for page_num, page_text in page_texts.items():
                if page_text.strip().startswith("[HÌNH ẢNH"):
                    continue
                self.current_page_is_tcvn3 = False
                parsed = self._parse_gemini_text_layout(page_text, page_num, source_file)
                elements.extend(parsed)

            # Delay giữa các batch để tránh rate limit (15 RPM free tier)
            if batch_idx < len(batches) - 1:
                time.sleep(delay_between_batches)

        return elements

    def _render_pdf_to_images(self, file_path: str, dpi: int = 150) -> List[Tuple[int, bytes]]:
        """Render tất cả trang PDF thành list[(page_num, jpeg_bytes)]."""
        pages: List[Tuple[int, bytes]] = []

        if pdfplumber is not None:
            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    try:
                        img = page.to_image(resolution=dpi).original
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        pages.append((page_idx + 1, buf.getvalue()))
                    except Exception as e:
                        logger.warning(f"Failed to render page {page_idx + 1}: {e}")
        else:
            # Fallback: dùng PyMuPDF (fitz) nếu có
            try:
                import fitz
                doc = fitz.open(file_path)
                for i, page in enumerate(doc):
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    pages.append((i + 1, pix.tobytes("jpeg")))
                doc.close()
            except ImportError:
                raise RuntimeError("Cần pdfplumber hoặc PyMuPDF (fitz) để render PDF thành ảnh.")

        return pages

    def _call_gemini_ocr(self, client, model_name: str, batch: List[Tuple[int, bytes]]) -> Optional[str]:
        """Gửi 1 batch ảnh trang lên Gemini và nhận text OCR."""
        page_nums = [p[0] for p in batch]

        # Build multimodal content
        contents = [f"Trích xuất text từ {len(batch)} trang PDF sau (trang {page_nums[0]} đến {page_nums[-1]}):"]

        for page_num, img_bytes in batch:
            contents.append(f"\n--- Trang {page_num} ---")
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        contents.append("\nTrả về text đã trích xuất, bắt đầu mỗi trang bằng === TRANG {n} ===")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self._GEMINI_OCR_SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=65536,
                    ),
                )
                if response.text:
                    return response.text
                
                # If we get here, response.text is empty or None
                # Raise an exception to trigger the retry logic
                raise ValueError(f"Gemini Vision OCR returned empty response. Safety ratings: {getattr(response, 'prompt_feedback', 'N/A')}")
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    err_str = str(e)
                    err_str_upper = err_str.upper()
                    if "429" in err_str_upper or "503" in err_str_upper or "EXHAUSTED" in err_str_upper:
                        wait = max(wait, 15)
                        # Try to parse recommended retry delay from Gemini API response
                        import re
                        match = re.search(r"Please retry in (\d+\.?\d*)s", err_str, re.IGNORECASE)
                        if match:
                            try:
                                wait = int(float(match.group(1))) + 2
                            except ValueError:
                                pass
                        else:
                            match_delay = re.search(r"retrydelay'?:?\s*'?(\d+)", err_str, re.IGNORECASE)
                            if match_delay:
                                try:
                                    wait = int(match_delay.group(1)) + 2
                                except ValueError:
                                    pass
                    logger.warning(f"Gemini Vision OCR failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini Vision OCR failed after {max_retries} attempts: {e}")
                    return None
        return None

    def _parse_gemini_ocr_response(self, response: str, expected_pages: List[int]) -> dict:
        """Parse Gemini OCR response theo marker === TRANG N ===, trả về {page_num: text}."""
        result = {}
        response = response.replace('\r\n', '\n')

        # Strip markdown wrapper nếu có
        text = response.strip()
        if text.startswith("```"):
            lines = text.split('\n')
            if len(lines) > 1:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = '\n'.join(lines)

        for i, page_num in enumerate(expected_pages):
            start_idx = -1
            marker_len = 0

            # 1. Thử tìm chính xác marker === TRANG {page_num} ===
            marker = f"=== TRANG {page_num} ==="
            idx = text.find(marker)
            if idx != -1:
                start_idx = idx
                marker_len = len(marker)
            else:
                # 2. Thử tìm bằng regex linh hoạt (không phân biệt hoa thường, cho phép các ký tự đặc biệt bao quanh)
                pattern = re.compile(
                    rf"(?:[=\-#\*]{{1,}}\s*)?(?:TRANG|trang)\s+{page_num}\b(?:\s*[=\-#\*]{{1,}})?",
                    re.IGNORECASE
                )
                match = pattern.search(text)
                if match:
                    start_idx = match.start()
                    marker_len = match.end() - match.start()

            if start_idx == -1:
                logger.warning(f"Marker for page {page_num} not found in Gemini OCR response. Skipping.")
                continue

            start_content = start_idx + marker_len

            # Tìm vị trí kết thúc (bắt đầu của trang tiếp theo)
            end_idx = len(text)
            if i + 1 < len(expected_pages):
                next_page_num = expected_pages[i + 1]
                # 1. Thử tìm chính xác marker === TRANG {next_page_num} ===
                next_marker = f"=== TRANG {next_page_num} ==="
                next_idx = text.find(next_marker, start_content)
                if next_idx != -1:
                    end_idx = next_idx
                else:
                    # 2. Thử tìm bằng regex linh hoạt cho trang kế tiếp
                    next_pattern = re.compile(
                        rf"(?:[=\-#\*]{{1,}}\s*)?(?:TRANG|trang)\s+{next_page_num}\b(?:\s*[=\-#\*]{{1,}})?",
                        re.IGNORECASE
                    )
                    next_match = next_pattern.search(text, start_content)
                    if next_match:
                        end_idx = next_match.start()

            result[page_num] = text[start_content:end_idx].strip()

        return result

    def _parse_gemini_text_layout(self, text: Optional[str], page_num: int, source_file: str) -> List[ExtractedElement]:
        """
        Parses Gemini OCR output with markdown heading markers.
        """
        if not text:
            return []

        lines = text.split("\n")
        elements: List[ExtractedElement] = []
        current_para_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("#### "):
                if current_para_lines:
                    elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                    current_para_lines = []
                heading_text = self._clean_text(stripped[5:])
                elements.append(ExtractedElement(page=page_num, type="heading", raw_text=heading_text, source_file=source_file))
            elif stripped.startswith("### "):
                if current_para_lines:
                    elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                    current_para_lines = []
                heading_text = self._clean_text(stripped[4:])
                elements.append(ExtractedElement(page=page_num, type="heading", raw_text=heading_text, source_file=source_file))
            elif stripped.startswith("## "):
                if current_para_lines:
                    elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                    current_para_lines = []
                heading_text = self._clean_text(stripped[3:])
                elements.append(ExtractedElement(page=page_num, type="heading", raw_text=heading_text, source_file=source_file))
            else:
                cleaned = self._clean_text(stripped, is_final=False)
                if not cleaned:
                    continue
                current_para_lines.append(cleaned)
                if cleaned[-1] in (".", "?", "!", "\u201d", '"'):
                    elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                    current_para_lines = []

        if current_para_lines:
            elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))

        return elements

    def _parse_text_layout(self, text: Optional[str], page_num: int, source_file: str) -> List[ExtractedElement]:
        """
        Parses raw text layout of a page line-by-line, merging lines into paragraphs
        where appropriate, and detecting headings and list items.
        """
        if not text:
            return []

        lines = text.split("\n")
        elements: List[ExtractedElement] = []
        current_para_lines: List[str] = []

        for line in lines:
            cleaned = self._clean_text(line, is_final=False)
            if not cleaned:
                continue

            if self._is_numbered_item(cleaned):
                if current_para_lines:
                    elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                    current_para_lines = []
                elements.append(ExtractedElement(page=page_num, type="numbered_item", raw_text=cleaned, source_file=source_file))
            elif self._is_heading(cleaned):
                if current_para_lines:
                    elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                    current_para_lines = []
                
                heading_text = cleaned
                if cleaned.lower().startswith("đọc thêm") or cleaned.lower().startswith("0ọc thêm"):
                    heading_text = self._clean_doc_them_heading(cleaned)
                
                elements.append(
                    ExtractedElement(
                        page=page_num,
                        type="heading",
                        raw_text=heading_text,
                        source_file=source_file
                    )
                )
            elif self._is_list_item(cleaned):
                if current_para_lines:
                    elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                    current_para_lines = []

                elements.append(
                    ExtractedElement(
                        page=page_num,
                        type="list",
                        raw_text=cleaned,
                        source_file=source_file
                    )
                )
            else:
                if not current_para_lines:
                    current_para_lines.append(cleaned)
                else:
                    prev_line = current_para_lines[-1]
                    if prev_line and prev_line[-1] in (".", "?", "!", "”", '"'):
                        elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                        current_para_lines = [cleaned]
                    else:
                        current_para_lines.append(cleaned)

        if current_para_lines:
            elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))

        return elements

    def _build_paragraph_element(self, lines: List[str], page_num: int, source_file: str) -> ExtractedElement:
        """
        Combines list of lines into a single paragraph element.
        """
        joined_text = " ".join(lines)
        return ExtractedElement(

            page=page_num,
            type="paragraph",
            raw_text=self._clean_text(joined_text),
            source_file=source_file
        )

    _config: Optional[dict] = None
    _ocr_correction_patterns: Optional[List[tuple]] = None
    _ocr_corrections: Optional[dict] = None
    _compiled_word_corrections: Optional[List[tuple]] = None

    @classmethod
    def _load_config(cls) -> dict:
        """Lazily load configuration from pdf_reader_config.json."""
        if cls._config is None:
            import json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(current_dir, "..", "config", "pdf_reader_config.json"))
            with open(config_path, "r", encoding="utf-8") as f:
                cls._config = json.load(f)
        return cls._config

    @classmethod
    def _get_ocr_patterns(cls) -> List[tuple]:
        """Lazily compile OCR correction regex patterns."""
        if cls._ocr_correction_patterns is None:
            cls._ocr_correction_patterns = []
            config = cls._load_config()
            ocr_v_syllables = config["ocr_v_syllables"]
            viet_alpha = config["viet_alpha"]
            
            for syllable in ocr_v_syllables:
                wrong_lower = 'u' + syllable[1:]
                wrong_upper = 'U' + syllable[1:]
                correct_upper = 'V' + syllable[1:]
                # Negative lookbehind/lookahead for Vietnamese alphabetic chars
                # ensures we only match at syllable boundaries
                lb = rf'(?<![{viet_alpha}])'
                la = rf'(?![{viet_alpha}])'
                cls._ocr_correction_patterns.append((
                    re.compile(lb + re.escape(wrong_lower) + la),
                    syllable
                ))
                cls._ocr_correction_patterns.append((
                    re.compile(lb + re.escape(wrong_upper) + la),
                    correct_upper
                ))
        return cls._ocr_correction_patterns

    @classmethod
    def _load_ocr_corrections(cls) -> dict:
        """Lazily load OCR correction dictionary from JSON config."""
        if cls._ocr_corrections is None:
            import json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.normpath(os.path.join(current_dir, "..", "config", "ocr_corrections.json"))
            with open(path, "r", encoding="utf-8") as f:
                cls._ocr_corrections = json.load(f)
            
            # Pre-compile case-insensitive word matching regex patterns
            cls._compiled_word_corrections = []
            config = cls._load_config()
            viet_alpha = config["viet_alpha"]
            
            lb = rf'(?<![{viet_alpha}])'
            la = rf'(?![{viet_alpha}])'
            # Sort corrections by key length (longest first) to prevent
            # shorter patterns from consuming characters needed by longer ones
            # (e.g. "ghỉ" must not match before "nghỉ lễ thử lửa")
            word_items = sorted(
                cls._ocr_corrections["word_corrections"].items(),
                key=lambda item: len(item[0]),
                reverse=True
            )
            for wrong, correct in word_items:
                pattern = re.compile(lb + re.escape(wrong) + la, re.IGNORECASE)
                cls._compiled_word_corrections.append((pattern, wrong, correct))
                
        return cls._ocr_corrections

    def _apply_ocr_corrections(self, text: str, is_final: bool = True, step: str = "all") -> str:
        """Applies data-driven OCR corrections from config to avoid hardcoded regexes."""
        if not text:
            return ""
            
        corrections = self._load_ocr_corrections()
        
        if step in ("pre_tcvn3", "all"):
            # 1. Immediate sentence-start or general character corrections for £ sign
            # This has to happen before TCVN3 check.
            if is_final:
                text = re.sub(r'^£ơ\b', 'Thơ', text)
                text = re.sub(r'(?<=\.\s)£ơ\b', 'Thơ', text)
                text = re.sub(r'(?<=\n)£ơ\b', 'Thơ', text)
                
                text = re.sub(r'^£', 'T', text)
                text = re.sub(r'(?<=\.\s)£', 'T', text)
                text = re.sub(r'(?<=\n)£', 'T', text)
                
            # Apply standard character-level replacements
            for wrong, correct in corrections["char_corrections"].items():
                text = text.replace(wrong, correct)
                
        if step in ("post_tcvn3", "all"):
            # Replace OCR typo '0ọc'/'0ỌC' -> 'đọc'/'ĐỌC' (dynamic case preservation)
            def replace_0oc(match):
                m = match.group(0)
                if m == '0ỌC':
                    return 'ĐỌC'
                elif m == '0ọc':
                    return 'đọc'
                return 'Đọc'
            text = re.sub(r'\b0ọc\b', replace_0oc, text, flags=re.IGNORECASE)

            # 2. Case-aware word replacements from JSON
            word_corrections = self._compiled_word_corrections or []
            for pattern, wrong, correct in word_corrections:
                def case_aware_replace(match, _correct=correct):
                    original = match.group(0)
                    if original.isupper():
                        return _correct.upper()
                    elif original and original[0].isupper():
                        return _correct[0].upper() + _correct[1:]
                    return _correct.lower()
                text = pattern.sub(case_aware_replace, text)
                
            # 3. Final sentence-start capitalizations
            if is_final:
                for wrong, correct in corrections["sentence_start_corrections"].items():
                    text = re.sub(r'^' + re.escape(wrong) + r'\b', correct, text)
                    text = re.sub(r'(?<=\.\s)' + re.escape(wrong) + r'\b', correct, text)
                    
        return text

    def _tcvn3_to_unicode(self, text: str, force: bool = False) -> str:
        """
        Dynamically detects if a text is TCVN3-encoded and decodes it to Unicode.
        """
        if not text:
            return ""
        tcvn3_sig_count = sum(1 for c in text if c in TCVN3_SIGNATURE_CHARS)
        
        # If the page is TCVN3-encoded, any element with at least 1 signature character is TCVN3.
        # Otherwise, we use the standard threshold of 2 (or 1 for short texts).
        threshold = 1 if force else 2
        
        if tcvn3_sig_count >= threshold or (not force and tcvn3_sig_count >= 1 and len(text) < 30):
            # Pre-process PDF extraction hyphen artifacts before converting
            processed_text = text.replace("−", "\u00ad").replace("", "-").replace("\u03bc", "\u00b5")
            return vietnamese.normalize(processed_text, target_charset="UNICODE", source_charset="TCVN3")
        return text

    def _clean_text(self, text: str, is_final: bool = True) -> str:
        """
        Normalizes spaces and Unicode characters for Vietnamese text.
        """
        if not text:
            return ""
        
        # 1. Pre-TCVN3: Apply early character replacements (£ -> t/T) to prevent false positives in encoding detection
        text = self._apply_ocr_corrections(text, is_final=is_final, step="pre_tcvn3")

        # 2. Decode TCVN3 to Unicode and normalize
        force_tcvn3 = getattr(self, "current_page_is_tcvn3", False)
        text = self._tcvn3_to_unicode(text, force=force_tcvn3)
        normalized = unicodedata.normalize("NFC", text)
        collapsed_spaces = re.sub(r'[ \t\r\f\v]+', ' ', normalized)
        
        # 3. Post-TCVN3: Apply word-level corrections and final sentence-start formatting
        text_cleaned = self._apply_ocr_corrections(collapsed_spaces, is_final=is_final, step="post_tcvn3")
        
        return text_cleaned.strip()


    def _is_in_bbox(self, char: dict, bbox: tuple) -> bool:
        """
        Returns True if the char obj lies inside the given bounding box (x0, top, x1, bottom).
        """
        x0 = char.get("x0")
        top = char.get("top")
        x1 = char.get("x1")
        bottom = char.get("bottom")
        
        if x0 is None or top is None or x1 is None or bottom is None:
            return False
            
        tx0, ttop, tx1, tbottom = bbox
        return (
            x0 >= tx0 - 1
            and x1 <= tx1 + 1
            and top >= ttop - 1
            and bottom <= tbottom + 1
        )

    def _is_heading(self, line: str) -> bool:
        """
        Heuristic to detect if a line is a Heading in Vietnamese textbook materials.
        """
        if not line or len(line) > 150:
            return False

        has_letters = any(c.isalpha() for c in line)
        if has_letters:
            # Check if mostly uppercase (at least 75% of letters are uppercase)
            # Exclude content in parentheses when checking uppercase ratio
            line_no_parens = re.sub(r'\(.*?\)', '', line)
            letters = [c for c in line_no_parens if c.isalpha()]
            upper_letters = [c for c in letters if c.isupper()]
            if letters and len(upper_letters) / len(letters) >= 0.75:
                return True

        if re.match(r'^[IVXLCDM]+\.?\s+', line):
            return True

        vietnamese_caps = "A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
        if re.match(rf'^\d+\.\d+(\.\d+)*\.?\s+[{vietnamese_caps}]', line):
            return True

        lower_line = line.lower()
        for kw in self.heading_keywords:
            if lower_line.startswith(kw):
                after_kw = lower_line[len(kw):]
                if not after_kw or not after_kw[0].isalnum():
                    if kw in {"văn bản", "bài", "chương", "phần"}:
                        after_kw_stripped = after_kw.strip()
                        if not after_kw_stripped:
                            return True
                        if after_kw_stripped.startswith(":") or after_kw_stripped.startswith("-"):
                            return True
                        words = after_kw_stripped.split()
                        first_word = words[0].rstrip(".:-") if words else ""
                        if (first_word.isdigit() or 
                            re.match(r'^[ivxlcdm]+$', first_word) or
                            first_word in {"học", "đọc", "tập", "trích", "thành", "phụ", "số"}):
                            return True
                        return False
                    return True

        return False

    def _is_numbered_item(self, line: str) -> bool:
        """Detect single-level numbered line like '1. Tác giả' or '2. So sánh...'"""
        if not line:
            return False
        vietnamese_caps = "A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
        return bool(re.match(rf'^\d+\.?\s+[{vietnamese_caps}]', line))

    def _is_list_item(self, line: str) -> bool:
        """
        Heuristic to check if a line is a list item.
        """
        if not line:
            return False
        
        return bool(re.match(r'^([•\-\*\+\–\—]|\d+[\)\.]|[a-zA-Z][\)\.])\s+', line))

    def _clean_doc_them_heading(self, line: str) -> str:
        """
        Normalizes and formats reading selection headings (Đọc thêm).
        Isolates Level 0 titles in 100% uppercase and corrects common OCR errors.
        """
        cleaned = re.sub(r'^[0oO]ọc\s+thêm', 'ĐỌC THÊM', line, flags=re.IGNORECASE)
        cleaned = re.sub(r'^đọc\s+thêm', 'ĐỌC THÊM', cleaned, flags=re.IGNORECASE)
        
        if not cleaned.startswith('ĐỌC THÊM'):
            return line
            
        # Strip parenthetical expressions, e.g. (Trích ...)
        cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned)
        
        # Clean up the separator after 'ĐỌC THÊM'
        match = re.match(r'^ĐỌC THÊM[\s,:\-\—\–]*(.*)$', cleaned, flags=re.IGNORECASE)
        if match:
            title_part = match.group(1).strip()
            # Clean up trailing noise
            title_part = re.sub(r'[\s\)\-\—\–]+$', '', title_part)
            title_part = re.sub(r'\bLm\b', '', title_part).strip()
            title_part = title_part.strip(",.-:")
            
            # Specific correction for 'TIÊN DẶN' -> 'TIỄN DẶN'
            title_part = re.sub(r'\bTIÊN\s+DẶN\b', 'TIỄN DẶN', title_part, flags=re.IGNORECASE)
            
            title_part = title_part.upper()
            if title_part:
                return f"ĐỌC THÊM: {title_part}"
            else:
                return "ĐỌC THÊM"
        return cleaned