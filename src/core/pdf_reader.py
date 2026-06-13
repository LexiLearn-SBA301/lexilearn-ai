import os
import re
import logging
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

from PIL import Image

logger = logging.getLogger("rag-service.pdf-reader")

import vietnamese

TCVN3_SIGNATURE_CHARS = "\u00b5\u00b8\u00b6\u00b7\u00b9\u00a8\u00bb\u00be\u00bc\u00bd\u00c6\u00a9\u00c7\u00cb\u00ae\u00d0\u00ce\u00cf\u00d1\u00aa\u00d6\u00d7\u00d8\u00dc\u00de\u00a7\u00a3\u00a4\u00a5\u00a6\u2212\u03bc\uf02d"

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
        
        # Configure Tesseract and Poppler paths if provided
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if pytesseract and tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.poppler_path: Optional[str] = os.getenv("POPPLER_PATH")
        
        # Load heading keywords from config
        config = self._load_config()
        self.heading_keywords = set(config["heading_keywords"])

    def read(self, file_path: str) -> List[ExtractedElement]:
        """
        Reads a PDF file and extracts a list of structured elements.
        Falls back to PyPDF2 and Tesseract OCR if primary extraction fails or returns no content.
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

        # Tầng 3: Tesseract OCR (cho PDF quét ảnh)
        elements = self._try_extract(self._extract_with_ocr, file_path, source_file, "Tesseract OCR")
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

    def _extract_with_ocr(self, file_path: str, source_file: str) -> List[ExtractedElement]:
        """
        Third fallback parser using Tesseract OCR for scanned image PDFs.
        Converts PDF pages to images and runs OCR to extract text.
        """
        if pytesseract is None or convert_from_path is None:
            raise ImportError(
                "OCR dependencies are not installed. "
                "Please run: pip install pytesseract pdf2image"
            )

        elements: List[ExtractedElement] = []
        
        logger.info(f"Converting '{source_file}' to images for OCR...")
        try:
            # Convert PDF pages to PIL images with high DPI for better font recognition
            if self.poppler_path:
                images = convert_from_path(file_path, dpi=300, poppler_path=self.poppler_path)
            else:
                images = convert_from_path(file_path, dpi=300)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images (check poppler installation): {e}")

        total_pages = len(images)
        logger.info(f"Starting OCR for {total_pages} pages in '{source_file}'...")

        for page_idx, image in enumerate(images):
            page_num = page_idx + 1
            if page_num % 10 == 0 or page_num == 1:
                logger.info(f"OCR progress: page {page_num}/{total_pages}")
                
            try:
                # Preprocess: convert to grayscale (optimal for textbooks to preserve decorative letters)
                preprocessed_image = self._preprocess_for_ocr(image, use_binarization=False)
                
                # Run Tesseract OCR for Vietnamese with optimized LSTM config and automatic layout
                custom_config = r'--oem 1'
                raw_text = pytesseract.image_to_string(preprocessed_image, lang='vie', config=custom_config)
                if isinstance(raw_text, bytes):
                    page_text = raw_text.decode('utf-8')
                elif isinstance(raw_text, str):
                    page_text = raw_text
                else:
                    page_text = str(raw_text)
                
                # OCR outputs raw unicode text, no TCVN3 encoding needed
                self.current_page_is_tcvn3 = False
                
                # Apply Vietnamese-specific OCR error corrections
                page_text = self._fix_ocr_vietnamese(page_text)
                
                parsed_elements = self._parse_text_layout(page_text, page_num, source_file)
                elements.extend(parsed_elements)
            except Exception as page_err:
                logger.warning(
                    f"Error running OCR on page {page_num} of '{source_file}': {page_err}. "
                    "Skipping page."
                )
                continue

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

            if self._is_heading(cleaned):
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

    def _preprocess_for_ocr(self, pil_image, use_binarization: bool = False) -> "Image.Image":
        """
        Applies preprocessing to improve OCR accuracy on complex and decorative fonts.
        For textbook PDFs, standard grayscale (use_binarization=False) works best to preserve
        drop caps and avoid noisy background borders.
        """
        if not use_binarization:
            return pil_image.convert('L')

        try:
            import cv2
            import numpy as np
            
            # Convert PIL image to OpenCV numpy array
            img = np.array(pil_image)
            
            # Convert RGB to Grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
                
            # Light noise reduction using Gaussian Blur (preserves diacritics well)
            denoised = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Binarize with Adaptive Thresholding to resolve faded/special fonts and uneven illumination
            binary = cv2.adaptiveThreshold(
                denoised, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                blockSize=31, 
                C=10
            )
            
            # Convert back to PIL Image
            return Image.fromarray(binary)
        except Exception as preprocess_err:
            logger.warning(f"OpenCV preprocessing failed: {preprocess_err}. Falling back to standard grayscale.")
            return pil_image.convert('L')

    def _fix_ocr_vietnamese(self, text: str) -> str:
        """
        Fixes common Tesseract OCR misrecognitions for Vietnamese text.
        Primary fix: 'v' misread as 'u' (e.g. 'uăn' -> 'văn').
        Uses word-boundary-aware regex to avoid false positives.
        """
        if not text:
            return text
        patterns = self._get_ocr_patterns()
        for pattern, replacement in patterns:
            text = pattern.sub(replacement, text)
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
        if not line or len(line) > 120:
            return False

        has_letters = any(c.isalpha() for c in line)
        if has_letters:
            # Check if mostly uppercase (at least 75% of letters are uppercase)
            # This handles OCR noise at the end of lines (e.g. "RA-MA BUỘC TỘI va v‹")
            letters = [c for c in line if c.isalpha()]
            upper_letters = [c for c in letters if c.isupper()]
            if len(upper_letters) / len(letters) >= 0.75:
                return True

        if re.match(r'^[IVXLCDM]+\.?\s+', line):
            return True

        vietnamese_caps = "A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶEÈÉẺẼẸÊỀẾỂỄỆIÌÍỈĨỊOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢUÙÚỦŨỤƯỪỨỬỮỰYỲÝỶỸỴ"
        if re.match(rf'^\d+(\.\d+)*\.?\s+[{vietnamese_caps}]', line):
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
