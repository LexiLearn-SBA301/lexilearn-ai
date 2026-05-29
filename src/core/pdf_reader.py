import os
import re
import logging
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

try:
    # pyrefly: ignore [missing-import]
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore

try:
    # pyrefly: ignore [missing-import]
    import PyPDF2
except ImportError:
    PyPDF2 = None  # type: ignore

# Set up logging
logger = logging.getLogger("rag-service.pdf-reader")

# TCVN3 (ABC) to Unicode mapping dictionary
TCVN3_MAP = {
    # Lowercase a block
    'µ': 'à', '¸': 'á', '¶': 'ả', '·': 'ã', '¹': 'ạ',
    # ă block
    '¨': 'ă', '»': 'ằ', '¾': 'ắ', '¼': 'ẳ', '½': 'ẵ', 'Æ': 'ặ',
    # â block
    '©': 'â', 'Ç': 'ầ', 'Ê': 'ấ', 'È': 'ẩ', 'É': 'ẫ', 'Ë': 'ậ',
    # đ
    '®': 'đ',
    # e block
    'Ì': 'è', 'Ð': 'é', 'Î': 'ẻ', 'Ï': 'ẽ', 'Ñ': 'ẹ',
    # ê block
    'ª': 'ê', 'Ò': 'ề', 'Õ': 'ế', 'Ó': 'ể', 'Ô': 'ễ', 'Ö': 'ệ',
    # i block
    '×': 'ì', 'Ý': 'í', 'Ø': 'ỉ', 'Ü': 'ĩ', 'Þ': 'ị',
    # o block
    'ß': 'ò', 'ã': 'ó', 'á': 'ỏ', 'â': 'õ', 'ä': 'ọ',
    # ô block
    '«': 'ô', 'å': 'ồ', 'è': 'ố', 'æ': 'ổ', 'ç': 'ỗ', 'é': 'ộ',
    # ơ block
    '¬': 'ơ', 'ê': 'ờ', 'í': 'ớ', 'ë': 'ở', 'ì': 'ỡ', 'î': 'ợ',
    # u block
    'ó': 'ù', 'ñ': 'ú', 'ò': 'ủ', 'ô': 'ụ', 'õ': 'ũ',
    'ø': 'ứ', 'ö': 'ừ', '÷': 'ử', 'ù': 'ự', 'ú': 'ữ', '−': 'ư',
    # y block
    'û': 'ỳ', 'ü': 'ý', 'þ': 'ỷ', '¡': 'ỹ', '¢': 'ỵ',
    # Uppercase
    '§': 'Đ', '£': 'Ă', '¤': 'Â', '¥': 'Ê', '¦': 'Ô',
    # PDF specific variations / OCR noise
    'μ': 'à',  # Greek mu
    'µ': 'à',  # Micro sign
    '': '-',  # symbol font dash
}


# Unique characters in TCVN3 that are not standard Unicode lowercase Vietnamese vowels/letters
# Helps in distinguishing standard Unicode text from TCVN3 text and preventing false positives
TCVN3_SIGNATURE_CHARS = "µ¸¶·¹¨»¾¼½Æ©ÇÊÈÉË®ÌÐÎÏÑªÒÕÓÔÖ×ÝØÜÞ§£¤¥¦−μ"

@dataclass


class ExtractedElement:
    """
    Represents a structured element extracted from a PDF document page.
    """
    page: int           # Page number in the PDF (1-indexed)
    type: str           # "paragraph" | "table" | "list" | "heading" | "unknown"
    raw_text: str       # Normalized text content preserving Vietnamese Unicode
    source_file: str    # Filename or path of the source PDF document

class PDFReader:
    """
    PDF Reader designed to ingest Vietnamese Literature textbook PDFs.
    Uses pdfplumber as the primary extraction engine with tabular data extraction support,
    and falls back to PyPDF2 if pdfplumber fails.
    """

    # Common headings keywords for Vietnamese high school textbooks
    HEADING_KEYWORDS = {
        "chương", "bài", "phần", "ghi nhớ", "tiểu dẫn", 
        "tác giả", "tác phẩm", "tóm tắt", "luyện tập", 
        "đọc hiểu", "đọc - hiểu", "văn bản", "tri thức ngữ văn"
    }

    def __init__(self) -> None:
        pass

    def read(self, file_path: str) -> List[ExtractedElement]:
        """
        Reads a PDF file and extracts a list of structured elements.
        
        Args:
            file_path: The absolute or relative path to the PDF file.
            
        Returns:
            A list of ExtractedElement instances preserving document order.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        source_file = os.path.basename(file_path)
        logger.info(f"Starting extraction for PDF file: {source_file}")

        # Attempt extraction using pdfplumber (primary)
        try:
            return self._extract_with_pdfplumber(file_path, source_file)
        except Exception as e:
            logger.warning(
                f"Primary extraction with pdfplumber failed for '{source_file}' due to: {e}. "
                "Falling back to PyPDF2."
            )

        # Fallback to PyPDF2
        try:
            return self._extract_with_pypdf(file_path, source_file)
        except Exception as e:
            logger.error(f"Fallback extraction with PyPDF2 also failed for '{source_file}': {e}")
            # Ensure we never crash the ingestion pipeline, return empty list
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
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                try:
                    # 1. Detect and extract tables
                    tables = page.find_tables()
                    table_elements: List[ExtractedElement] = []
                    
                    for table in tables:
                        table_data = table.extract()
                        if not table_data:
                            continue
                        
                        formatted_rows = []
                        for row in table_data:
                            # Clean cells and join with pipe symbol
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

                    # 2. Extract text (excluding table bounding boxes if tables exist)
                    if tables:
                        try:
                            # Filter out character objects that reside inside table bounding boxes
                            filtered_page = page.filter(
                                lambda obj: not (
                                    obj.get("object_type") == "char"
                                    and any(self._is_in_bbox(obj, t.bbox) for t in tables)
                                )
                            )
                            page_text = filtered_page.extract_text()
                        except Exception as filter_err:
                            logger.warning(
                                f"Failed to filter table characters on page {page_num} "
                                f"of {source_file}: {filter_err}. Proceeding with standard text extraction."
                            )
                            page_text = page.extract_text()
                    else:
                        page_text = page.extract_text()

                    # 3. Process extracted text into structured headings, lists, and paragraphs
                    parsed_elements = self._parse_text_layout(page_text, page_num, source_file)
                    
                    # Append all structured text elements first, then tables at the end of the page
                    elements.extend(parsed_elements)
                    elements.extend(table_elements)

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
                    parsed_elements = self._parse_text_layout(page_text, page_num, source_file)
                    elements.extend(parsed_elements)
                except Exception as page_err:
                    logger.warning(
                        f"Error extracting page {page_num} from '{source_file}' using PyPDF2: {page_err}. "
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
            cleaned = self._clean_text(line)
            if not cleaned:
                continue

            # Classify line using heuristics
            if self._is_heading(cleaned):
                # Flush existing paragraph
                if current_para_lines:
                    elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                    current_para_lines = []
                
                elements.append(
                    ExtractedElement(
                        page=page_num,
                        type="heading",
                        raw_text=cleaned,
                        source_file=source_file
                    )
                )
            elif self._is_list_item(cleaned):
                # Flush existing paragraph
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
                # Normal paragraph line: check if we should start a new paragraph or merge
                if not current_para_lines:
                    current_para_lines.append(cleaned)
                else:
                    prev_line = current_para_lines[-1]
                    # If previous line ends with typical sentence endings, start a new paragraph
                    if prev_line and prev_line[-1] in (".", "?", "!", "”", '"'):
                        elements.append(self._build_paragraph_element(current_para_lines, page_num, source_file))
                        current_para_lines = [cleaned]
                    else:
                        # Otherwise merge lines into a single paragraph
                        current_para_lines.append(cleaned)

        # Flush any remaining paragraph at the end of the page
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

    def _tcvn3_to_unicode(self, text: str) -> str:
        """
        Dynamically detects if a text is TCVN3-encoded and decodes it to Unicode.
        Uses signature characters to prevent false positive conversions on standard Unicode.
        """
        if not text:
            return ""
        # Count TCVN3 signature characters
        tcvn3_sig_count = sum(1 for c in text if c in TCVN3_SIGNATURE_CHARS)
        
        # Convert if it contains multiple signature characters, or if it is a short text with at least one signature char
        if tcvn3_sig_count >= 2 or (tcvn3_sig_count >= 1 and len(text) < 30):
            return "".join(TCVN3_MAP.get(c, c) for c in text)
        return text


    def _clean_text(self, text: str) -> str:
        """
        Normalizes spaces and Unicode characters for Vietnamese text.
        Converts legacy TCVN3 encoding to standard Unicode.
        Preserves casing and all accents.
        """
        if not text:
            return ""
        # Convert legacy TCVN3 to Unicode first
        text = self._tcvn3_to_unicode(text)
        # Normalize to Unicode NFC form (standard representation for Vietnamese)
        normalized = unicodedata.normalize("NFC", text)
        # Collapse multiple horizontal whitespaces/tabs into a single space
        collapsed_spaces = re.sub(r'[ \t\r\f\v]+', ' ', normalized)
        return collapsed_spaces.strip()


    def _is_in_bbox(self, char: dict, bbox: tuple) -> bool:
        """
        Returns True if the char obj lies inside the given bounding box (x0, top, x1, bottom).
        Includes a 1-point tolerance margin on edges.
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
        Headings must be short (<= 120 chars) and match typical structured section forms.
        """
        if not line or len(line) > 120:
            return False

        # 1. All-uppercase check (must contain at least some alphabetical characters)
        has_letters = any(c.isalpha() for c in line)
        if has_letters and line.isupper():
            return True

        # 2. Roman Numeral Headers (e.g., "I. Tác giả", "IV. Luyện tập")
        if re.match(r'^[IVXLCDM]+\.?\s+', line):
            return True

        # 3. Numbered sections (e.g., "1. Khái quát", "2.3. Giá trị nghệ thuật")
        # Handles Vietnamese accented capitals starting the title text
        vietnamese_caps = "A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶEÈÉẺẼẸÊỀẾỂỄỆIÌÍỈĨỊOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢUÙÚỦŨỤƯỪỨỬỮỰYỲÝỶỸỴ"
        if re.match(rf'^\d+(\.\d+)*\.?\s+[{vietnamese_caps}]', line):
            return True

        # 4. Keyword matches at start of line (e.g., "Bài 1:", "Chương II:", "Ghi nhớ")
        lower_line = line.lower()
        for kw in self.HEADING_KEYWORDS:
            if lower_line.startswith(kw + " ") or lower_line == kw:
                # To avoid false positives on common words starting normal sentences:
                if kw in {"văn bản", "bài", "chương", "phần"}:
                    after_kw = lower_line[len(kw):].strip()
                    if not after_kw:
                        return True
                    if after_kw.startswith(":") or after_kw.startswith("-"):
                        return True
                    # Must be followed by digit, roman numeral, or specific heading words
                    words = after_kw.split()
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
        Detects standard bullet characters or lowercased index notations.
        """
        if not line:
            return False
        
        # Matches:
        # - Bullets: - • * + – —
        # - Alphabetic/Numeric list markers with dot or parentheses: a), a., 1), 1.
        return bool(re.match(r'^([•\-\*\+\–\—]|\d+[\)\.]|[a-zA-Z][\)\.])\s+', line))

