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
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        source_file = os.path.basename(file_path)
        logger.info(f"Starting extraction for PDF file: {source_file}")

        try:
            return self._extract_with_pdfplumber(file_path, source_file)
        except Exception as e:
            logger.warning(
                f"Primary extraction with pdfplumber failed for '{source_file}' due to: {e}. "
                "Falling back to PyPDF2."
            )

        try:
            return self._extract_with_pypdf(file_path, source_file)
        except Exception as e:
            logger.error(f"Fallback extraction with PyPDF2 also failed for '{source_file}': {e}")
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

            if self._is_heading(cleaned):
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

    def _clean_text(self, text: str) -> str:
        """
        Normalizes spaces and Unicode characters for Vietnamese text.
        """
        if not text:
            return ""
        force_tcvn3 = getattr(self, "current_page_is_tcvn3", False)
        text = self._tcvn3_to_unicode(text, force=force_tcvn3)
        normalized = unicodedata.normalize("NFC", text)
        collapsed_spaces = re.sub(r'[ \t\r\f\v]+', ' ', normalized)
        return collapsed_spaces.strip()

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
        if has_letters and line.isupper():
            return True

        if re.match(r'^[IVXLCDM]+\.?\s+', line):
            return True

        vietnamese_caps = "A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶEÈÉẺẼẸÊỀẾỂỄỆIÌÍỈĨỊOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢUÙÚỦŨỤƯỪỨỬỮỰYỲÝỶỸỴ"
        if re.match(rf'^\d+(\.\d+)*\.?\s+[{vietnamese_caps}]', line):
            return True

        lower_line = line.lower()
        for kw in self.HEADING_KEYWORDS:
            if lower_line.startswith(kw + " ") or lower_line == kw:
                if kw in {"văn bản", "bài", "chương", "phần"}:
                    after_kw = lower_line[len(kw):].strip()
                    if not after_kw:
                        return True
                    if after_kw.startswith(":") or after_kw.startswith("-"):
                        return True
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
        """
        if not line:
            return False
        
        return bool(re.match(r'^([•\-\*\+\–\—]|\d+[\)\.]|[a-zA-Z][\)\.])\s+', line))
