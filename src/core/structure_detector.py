import os
import re
import logging
import unicodedata
from dataclasses import dataclass
from typing import List, Optional
from core.pdf_reader import ExtractedElement

logger = logging.getLogger("rag-service.structure-detector")


@dataclass
class DocumentSection:
    """
    Represents a hierarchical section within the document.
    """
    title: str
    level: int
    page_start: int
    page_end: int
    content: List[str]
    parent_title: Optional[str]


class StructureDetector:
    """
    Detects document hierarchy, titles, sections, and subsections
    from a flat list of ExtractedElements.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.roman_pattern = re.compile(r"^[IVXLCDM]+(?:[\.\-\s]\s*|\s+)")
        self.numbered_pattern = re.compile(r"^\d+(\.\d+)*\.?\s+")
        self.letter_pattern = re.compile(r"^[a-z][\)\.]\s+")

        # Load configuration
        if not config_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(current_dir, "..", "config", "structure_detector_config.json"))
        
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        self.generic_keywords = set(config["generic_keywords"])

    def _normalize_for_matching(self, text: str) -> str:
        if not text:
            return ""
        # Remove accents
        normalized = unicodedata.normalize('NFD', text)
        stripped = "".join([c for c in normalized if not unicodedata.combining(c)])
        stripped = stripped.replace('Đ', 'D').replace('đ', 'd')
        # Clean punctuation and extra spaces
        cleaned = re.sub(r'[^a-zA-Z0-9\s-]', ' ', stripped)
        return re.sub(r'\s+', ' ', cleaned).strip().lower()

    def _is_garbage_heading(self, text: str) -> bool:
        if not text:
            return True
        # Reject typical garbled symbols from OCR noise (e.g. ^, %, @, #, $, ~, |)
        if re.search(r'[\^%@#$~|]', text):
            return True
        # If it contains almost no letters (excluding spaces/punctuation/digits)
        letters = "".join(c for c in text if c.isalpha())
        if len(letters) < 2:
            return True
        return False

    def _is_generic_section_heading(self, text: str) -> bool:
        normalized = self._normalize_for_matching(text)
        for kw in self.generic_keywords:
            if normalized == kw or normalized.startswith(kw + " ") or normalized.startswith(kw + " -"):
                return True
        return False

    def _is_roman_heading(self, text: str) -> bool:
        return bool(self.roman_pattern.match(text))

    def _is_numbered_heading(self, text: str) -> bool:
        return bool(self.numbered_pattern.match(text))

    def _is_letter_heading(self, text: str) -> bool:
        return bool(self.letter_pattern.match(text))

    def _is_main_title(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
            
        # Reject publisher/institutional names from being main titles (Level 0)
        lower_text = text.lower()
        if "nhà xuất bản" in lower_text or "bộ giáo dục" in lower_text:
            return False
            
        # Strip parenthesis for uppercase check
        text_no_parens = re.sub(r'\(.*?\)', '', text).strip()
            
        # Handle uppercase ratio instead of strict isupper()
        # Because TCVN3 decoding often results in mixed cases (e.g. TÂY TIếN)
        letters = [c for c in text_no_parens if c.isalpha()]
        if not letters:
            return False
            
        upper_letters = [c for c in letters if c.isupper()]
        upper_ratio = len(upper_letters) / len(letters)
            
        if (
            len(text) <= 150
            and upper_ratio >= 0.75
            and not (
                self._is_roman_heading(text)
                or self._is_numbered_heading(text)
                or self._is_letter_heading(text)
            )
        ):
            return True
        return False

    def _classify_heading_level(self, text: str) -> int:
        text_stripped = text.strip()
        
        if self._is_generic_section_heading(text_stripped):
            return 1
            
        if self._is_roman_heading(text_stripped):
            return 1
        elif self._is_letter_heading(text_stripped):
            return 3
        elif self._is_numbered_heading(text_stripped):
            return 2
        elif self._is_main_title(text_stripped):
            return 0
        else:
            logger.warning(f"Unclassifiable heading: '{text}'. Defaulting to Level 1.")
            return 1

    def _reclassify_numbered_items(self, elements: List[ExtractedElement]) -> List[ExtractedElement]:
        """
        Whitelist rule: numbered_item chỉ là heading khi:
          1. Section cha gần nhất là heading La Mã (I., II., III....)
          2. Bản thân dòng ngắn (< 50 ký tự)
          3. Không chứa dấu ?
        Tất cả trường hợp khác → chuyển thành paragraph.
        """
        result = []
        last_confirmed_heading_type = None  # Track loại heading gần nhất

        for el in elements:
            if el.type == "heading":
                # Ghi nhận loại heading: "roman" hoặc "other"
                if self._is_roman_heading(el.raw_text.strip()):
                    last_confirmed_heading_type = "roman"
                else:
                    last_confirmed_heading_type = "other"
                result.append(el)

            elif el.type == "numbered_item":
                text = el.raw_text.strip() if el.raw_text else ""
                is_heading = (
                    last_confirmed_heading_type == "roman"
                    and len(text) < 50
                    and "?" not in text
                )

                if is_heading:
                    # Promote to heading
                    result.append(ExtractedElement(
                        page=el.page, type="heading",
                        raw_text=el.raw_text, source_file=el.source_file
                    ))
                else:
                    # Demote to paragraph (content)
                    result.append(ExtractedElement(
                        page=el.page, type="paragraph",
                        raw_text=el.raw_text, source_file=el.source_file
                    ))
            else:
                result.append(el)

        return result

    def _get_known_works(self) -> dict:
        if not hasattr(self, "_known_works_map"):
            import json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(current_dir, "..", "config", "ingest_service_config.json"))
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            known_works = list(config.get("work_to_author", {}).keys())
            
            # Create a mapping of normalized work -> original canonical work
            self._known_works_map = {}
            for kw in known_works:
                norm_kw = self._normalize_for_matching(kw)
                self._known_works_map[norm_kw] = kw
                
        return self._known_works_map

    def _promote_known_titles(self, elements: List[ExtractedElement]) -> List[ExtractedElement]:
        """
        Scans all elements (including paragraphs). If an element matches a known
        work title (from work_to_author), it splits the element into a special Level 0 
        heading and leaves the rest as a paragraph.
        
        STRICT RULES to prevent false positives:
        - Headings: always eligible for promotion (startswith match).
        - Paragraphs: only promoted if text is short AND closely matches the title length.
        - Short titles (≤ 3 words): require near-exact match (text length ≤ title + 5 chars).
        """
        known_works_map = self._get_known_works()
        sorted_norm_kws = sorted(known_works_map.keys(), key=len, reverse=True)
        
        promoted = []
        for el in elements:
            if not el.raw_text or len(el.raw_text.strip()) == 0:
                promoted.append(el)
                continue
                
            norm_text = self._normalize_for_matching(el.raw_text)
            matched_kw = None
            
            for norm_kw in sorted_norm_kws:
                if not norm_text.startswith(norm_kw):
                    continue
                
                kw_word_count = len(norm_kw.split())
                text_len = len(norm_text)
                kw_len = len(norm_kw)
                overshoot = text_len - kw_len
                
                if el.type == "heading":
                    # Headings: allow startswith match but still guard against
                    # extremely short titles matching long headings
                    if kw_word_count <= 3 and overshoot > 10:
                        continue  # e.g. heading "Nhàn đàm về văn học" should NOT match "nhàn"
                    matched_kw = known_works_map[norm_kw]
                    break
                else:
                    # Paragraphs: very strict rules
                    # 1. Text must be short (< 80 chars normalized)
                    if text_len > 80:
                        continue
                    # 2. Short titles (≤ 3 words like "sóng", "nhàn", "thuốc", "từ ấy"):
                    #    require near-exact match (overshoot ≤ 5 chars)
                    if kw_word_count <= 3 and overshoot > 5:
                        continue
                    # 3. Longer titles: allow some slack (overshoot ≤ 20 chars)
                    if overshoot > 20:
                        continue
                    matched_kw = known_works_map[norm_kw]
                    break
                    
            if matched_kw:
                # We found a known work! Force it as a main_title_heading
                # Use the canonical matched_kw to guarantee a 100% match in IngestService
                promoted.append(ExtractedElement(
                    page=el.page,
                    type="main_title_heading",
                    raw_text=matched_kw.upper(),
                    source_file=el.source_file
                ))
                
                # Put the entire original text as a paragraph to ensure no data is lost
                promoted.append(ExtractedElement(
                    page=el.page,
                    type="paragraph",
                    raw_text=el.raw_text,
                    source_file=el.source_file
                ))
            else:
                promoted.append(el)
                
        return promoted

    def detect(self, elements: List[ExtractedElement]) -> List[DocumentSection]:
        """
        Converts flat ExtractedElements into hierarchical DocumentSections.
        """
        if not elements:
            return []

        # NEW: Reclassify numbered_items based on whitelist rule
        elements = self._reclassify_numbered_items(elements)
        
        # NEW: Force promote known titles (even if hidden in paragraphs)
        elements = self._promote_known_titles(elements)

        # Pre-process: merge consecutive heading elements that represent a single split heading
        processed_elements = []
        i = 0
        n = len(elements)
        while i < n:
            el = elements[i]
            if el.type in ["heading", "main_title_heading"] and el.raw_text and el.raw_text.strip():
                if el.type == "heading" and self._is_garbage_heading(el.raw_text):
                    i += 1
                    continue
                # Check if next elements are heading continuations on the same page
                merged_text = el.raw_text.strip()
                j = i + 1
                while j < n:
                    next_el = elements[j]
                    if not next_el.raw_text or not next_el.raw_text.strip():
                        j += 1
                        continue
                    
                    if next_el.type == "heading" and next_el.page == el.page and el.type != "main_title_heading":
                        next_stripped = next_el.raw_text.strip()
                        is_continuation = not (
                            self._is_roman_heading(next_stripped)
                            or self._is_numbered_heading(next_stripped)
                            or self._is_letter_heading(next_stripped)
                            or self._is_generic_section_heading(next_stripped)
                        )
                        if is_continuation:
                            merged_text += " " + next_stripped
                            j += 1
                            continue
                    break
                
                if j > i + 1:
                    el = ExtractedElement(
                        page=el.page,
                        type=el.type,
                        raw_text=merged_text,
                        source_file=el.source_file
                    )
                    i = j - 1
            
            processed_elements.append(el)
            i += 1
        
        elements = processed_elements

        sections: List[DocumentSection] = []
        section_stack: List[DocumentSection] = []

        for element in elements:
            if not element.raw_text or not element.raw_text.strip():
                logger.debug(f"Skipping element without raw_text on page {element.page}")
                continue

            text = element.raw_text.strip()

            if element.type in ["heading", "main_title_heading"]:
                if element.type == "main_title_heading":
                    level = 0
                else:
                    level = self._classify_heading_level(text)

                parent_title: Optional[str] = None
                for section in reversed(section_stack):
                    if section.level < level:
                        parent_title = section.title
                        break

                new_section = DocumentSection(
                    title=text,
                    level=level,
                    page_start=element.page,
                    page_end=element.page,
                    content=[],
                    parent_title=parent_title
                )

                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()

                section_stack.append(new_section)
                sections.append(new_section)

            else:
                if not sections:
                    default_section = DocumentSection(
                        title="Untitled",
                        level=0,
                        page_start=element.page,
                        page_end=element.page,
                        content=[],
                        parent_title=None
                    )
                    sections.append(default_section)
                    section_stack.append(default_section)

                active_section = sections[-1]
                active_section.content.append(text)
                active_section.page_end = max(active_section.page_end, element.page)

        return sections
