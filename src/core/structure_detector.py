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
        self.roman_pattern = re.compile(r"^[IVXLCDM]+\.?\s+")
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
        if (
            len(text) <= 80
            and text.isupper()
            and any(c.isalpha() for c in text)
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

    def detect(self, elements: List[ExtractedElement]) -> List[DocumentSection]:
        """
        Converts flat ExtractedElements into hierarchical DocumentSections.
        """
        if not elements:
            return []

        # Pre-process: merge consecutive heading elements that represent a single split heading
        processed_elements = []
        i = 0
        n = len(elements)
        while i < n:
            el = elements[i]
            if el.type == "heading" and el.raw_text and el.raw_text.strip():
                if self._is_garbage_heading(el.raw_text):
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
                    
                    if next_el.type == "heading" and next_el.page == el.page:
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
                        type="heading",
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

            if element.type == "heading":
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
