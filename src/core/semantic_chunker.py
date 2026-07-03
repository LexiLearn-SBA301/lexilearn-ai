import re
import logging
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
from core.structure_detector import DocumentSection

logger = logging.getLogger("rag-service.semantic-chunker")


@dataclass
class SemanticChunk:
    """
    Represents a semantically cohesive chunk of a document.
    """
    chunk_id: str
    title: str
    content: str
    content_type: str
    page_start: int
    page_end: int
    section_title: str
    subsection_title: Optional[str]
    parent_section: Optional[str]
    tags: List[str]
    token_count: int
    char_count: int
    has_overlap: bool
    overlap_from_chunk: Optional[str]
    # Metadata extracted by AI Analyzer
    ten_tac_pham: Optional[str] = None
    tac_gia: Optional[str] = None
    is_biography: bool = False
    nam_sang_tac: Optional[int] = None

class SemanticChunker:
    """
    Chunks DocumentSections into SemanticChunks optimized for retrieval.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the SemanticChunker, loading configuration from config_path or a default location.
        """
        import os
        import json

        # Load configuration
        if not config_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(current_dir, "..", "config", "chunker_config.json"))

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        logger.info(f"Loaded semantic chunker configuration from {config_path}")

        # Assign lists and patterns directly from config
        self.tag_patterns = self.config["tag_patterns"]
        self.known_characters = self.config["known_characters"]
        self.excluded_characters = self.config["excluded_characters"]
        self.evidence_prefixes = self.config["evidence_prefixes"]
        self.split_markers = self.config["split_markers"]
        self.split_topics = self.config["split_topics"]
        self.analysis_keywords = self.config["analysis_keywords"]
        self._character_cache = {}

    def _slugify(self, text: str) -> str:
        """
        Converts Vietnamese text to a clean ASCII-safe slug.
        """
        if not text:
            return "chunk"
        normalized = unicodedata.normalize('NFD', text)
        ascii_text = "".join([c for c in normalized if not unicodedata.combining(c)])
        ascii_text = ascii_text.replace('Đ', 'D').replace('đ', 'd')
        ascii_text = ascii_text.lower()
        ascii_text = re.sub(r'[^a-z0-9]+', '_', ascii_text)
        slug = re.sub(r'_+', '_', ascii_text).strip('_')
        return slug if slug else "chunk"

    def _generate_chunk_id(self, title: str, index: int) -> str:
        """
        Generates a unique chunk ID using a slug and sequential index.
        """
        slug = self._slugify(title)
        return f"{slug}_{index:03d}"

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimates the token count of a text using underthesea word tokenization.
        """
        if not text:
            return 0
        try:
            from underthesea import word_tokenize
            return len(word_tokenize(text))
        except Exception as e:
            logger.warning(f"Underthesea word_tokenize failed: {e}. Falling back to space split.")
            return len(text.split())

    def _get_characters(self, text: str) -> Set[str]:
        """
        Extracts names of characters from the text using a hybrid approach of
        known list lookup, syntactic heuristics, and Underthesea NER models.
        Uses a local cache to avoid redundant expensive NER model calls.
        """
        if not text:
            return set()

        if not hasattr(self, "_character_cache"):
            self._character_cache = {}

        if text in self._character_cache:
            return self._character_cache[text]

        chars = set()

        # 1. Matches from the known_characters list
        for char in self.known_characters:
            if re.search(rf"\b{re.escape(char)}\b", text):
                chars.add(char)

        # 2. Extract using syntactic heuristics (e.g., 'nhân vật Mị')
        char_after_phrases = [
            r"nhân vật\s+([A-ZĐ][a-zA-Zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ\s]+)",
            r"hình tượng\s+([A-ZĐ][a-zA-Zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ\s]+)"
        ]
        for pattern in char_after_phrases:
            for match in re.finditer(pattern, text):
                raw_name = match.group(1).strip()
                name_words = []
                for w in raw_name.split():
                    if w and (w[0].isupper() or w[0] == 'Đ'):
                        name_words.append(w)
                    else:
                        break
                if name_words:
                    char_name = " ".join(name_words)
                    if char_name.lower() not in self.excluded_characters:
                        chars.add(char_name)

        # 3. Dynamic Named Entity Recognition (NER) via Underthesea
        try:
            from underthesea import ner
            entities = ner(text)
            current_name = []
            for word, pos, chunk, ent in entities:
                if ent == "B-PER":
                    if current_name:
                        chars.add(" ".join(current_name).replace("_", " "))
                    current_name = [word]
                elif ent == "I-PER":
                    current_name.append(word)
                else:
                    if current_name:
                        chars.add(" ".join(current_name).replace("_", " "))
                        current_name = []
            if current_name:
                chars.add(" ".join(current_name).replace("_", " "))
        except Exception as e:
            logger.warning(f"Underthesea NER failed: {e}. Falling back to heuristics.")

        # Clean, filter, and normalize names
        cleaned_chars = set()
        for name in chars:
            cleaned_name = name.replace("_", " ").strip()
            # Must contain letters, start with uppercase, and not be in excluded list
            if cleaned_name and cleaned_name.lower() not in self.excluded_characters:
                words = cleaned_name.split()
                if all(w[0].isupper() or w[0] == 'Đ' or (len(w) > 1 and w[0] == '(') for w in words if w):
                    cleaned_chars.add(cleaned_name)

        self._character_cache[text] = cleaned_chars
        return cleaned_chars

    def _should_split(self, current_paragraphs: List[str], next_paragraph: str) -> bool:
        """
        Decides whether to split and start a new chunk before the next paragraph.
        """
        if not current_paragraphs:
            return False
            
        current_text = "\n\n".join(current_paragraphs)
        current_tokens = self._estimate_token_count(current_text)
        if current_tokens > 800:
            return True
            
        last_paragraph = current_paragraphs[-1].strip()
        if last_paragraph:
            last_char = last_paragraph[-1]
            if last_char in ('"', '”', '»', ')'):
                return False
                
        next_stripped = next_paragraph.strip()
        next_lower = next_stripped.lower()
        if any(next_lower.startswith(prefix) for prefix in self.evidence_prefixes):
            return False
            
        if len(last_paragraph) < 50 and len(next_stripped) < 50:
            return False
            
        if last_paragraph.startswith(("-", "–", "—")) and next_stripped.startswith(("-", "–", "—")):
            return False
            
        # Optimize by getting characters of current paragraphs individually to leverage caching
        current_chars = set()
        for p in current_paragraphs:
            current_chars.update(self._get_characters(p))
            
        next_chars = self._get_characters(next_paragraph)
        if current_chars and next_chars and not next_chars.issubset(current_chars):
            return True
            
        if any(next_lower.startswith(m) for m in self.split_markers):
            return True
            
        current_topic = None
        for t in self.split_topics:
            if t in current_text.lower():
                current_topic = t
                break
        next_topic = None
        for t in self.split_topics:
            if t in next_lower:
                next_topic = t
                break
        if current_topic is not None and next_topic is not None and current_topic != next_topic:
            return True
            
        return False

    def _detect_content_type(self, title: str, content: str) -> str:
        """
        Detects the content type of a chunk based on heuristics.
        """
        title_lower = title.lower()
        if "luyện tập" in title_lower or "bài tập" in title_lower:
            return "exercise"
        if "ghi nhớ" in title_lower or "tổng kết" in title_lower or "tóm tắt" in title_lower:
            return "summary"
            
        if "|" in content:
            return "table"
            
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if not lines:
            return "prose"
            
        # Check for list (bullet points or numbered list on at least 3 lines and >= 50% of lines)
        list_patterns = [r'^[-–—•+*]\s+', r'^\d+[\.\)]\s+', r'^[a-z][\.\)]\s+']
        list_count = sum(1 for l in lines if any(re.match(pat, l, re.IGNORECASE) for pat in list_patterns))
        if len(lines) >= 3 and list_count / len(lines) >= 0.5:
            return "list"

        if len(lines) >= 3:
            avg_len = sum(len(l) for l in lines) / len(lines)
            if avg_len < 45:
                return "poem"
                
        # Check for dialogue (at least 2 lines and >= 30% start with dash/bullet, or regex matching conversation)
        dialogue_count = sum(1 for l in lines if l.startswith(("-", "–", "—")))
        if len(lines) >= 2 and dialogue_count / len(lines) >= 0.3:
            return "dialogue"
            
        dialogue_pattern = r'(:?\s*["“][^"”]+["”]\s*(?:nói|hỏi|thưa|đáp|kêu|bảo|trả lời))|(?:(?:nói|hỏi|thưa|đáp|kêu|bảo|trả lời)\s*:\s*["“])'
        if re.search(dialogue_pattern, content, re.IGNORECASE):
            return "dialogue"
            
        if any(kw in title_lower for kw in self.analysis_keywords):
            return "analysis"
        if re.search(r'(phân tích|giá trị nghệ thuật|giá trị hiện thực|giá trị nhân đạo|nét đặc sắc)', content, re.IGNORECASE):
            return "analysis"
            
        return "prose"

    def _generate_tags(self, title: str, content: str) -> List[str]:
        """
        Generates semantic tags using keyword matching and character extraction.
        """
        tags = set()
        title_content = f"{title}\n\n{content}".lower()

        for tag, keywords in self.tag_patterns.items():
            for kw in keywords:
                if kw in title_content:
                    tags.add(tag)
                    break

        # Extract characters using our updated _get_characters method which uses underthesea NER
        detected_chars = self._get_characters(f"{title}\n\n{content}")
        for char in detected_chars:
            tags.add(f"nhan_vat_{self._slugify(char)}")
            tags.add("nhan_vat")

        return sorted(list(tags))

    def _generate_overlap(self, prev_chunk: SemanticChunk) -> str:
        """
        Retrieves suffix content from the previous chunk for contextual overlap.
        """
        if prev_chunk.content_type in ("exercise", "table"):
            return ""
            
        words = prev_chunk.content.split()
        if len(words) < 50:
            return ""
            
        overlap_words = words[-80:]
        return " ".join(overlap_words)

    def chunk(self, sections: List[DocumentSection]) -> List[SemanticChunk]:
        """
        Main function to chunk hierarchical DocumentSections into SemanticChunks.
        """
        if not sections:
            return []
            
        chunks: List[SemanticChunk] = []
        slug_counters: Dict[str, int] = {}
        
        for section in sections:
            if section.level in (0, 1):
                section_title = section.title
                subsection_title = None
            else:
                section_title = section.parent_title if section.parent_title else "Untitled"
                subsection_title = section.title
                
            parent_section = section.parent_title
            
            section_slug = self._slugify(section.title)
            if section_slug not in slug_counters:
                slug_counters[section_slug] = 0
                
            if not section.content:
                slug_counters[section_slug] += 1
                chunk_id = self._generate_chunk_id(section.title, slug_counters[section_slug])
                content_type = self._detect_content_type(section.title, "")
                tags = self._generate_tags(section.title, "")
                
                empty_chunk = SemanticChunk(
                    chunk_id=chunk_id,
                    title=section.title,
                    content="",
                    content_type=content_type,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    section_title=section_title,
                    subsection_title=subsection_title,
                    parent_section=parent_section,
                    tags=tags,
                    token_count=0,
                    char_count=0,
                    has_overlap=False,
                    overlap_from_chunk=None
                )
                chunks.append(empty_chunk)
                continue
                
            current_group: List[str] = []
            section_chunks: List[SemanticChunk] = []
            
            for paragraph in section.content:
                if not paragraph or not paragraph.strip():
                    logger.debug(f"Skipping empty paragraph in section '{section.title}'")
                    continue
                    
                p_text = paragraph.strip()
                
                if self._should_split(current_group, p_text):
                    slug_counters[section_slug] += 1
                    chunk_id = self._generate_chunk_id(section.title, slug_counters[section_slug])
                    
                    raw_content = "\n\n".join(current_group)
                    
                    overlap_text = ""
                    has_overlap = False
                    overlap_from_chunk = None
                    if section_chunks:
                        prev_chunk = section_chunks[-1]
                        overlap_text = self._generate_overlap(prev_chunk)
                        if overlap_text:
                            has_overlap = True
                            overlap_from_chunk = prev_chunk.chunk_id
                            
                    final_content = f"{overlap_text}\n\n{raw_content}" if overlap_text else raw_content
                    
                    content_type = self._detect_content_type(section.title, final_content)
                    tags = self._generate_tags(section.title, final_content)
                    token_count = self._estimate_token_count(final_content)
                    char_count = len(final_content)
                    
                    new_chunk = SemanticChunk(
                        chunk_id=chunk_id,
                        title=section.title,
                        content=final_content,
                        content_type=content_type,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        section_title=section_title,
                        subsection_title=subsection_title,
                        parent_section=parent_section,
                        tags=tags,
                        token_count=token_count,
                        char_count=char_count,
                        has_overlap=has_overlap,
                        overlap_from_chunk=overlap_from_chunk
                    )
                    section_chunks.append(new_chunk)
                    chunks.append(new_chunk)
                    
                    current_group = [p_text]
                else:
                    current_group.append(p_text)
                    
            if current_group:
                slug_counters[section_slug] += 1
                chunk_id = self._generate_chunk_id(section.title, slug_counters[section_slug])
                
                raw_content = "\n\n".join(current_group)
                
                overlap_text = ""
                has_overlap = False
                overlap_from_chunk = None
                if section_chunks:
                    prev_chunk = section_chunks[-1]
                    overlap_text = self._generate_overlap(prev_chunk)
                    if overlap_text:
                        has_overlap = True
                        overlap_from_chunk = prev_chunk.chunk_id
                        
                final_content = f"{overlap_text}\n\n{raw_content}" if overlap_text else raw_content
                
                content_type = self._detect_content_type(section.title, final_content)
                tags = self._generate_tags(section.title, final_content)
                token_count = self._estimate_token_count(final_content)
                char_count = len(final_content)
                
                new_chunk = SemanticChunk(
                    chunk_id=chunk_id,
                    title=section.title,
                    content=final_content,
                    content_type=content_type,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    section_title=section_title,
                    subsection_title=subsection_title,
                    parent_section=parent_section,
                    tags=tags,
                    token_count=token_count,
                    char_count=char_count,
                    has_overlap=has_overlap,
                    overlap_from_chunk=overlap_from_chunk
                )
                chunks.append(new_chunk)
                
        return chunks
