import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set
from difflib import SequenceMatcher
from core.semantic_chunker import SemanticChunk

logger = logging.getLogger("rag-service.chunk-validator")


@dataclass
class ValidationResult:
    """
    Holds the validation details and quality score for a semantic chunk.
    """
    chunk_id: str
    passed: bool
    quality_score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidatedChunk:
    """
    Combines the original SemanticChunk with its ValidationResult.
    """
    chunk: SemanticChunk
    validation: ValidationResult


class ChunkValidator:
    """
    Validates the quality, metadata completeness, overlap, and consistency of SemanticChunks.
    """
    def __init__(self, min_token_limit: int = 50, max_token_limit: int = 1200, duplicate_threshold: float = 0.95):
        self.min_token_limit = min_token_limit
        self.max_token_limit = max_token_limit
        self.duplicate_threshold = duplicate_threshold
        self.valid_content_types = {"prose", "poem", "dialogue", "exercise", "table", "analysis", "summary", "list"}

    def validate_chunk(self, chunk: SemanticChunk, all_chunks: List[SemanticChunk]) -> ValidatedChunk:
        """
        Validates a single SemanticChunk against 10 core rules and calculates its quality score.
        """
        errors = []
        warnings = []
        
        # Rule 1: Empty or short content (Error)
        if not chunk.content or not chunk.content.strip():
            errors.append("Empty content")
        elif len(chunk.content.strip()) < 20:
            errors.append("Empty content")  # Grouping under "Empty content" as per expected fail case
            
        # Rule 2: Min tokens check (Warning)
        if chunk.token_count < self.min_token_limit:
            warnings.append("Token count too low")
            
        # Rule 3: Max tokens check (Warning)
        if chunk.token_count > self.max_token_limit:
            warnings.append("Token count too high")
            
        # Rule 4: Duplicate content check (Warning)
        if self._is_duplicate(chunk, all_chunks):
            warnings.append("Duplicate content detected")
            
        # Rule 5: Overlap validation (Warning)
        self._validate_overlap(chunk, all_chunks, warnings)
        
        # Rule 6: Page range check (Error)
        if chunk.page_start > chunk.page_end:
            errors.append("Invalid page range: page_start > page_end")
            
        # Rule 7: Missing title check (Warning)
        if not chunk.title or not chunk.title.strip():
            warnings.append("Missing title")
            
        # Rule 8: Missing section_title check (Warning)
        if not chunk.section_title or not chunk.section_title.strip():
            warnings.append("Missing section title")
            
        # Rule 9: Missing tags check (Warning)
        if not chunk.tags or len(chunk.tags) == 0:
            warnings.append("Missing tags")
            
        # Rule 10: Content type check (Error)
        if chunk.content_type not in self.valid_content_types:
            errors.append(f"Invalid content type: {chunk.content_type}")
            
        # Calculate quality score
        passed = len(errors) == 0
        quality_score = self._calculate_score(chunk, passed, errors, warnings)
        
        validation_result = ValidationResult(
            chunk_id=chunk.chunk_id,
            passed=passed,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings
        )
        
        return ValidatedChunk(chunk=chunk, validation=validation_result)

    def validate(self, chunks: List[SemanticChunk]) -> List[ValidatedChunk]:
        """
        Validates a list of semantic chunks and returns a list of ValidatedChunks.
        """
        validated_chunks = []
        for chunk in chunks:
            validated_chunks.append(self.validate_chunk(chunk, chunks))
        return validated_chunks

    def _is_duplicate(self, chunk: SemanticChunk, all_chunks: List[SemanticChunk]) -> bool:
        """
        Checks if the chunk content is >95% similar to any other chunk in the list.
        """
        if not chunk.content or len(chunk.content.strip()) < 20:
            return False
            
        for other in all_chunks:
            if other.chunk_id == chunk.chunk_id:
                continue
                
            # Length-based pre-filtering optimization
            len_diff = abs(len(chunk.content) - len(other.content))
            max_len = max(len(chunk.content), len(other.content), 1)
            if len_diff / max_len > (1.0 - self.duplicate_threshold):
                continue
                
            similarity = SequenceMatcher(None, chunk.content, other.content).ratio()
            if similarity > self.duplicate_threshold:
                return True
        return False

    def _validate_overlap(self, chunk: SemanticChunk, all_chunks: List[SemanticChunk], warnings: List[str]):
        """
        Checks for overlap metadata consistency and correctness.
        """
        if chunk.has_overlap:
            if not chunk.overlap_from_chunk:
                warnings.append("Chunk has overlap flag set to True but missing overlap_from_chunk ID")
            else:
                # Check if referenced chunk exists
                ref_chunk = next((c for c in all_chunks if c.chunk_id == chunk.overlap_from_chunk), None)
                if not ref_chunk:
                    warnings.append(f"Referenced overlap chunk {chunk.overlap_from_chunk} not found")
                else:
                    # Normalized comparison of overlap text
                    parts = chunk.content.split("\n\n", 1)
                    if len(parts) >= 2:
                        overlap_part = parts[0].strip()
                        # Clean/normalize both strings to avoid spacing issues
                        overlap_clean = "".join(overlap_part.split())
                        ref_clean = "".join(ref_chunk.content.split())
                        if overlap_clean and overlap_clean not in ref_clean:
                            warnings.append(f"Overlap text from {chunk.overlap_from_chunk} does not match referenced chunk's content")
        else:
            if chunk.overlap_from_chunk:
                warnings.append("Chunk has overlap flag set to False but has overlap_from_chunk ID")

    def _calculate_score(self, chunk: SemanticChunk, passed: bool, errors: List[str], warnings: List[str]) -> float:
        """
        Calculates a quality score from 0 to 100 based on standard deductions.
        """
        score = 100.0
        
        # Deduct for errors
        for error in errors:
            if error == "Empty content" or error == "Content length too short":
                score -= 80.0
            elif error == "Invalid page range: page_start > page_end":
                score -= 40.0
            elif error.startswith("Invalid content type"):
                score -= 30.0
            else:
                score -= 20.0
                
        # Deduct for warnings
        for warning in warnings:
            if warning == "Token count too low":
                if "Empty content" not in errors:
                    score -= 10.0
            elif warning == "Token count too high":
                score -= 15.0
            elif warning == "Duplicate content detected":
                score -= 25.0
            elif "overlap" in warning.lower():
                score -= 10.0
            elif warning == "Missing title":
                score -= 10.0
            elif warning == "Missing section title":
                score -= 10.0
            elif warning == "Missing tags":
                score -= 10.0
                
        # Minor deduction for missing optional metadata only if chunk is otherwise valid
        if passed:
            if not chunk.subsection_title or not chunk.subsection_title.strip():
                score -= 5.0
            
        return float(max(0.0, min(100.0, score)))
