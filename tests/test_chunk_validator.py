import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from core.semantic_chunker import SemanticChunk
from core.chunk_validator import ChunkValidator, ValidationResult, ValidatedChunk


@pytest.fixture
def validator():
    return ChunkValidator()


def create_base_chunk(
    chunk_id="chunk_01",
    title="1. Nhân vật Tràng",
    content="Tràng là một nhân vật ngụ cư nghèo khổ trong tác phẩm Vợ nhặt của nhà văn Kim Lân. Qua nhân vật này, tác giả đã thể hiện tinh thần nhân đạo sâu sắc và khát vọng sống mãnh liệt của con người Việt Nam trước Cách mạng tháng Tám.",
    content_type="prose",
    page_start=5,
    page_end=5,
    section_title="III. Phân tích",
    subsection_title=None,
    parent_section=None,
    tags=None,
    token_count=320,
    char_count=200,
    has_overlap=False,
    overlap_from_chunk=None
):
    if tags is None:
        tags = ["nhan_vat_trang", "tam_ly_nhan_vat"]
    return SemanticChunk(
        chunk_id=chunk_id,
        title=title,
        content=content,
        content_type=content_type,
        page_start=page_start,
        page_end=page_end,
        section_title=section_title,
        subsection_title=subsection_title,
        parent_section=parent_section,
        tags=tags,
        token_count=token_count,
        char_count=char_count,
        has_overlap=has_overlap,
        overlap_from_chunk=overlap_from_chunk
    )


def test_valid_chunk(validator):
    """
    Test standard valid chunk without subsection_title.
    Expected: passed = True, quality_score = 95.0, no errors, no warnings.
    """
    chunk = create_base_chunk()
    validated = validator.validate_chunk(chunk, [chunk])
    
    assert validated.validation.passed is True
    assert validated.validation.quality_score == 95.0
    assert len(validated.validation.errors) == 0
    assert len(validated.validation.warnings) == 0


def test_perfect_valid_chunk(validator):
    """
    Test standard valid chunk with subsection_title.
    Expected: passed = True, quality_score = 100.0, no errors, no warnings.
    """
    chunk = create_base_chunk(subsection_title="1.1 Ngoại hình nhân vật")
    validated = validator.validate_chunk(chunk, [chunk])
    
    assert validated.validation.passed is True
    assert validated.validation.quality_score == 100.0
    assert len(validated.validation.errors) == 0
    assert len(validated.validation.warnings) == 0


def test_rule_1_empty_content(validator):
    """
    Test empty content or content too short (< 20 chars).
    Expected: passed = False, contains 'Empty content' error.
    """
    
    chunk_empty = create_base_chunk(content="", token_count=0)
    validated_empty = validator.validate_chunk(chunk_empty, [chunk_empty])
    assert validated_empty.validation.passed is False
    assert "Empty content" in validated_empty.validation.errors

    
    chunk_ws = create_base_chunk(content="   \n   ", token_count=0)
    validated_ws = validator.validate_chunk(chunk_ws, [chunk_ws])
    assert validated_ws.validation.passed is False
    assert "Empty content" in validated_ws.validation.errors

    
    chunk_short = create_base_chunk(content="Một hai ba bốn.", token_count=5)
    validated_short = validator.validate_chunk(chunk_short, [chunk_short])
    assert validated_short.validation.passed is False
    assert "Empty content" in validated_short.validation.errors


def test_rule_2_3_token_limits(validator):
    """
    Test token counts outside the limits.
    Rule 2: < 50 tokens (Warning)
    Rule 3: > 1200 tokens (Warning)
    """
    
    chunk_low = create_base_chunk(token_count=45)
    validated_low = validator.validate_chunk(chunk_low, [chunk_low])
    assert validated_low.validation.passed is True  
    assert "Token count too low" in validated_low.validation.warnings
    
    assert validated_low.validation.quality_score == 85.0

    
    chunk_high = create_base_chunk(token_count=1300)
    validated_high = validator.validate_chunk(chunk_high, [chunk_high])
    assert validated_high.validation.passed is True  
    assert "Token count too high" in validated_high.validation.warnings
    
    assert validated_high.validation.quality_score == 80.0


def test_rule_4_duplicate_detection(validator):
    """
    Test duplicate detection (>95% similarity).
    """
    chunk1 = create_base_chunk(chunk_id="c1", content="Nội dung kiểm tra trùng lặp cho RAG pipeline.")
    
    chunk2 = create_base_chunk(chunk_id="c2", content="Nội dung kiểm tra trùng lặp cho RAG pipeline. ")
    
    all_chunks = [chunk1, chunk2]
    
    validated1 = validator.validate_chunk(chunk1, all_chunks)
    validated2 = validator.validate_chunk(chunk2, all_chunks)
    
    assert "Duplicate content detected" in validated1.validation.warnings
    assert "Duplicate content detected" in validated2.validation.warnings
    
    
    chunk3 = create_base_chunk(chunk_id="c3", content="Nội dung hoàn toàn khác biệt để kiểm thử hoạt động của validator.")
    validated3 = validator.validate_chunk(chunk3, [chunk1, chunk3])
    assert "Duplicate content detected" not in validated3.validation.warnings


def test_rule_5_overlap_consistency(validator):
    """
    Test overlap flag and ID consistency rules.
    """
    
    chunk1 = create_base_chunk(has_overlap=True, overlap_from_chunk=None)
    validated1 = validator.validate_chunk(chunk1, [chunk1])
    assert "Chunk has overlap flag set to True but missing overlap_from_chunk ID" in validated1.validation.warnings

    
    chunk2 = create_base_chunk(has_overlap=False, overlap_from_chunk="some_id")
    validated2 = validator.validate_chunk(chunk2, [chunk2])
    assert "Chunk has overlap flag set to False but has overlap_from_chunk ID" in validated2.validation.warnings

    
    chunk3 = create_base_chunk(has_overlap=True, overlap_from_chunk="nonexistent_id")
    validated3 = validator.validate_chunk(chunk3, [chunk3])
    assert "Referenced overlap chunk nonexistent_id not found" in validated3.validation.warnings

    
    prev_chunk = create_base_chunk(chunk_id="c1", content="Nội dung của đoạn văn thứ nhất làm tiền đề cho đoạn sau.")
    curr_chunk = create_base_chunk(
        chunk_id="c2",
        content="Nội dung không liên quan\n\nNội dung của đoạn văn thứ hai tiếp nối.",
        has_overlap=True,
        overlap_from_chunk="c1"
    )
    validated4 = validator.validate_chunk(curr_chunk, [prev_chunk, curr_chunk])
    assert any("does not match referenced chunk's content" in w for w in validated4.validation.warnings)

    
    curr_chunk_ok = create_base_chunk(
        chunk_id="c3",
        content="tiền đề cho đoạn sau.\n\nNội dung của đoạn văn thứ hai tiếp nối.",
        has_overlap=True,
        overlap_from_chunk="c1"
    )
    validated5 = validator.validate_chunk(curr_chunk_ok, [prev_chunk, curr_chunk_ok])
    
    assert not any("does not match referenced chunk's content" in w for w in validated5.validation.warnings)


def test_rule_6_invalid_page_range(validator):
    """
    Test page range validation.
    Rule 6: page_start > page_end (Error)
    """
    chunk = create_base_chunk(page_start=10, page_end=5)
    validated = validator.validate_chunk(chunk, [chunk])
    
    assert validated.validation.passed is False
    assert "Invalid page range: page_start > page_end" in validated.validation.errors


def test_rule_7_8_9_missing_metadata(validator):
    """
    Test missing metadata checks.
    Rule 7: Missing title
    Rule 8: Missing section_title
    Rule 9: Missing tags
    """
    
    chunk_title = create_base_chunk(title="")
    val_title = validator.validate_chunk(chunk_title, [chunk_title])
    assert "Missing title" in val_title.validation.warnings

    
    chunk_sect = create_base_chunk(section_title=" ")
    val_sect = validator.validate_chunk(chunk_sect, [chunk_sect])
    assert "Missing section title" in val_sect.validation.warnings

    
    chunk_tags = create_base_chunk(tags=[])
    val_tags = validator.validate_chunk(chunk_tags, [chunk_tags])
    assert "Missing tags" in val_tags.validation.warnings


def test_rule_10_invalid_content_type(validator):
    """
    Test content type validation.
    Rule 10: Invalid content type (Error)
    """
    chunk = create_base_chunk(content_type="unsupported_type")
    validated = validator.validate_chunk(chunk, [chunk])
    
    assert validated.validation.passed is False
    assert "Invalid content type: unsupported_type" in validated.validation.errors


def test_scoring_fail_case_from_prompt(validator):
    """
    Test the specific fail test case from the prompt:
    Input: title = "", content = "", token_count = 5.
    Expected: passed = False, quality_score = 10, errors = ["Empty content"], warnings = ["Missing title", "Token count too low"]
    """
    chunk = create_base_chunk(
        title="",
        content="",
        token_count=5,
        tags=[] 
    )
    
    
    
    
    
    
    
    
    
    chunk = create_base_chunk(
        title="",
        content="",
        token_count=5,
        section_title="Valid Section",
        tags=["some_tag"]
    )
    validated = validator.validate_chunk(chunk, [chunk])
    
    assert validated.validation.passed is False
    assert validated.validation.quality_score == 10.0
    assert "Empty content" in validated.validation.errors
    assert "Missing title" in validated.validation.warnings
    assert "Token count too low" in validated.validation.warnings
