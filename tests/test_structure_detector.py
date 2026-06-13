import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from core.pdf_reader import ExtractedElement
from core.structure_detector import StructureDetector, DocumentSection


@pytest.fixture
def detector():
    return StructureDetector()


def test_is_roman_heading(detector):
    """
    Test Roman numeral heading classification (Level 1).
    """
    assert detector._is_roman_heading("I. TÁC GIẢ") is True
    assert detector._is_roman_heading("IV. TỔNG KẾT") is True
    assert detector._is_roman_heading("XII. PHẦN MỞ RỘNG") is True
    
    assert detector._is_roman_heading("i. tác giả") is False
    assert detector._is_roman_heading("a. tác giả") is False
    assert detector._is_roman_heading("1. tác giả") is False


def test_is_numbered_heading(detector):
    """
    Test numbered heading classification (Level 2).
    """
    assert detector._is_numbered_heading("1. Nhân vật Tràng") is True
    assert detector._is_numbered_heading("2.3. Giá trị nhân đạo") is True
    assert detector._is_numbered_heading("10. Luyện tập") is True
    assert detector._is_numbered_heading("1.1.2. Phân tích") is True
    assert detector._is_numbered_heading("I. TÁC GIẢ") is False
    assert detector._is_numbered_heading("a) Hoàn cảnh") is False


def test_is_letter_heading(detector):
    """
    Test letter heading classification (Level 3).
    """
    assert detector._is_letter_heading("a) Hoàn cảnh xuất hiện") is True
    assert detector._is_letter_heading("b. Diễn biến tâm lý") is True
    
    assert detector._is_letter_heading("A) Hoàn cảnh") is False
    assert detector._is_letter_heading("I. Tác giả") is False
    assert detector._is_letter_heading("1. Tác giả") is False


def test_is_main_title(detector):
    """
    Test main title classification (Level 0).
    """
    assert detector._is_main_title("VỢ NHẶT") is True
    assert detector._is_main_title("TÂY TIẾN") is True
    assert detector._is_main_title("CHÍ PHÈO") is True
    
    assert detector._is_main_title("VỢ NHẶT " * 20) is False
    
    assert detector._is_main_title("Vợ Nhặt") is False
    
    assert detector._is_main_title("I. VỢ NHẶT") is False
    assert detector._is_main_title("1. VỢ NHẶT") is False
    assert detector._is_main_title("a) VỢ NHẶT") is False
    
    assert detector._is_main_title("123456") is False


def test_classify_heading_level(detector):
    """
    Test full heading level classification logic.
    """
    assert detector._classify_heading_level("VỢ NHẶT") == 0
    assert detector._classify_heading_level("I. TÁC GIẢ") == 1
    assert detector._classify_heading_level("1. Nhân vật Tràng") == 2
    assert detector._classify_heading_level("a) Hoàn cảnh") == 3
    
    assert detector._classify_heading_level("ngẫu nhiên một tiêu đề") == 1


def test_empty_input(detector):
    """
    Test that empty inputs return an empty list of sections.
    """
    assert detector.detect([]) == []


def test_no_headings(detector):
    """
    Test input containing only non-heading elements.
    """
    elements = [
        ExtractedElement(page=1, type="paragraph", raw_text="Dòng văn bản thứ nhất", source_file="test.pdf"),
        ExtractedElement(page=2, type="paragraph", raw_text="Dòng văn bản thứ hai", source_file="test.pdf")
    ]
    sections = detector.detect(elements)
    assert len(sections) == 1
    s = sections[0]
    assert s.title == "Untitled"
    assert s.level == 0
    assert s.page_start == 1
    assert s.page_end == 2
    assert s.content == ["Dòng văn bản thứ nhất", "Dòng văn bản thứ hai"]
    assert s.parent_title is None


def test_content_assignment(detector):
    """
    Test that paragraphs are correctly assigned to their respective headings and pages are tracked.
    """
    elements = [
        ExtractedElement(page=1, type="heading", raw_text="VỢ NHẶT", source_file="test.pdf"),
        ExtractedElement(page=1, type="paragraph", raw_text="Mở đầu tác phẩm...", source_file="test.pdf"),
        ExtractedElement(page=2, type="heading", raw_text="I. TÁC GIẢ", source_file="test.pdf"),
        ExtractedElement(page=2, type="paragraph", raw_text="Kim Lân là nhà văn...", source_file="test.pdf"),
        ExtractedElement(page=3, type="paragraph", raw_text="Ông viết nhiều về nông thôn...", source_file="test.pdf"),
    ]
    sections = detector.detect(elements)
    assert len(sections) == 2
    
    assert sections[0].title == "VỢ NHẶT"
    assert sections[0].level == 0
    assert sections[0].page_start == 1
    assert sections[0].page_end == 1
    assert sections[0].content == ["Mở đầu tác phẩm..."]
    assert sections[0].parent_title is None

    assert sections[1].title == "I. TÁC GIẢ"
    assert sections[1].level == 1
    assert sections[1].page_start == 2
    assert sections[1].page_end == 3
    assert sections[1].content == ["Kim Lân là nhà văn...", "Ông viết nhiều về nông thôn..."]
    assert sections[1].parent_title == "VỢ NHẶT"


def test_detect_full_hierarchy(detector):
    """
    Test detection of a multi-level hierarchy, tracking parents and levels.
    """
    elements = [
        ExtractedElement(page=12, type="heading", raw_text="VỢ NHẶT", source_file="test.pdf"),
        ExtractedElement(page=12, type="paragraph", raw_text="Đoạn mở đầu", source_file="test.pdf"),
        
        ExtractedElement(page=12, type="heading", raw_text="I. TÁC GIẢ", source_file="test.pdf"),
        ExtractedElement(page=12, type="heading", raw_text="1. Tiểu sử", source_file="test.pdf"),
        ExtractedElement(page=13, type="paragraph", raw_text="Kim Lân...", source_file="test.pdf"),
        
        ExtractedElement(page=13, type="heading", raw_text="II. TÁC PHẨM", source_file="test.pdf"),
        ExtractedElement(page=13, type="heading", raw_text="1. Hoàn cảnh sáng tác", source_file="test.pdf"),
        ExtractedElement(page=13, type="heading", raw_text="a) Bối cảnh lịch sử", source_file="test.pdf"),
        ExtractedElement(page=14, type="paragraph", raw_text="Năm 1945 đói khủng khiếp...", source_file="test.pdf"),
    ]
    
    sections = detector.detect(elements)
    assert len(sections) == 6
    
    
    assert sections[0].title == "VỢ NHẶT"
    assert sections[0].level == 0
    assert sections[0].parent_title is None
    assert sections[0].page_start == 12
    assert sections[0].page_end == 12
    assert sections[0].content == ["Đoạn mở đầu"]

    
    assert sections[1].title == "I. TÁC GIẢ"
    assert sections[1].level == 1
    assert sections[1].parent_title == "VỢ NHẶT"
    
    
    assert sections[2].title == "1. Tiểu sử"
    assert sections[2].level == 2
    assert sections[2].parent_title == "I. TÁC GIẢ"
    assert sections[2].page_start == 12
    assert sections[2].page_end == 13
    assert sections[2].content == ["Kim Lân..."]

    
    assert sections[3].title == "II. TÁC PHẨM"
    assert sections[3].level == 1
    assert sections[3].parent_title == "VỢ NHẶT"

    
    assert sections[4].title == "1. Hoàn cảnh sáng tác"
    assert sections[4].level == 2
    assert sections[4].parent_title == "II. TÁC PHẨM"

    
    assert sections[5].title == "a) Bối cảnh lịch sử"
    assert sections[5].level == 3
    assert sections[5].parent_title == "1. Hoàn cảnh sáng tác"
    assert sections[5].page_start == 13
    assert sections[5].page_end == 14
    assert sections[5].content == ["Năm 1945 đói khủng khiếp..."]


def test_detect_split_headings(detector):
    """
    Test that consecutive heading elements representing a single split heading are merged correctly.
    """
    elements = [
        ExtractedElement(page=40, type="heading", raw_text="TRUYỆN AN DƯƠNG VƯƠNG", source_file="test.pdf"),
        ExtractedElement(page=40, type="heading", raw_text="VÀ MỊ CHÂU - TRỌNG THỦY", source_file="test.pdf"),
        ExtractedElement(page=40, type="paragraph", raw_text="(Truyền thuyết)", source_file="test.pdf"),
        ExtractedElement(page=40, type="heading", raw_text="KẾT QUẢ CẦN ĐẠT", source_file="test.pdf"),
        ExtractedElement(page=40, type="heading", raw_text="TIỂU DẪN", source_file="test.pdf"),
    ]
    sections = detector.detect(elements)
    
    assert len(sections) == 3
    assert sections[0].title == "TRUYỆN AN DƯƠNG VƯƠNG VÀ MỊ CHÂU - TRỌNG THỦY"
    assert sections[0].level == 0
    assert sections[0].content == ["(Truyền thuyết)"]
    
    assert sections[1].title == "KẾT QUẢ CẦN ĐẠT"
    assert sections[1].level == 1
    assert sections[1].parent_title == "TRUYỆN AN DƯƠNG VƯƠNG VÀ MỊ CHÂU - TRỌNG THỦY"
    
    assert sections[2].title == "TIỂU DẪN"
    assert sections[2].level == 1
    assert sections[2].parent_title == "TRUYỆN AN DƯƠNG VƯƠNG VÀ MỊ CHÂU - TRỌNG THỦY"
