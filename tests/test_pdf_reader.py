import os
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from core.pdf_reader import PDFReader

@pytest.fixture
def reader():
    return PDFReader()

def test_text_cleaning(reader):
    """
    Test standard text cleaning, spacing collapses, and Vietnamese Unicode normalization.
    """
    
    assert reader._clean_text("   đoạn   văn   này   có   nhiều  khoảng   trống  ") == "đoạn văn này có nhiều khoảng trống"
    
    assert reader._clean_text("Ngữ Văn Lớp 12") == "Ngữ Văn Lớp 12"
    
    assert reader._clean_text("Dòng 1\tDòng 2\r") == "Dòng 1 Dòng 2"
    
    assert reader._clean_text("®Æng ®øc siªu") == "đặng đức siêu"
    assert reader._clean_text("Nhµ xuÊt b¶n Gi¸o dôc") == "Nhà xuất bản Giáo dục"
    assert reader._clean_text("®−îc cuéc sèng cùc nhôc") == "được cuộc sống cực nhục"
    assert reader._clean_text("giμu chÊt th¬ vμ ®Ëm mμu") == "giàu chất thơ và đậm màu"


def test_heading_detection(reader):
    """
    Test heading detection heuristics for Vietnamese literature context.
    """
    
    assert reader._is_heading("PHÂN TÍCH NHÂN VẬT TRÀNG") is True
    assert reader._is_heading("TRÀNG") is True
    
    assert reader._is_heading("I. Tác giả Kim Lân") is True
    assert reader._is_heading("IV. Tổng kết") is True
    
    assert reader._is_heading("1. Tiểu dẫn") is True
    assert reader._is_heading("2.3. Giá trị nhân đạo") is True
    
    assert reader._is_heading("Bài 1: Khái quát văn học") is True
    assert reader._is_heading("Chương II: Văn học trung đại") is True
    assert reader._is_heading("Ghi nhớ") is True
    
    
    assert reader._is_heading("") is False
    assert reader._is_heading("Đây là một đoạn văn bình thường trong tác phẩm của Kim Lân và không phải tiêu đề.") is False
    assert reader._is_heading("12345") is False  

def test_list_item_detection(reader):
    """
    Test list item detection heuristics.
    """
    
    assert reader._is_list_item("- Chi tiết cái đói") is True
    assert reader._is_list_item("• Biện pháp tu từ") is True
    assert reader._is_list_item("* Ý nghĩa nhan đề") is True
    
    assert reader._is_list_item("a) Hoàn cảnh") is True
    assert reader._is_list_item("1) Nhân vật Tràng") is True
    assert reader._is_list_item("b. Phân tích chi tiết") is True
    
    
    assert reader._is_list_item("a là một chữ cái") is False
    assert reader._is_list_item("Một dòng bình thường.") is False

def test_parse_text_layout(reader):
    """
    Test reconstruction of paragraphs from lines (merging lines that don't end in punctuation
    and splitting when headings or lists appear).
    """
    text = (
        "I. Tác giả Kim Lân\n"
        "Kim Lân là nhà văn chuyên viết truyện ngắn.\n"
        "Ông có am hiểu sâu sắc về nông thôn và người nông dân.\n"
        "Một số tác phẩm tiêu biểu:\n"
        "- Vợ nhặt\n"
        "- Làng\n"
        "II. Tác phẩm Vợ nhặt"
    )
    
    elements = reader._parse_text_layout(text, page_num=1, source_file="test.pdf")
    
    
    assert len(elements) == 7
    
    
    assert elements[0].type == "heading"
    assert elements[0].raw_text == "I. Tác giả Kim Lân"
    
    
    assert elements[1].type == "paragraph"
    assert elements[1].raw_text == "Kim Lân là nhà văn chuyên viết truyện ngắn."
    
    
    assert elements[2].type == "paragraph"
    assert elements[2].raw_text == "Ông có am hiểu sâu sắc về nông thôn và người nông dân."
    
    
    assert elements[3].type == "paragraph"
    assert elements[3].raw_text == "Một số tác phẩm tiêu biểu:"
    
    
    assert elements[4].type == "list"
    assert elements[4].raw_text == "- Vợ nhặt"
    assert elements[5].type == "list"
    assert elements[5].raw_text == "- Làng"
    
    
    assert elements[6].type == "heading"
    assert elements[6].raw_text == "II. Tác phẩm Vợ nhặt"

    
    text_merge = (
        "Kim Lân là nhà văn chuyên viết\n"
        "truyện ngắn về nông thôn."
    )
    elements_merge = reader._parse_text_layout(text_merge, page_num=1, source_file="test.pdf")
    assert len(elements_merge) == 1
    assert elements_merge[0].type == "paragraph"
    assert elements_merge[0].raw_text == "Kim Lân là nhà văn chuyên viết truyện ngắn về nông thôn."


@patch("os.path.exists", return_value=True)
@patch("pdfplumber.open")
def test_read_with_pdfplumber_success(mock_open, mock_exists, reader):
    """
    Test successful extraction path with pdfplumber.
    """
    
    mock_pdf = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_pdf
    
    mock_page = MagicMock()
    mock_pdf.pages = [mock_page]
    
    
    mock_table = MagicMock()
    mock_table.bbox = (10, 20, 100, 200)
    mock_table.extract.return_value = [["Cột A", "Cột B"], ["Dữ liệu 1", "Dữ liệu 2"]]
    mock_page.find_tables.return_value = [mock_table]
    
    
    mock_filtered_page = MagicMock()
    mock_filtered_page.extract_text.return_value = "I. Đọc hiểu\nVăn bản này nói về..."
    mock_page.filter.return_value = mock_filtered_page
    
    results = reader.read("dummy.pdf")
    
    
    assert len(results) == 3
    assert results[0].type == "heading"
    assert results[0].raw_text == "I. Đọc hiểu"
    
    assert results[1].type == "paragraph"
    assert results[1].raw_text == "Văn bản này nói về..."
    
    assert results[2].type == "table"
    assert results[2].raw_text == "Cột A | Cột B\nDữ liệu 1 | Dữ liệu 2"
    
    
    for el in results:
        assert el.source_file == "dummy.pdf"

@patch("os.path.exists", return_value=True)
@patch("pdfplumber.open")
@patch("PyPDF2.PdfReader")
def test_read_fallback_to_pypdf(mock_pypdf, mock_pdfplumber_open, mock_exists, reader):
    """
    Test fallback mechanism to PyPDF2 when pdfplumber fails.
    """
    
    mock_pdfplumber_open.side_effect = Exception("pdfplumber corrupted error")
    
    
    mock_reader_instance = MagicMock()
    mock_pypdf.return_value = mock_reader_instance
    
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Nội dung từ PyPDF2."
    mock_reader_instance.pages = [mock_page]
    
    
    with patch("builtins.open", MagicMock()):
        results = reader.read("dummy.pdf")
        
    assert len(results) == 1
    assert results[0].type == "paragraph"
    assert results[0].raw_text == "Nội dung từ PyPDF2."
    assert results[0].source_file == "dummy.pdf"

