import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from core.structure_detector import DocumentSection
from core.semantic_chunker import SemanticChunker, SemanticChunk


@pytest.fixture
def chunker():
    return SemanticChunker()


def test_empty_input(chunker):
    """
    Test empty input returns an empty list.
    """
    assert chunker.chunk([]) == []


def test_single_section_single_chunk(chunker):
    """
    Test that a section with a few related paragraphs is grouped into a single chunk.
    """
    sections = [
        DocumentSection(
            title="I. TÁC GIẢ",
            level=1,
            page_start=5,
            page_end=5,
            content=[
                "Kim Lân là một nhà văn xuất sắc của nền văn học Việt Nam hiện đại.",
                "Ông có vốn sống phong phú và sâu sắc về nông thôn và người nông dân.",
                "Tác phẩm của ông luôn tràn đầy tinh thần nhân đạo."
            ],
            parent_title=None
        )
    ]
    chunks = chunker.chunk(sections)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.title == "I. TÁC GIẢ"
    assert chunk.page_start == 5
    assert chunk.page_end == 5
    assert chunk.content_type == "prose"
    assert "tac_gia" in chunk.tags


def test_content_type_detection(chunker):
    """
    Test detection of different content types: prose, poem, dialogue, exercise, table, analysis, summary, list.
    """
    # 1. Exercise (based on title)
    c1 = chunker._detect_content_type("Luyện tập bài học", "Câu 1. Phân tích...")
    assert c1 == "exercise"
    
    # 2. Summary (based on title)
    c2 = chunker._detect_content_type("Ghi nhớ cuối bài", "Tóm tắt các giá trị...")
    assert c2 == "summary"
    
    # 3. Table
    c3 = chunker._detect_content_type("Bảng so sánh", "Cột A | Cột B | Cột C\nGiá trị | Nghệ thuật | Ý nghĩa")
    assert c3 == "table"
    
    # 4. Poem (multiple short lines)
    poem_content = (
        "Sông Mã xa rồi Tây Tiến ơi\n"
        "Nhớ về rừng núi nhớ chơi vơi\n"
        "Sài Khao sương lấp đoàn quân mỏi\n"
        "Mường Lát hoa về trong đêm hơi"
    )
    c4 = chunker._detect_content_type("Tây Tiến", poem_content)
    assert c4 == "poem"
    
    # 5. Dialogue
    dialogue_content = (
        "– Con đói quá u ơi!\n"
        "– Thôi nín đi con, u đi kiếm cái ăn cho."
    )
    c5 = chunker._detect_content_type("Trò chuyện", dialogue_content)
    assert c5 == "dialogue"
    
    # 6. List
    list_content = (
        "- Giá trị hiện thực độc đáo\n"
        "- Giá trị nhân đạo sâu sắc\n"
        "- Đặc sắc nghệ thuật dựng truyện"
    )
    c6 = chunker._detect_content_type("Đánh giá", list_content)
    assert c6 == "list"
    
    # 7. Analysis (based on keywords)
    analysis_content = "Đoạn văn này phân tích giá trị nhân đạo sâu sắc trong tác phẩm Vợ nhặt."
    c7 = chunker._detect_content_type("Phân tích tác phẩm", analysis_content)
    assert c7 == "analysis"
    
    # 8. Prose
    prose_content = "Kim Lân là nhà văn chuyên viết truyện ngắn và đã có một số tác phẩm có giá trị."
    c8 = chunker._detect_content_type("Giới thiệu", prose_content)
    assert c8 == "prose"


def test_tag_generation(chunker):
    """
    Test semantic tag generation from title and content.
    """
    title = "Phân tích giá trị nhân đạo của tác phẩm Vợ nhặt"
    content = "Tác phẩm thể hiện sâu sắc hoàn cảnh sống của nhân vật Tràng và Mị trong nạn đói năm 1945."
    tags = chunker._generate_tags(title, content)
    
    assert "nhan_vat" in tags
    assert "nhan_vat_trang" in tags
    assert "nhan_vat_mi" in tags
    assert "gia_tri_nhan_dao" in tags
    assert "tac_pham" in tags


def test_chunk_id_generation(chunker):
    """
    Test chunk ID generation format.
    """
    chunk_id = chunker._generate_chunk_id("Nhân vật Tràng", 2)
    assert chunk_id == "nhan_vat_trang_002"


def test_section_subsection_mapping(chunker):
    """
    Test that section level determines section_title and subsection_title mapping.
    """
    # Level 1 section
    sec_l1 = DocumentSection(
        title="I. TÁC GIẢ KIM LÂN",
        level=1,
        page_start=4,
        page_end=4,
        content=["Nhà văn Kim Lân sinh năm 1920."],
        parent_title="VỢ NHẶT"
    )
    chunks_l1 = chunker.chunk([sec_l1])
    assert chunks_l1[0].section_title == "I. TÁC GIẢ KIM LÂN"
    assert chunks_l1[0].subsection_title is None
    assert chunks_l1[0].parent_section == "VỢ NHẶT"

    # Level 2 section
    sec_l2 = DocumentSection(
        title="1. Hoàn cảnh sáng tác",
        level=2,
        page_start=6,
        page_end=6,
        content=["Tác phẩm được viết sau Cách mạng tháng Tám."],
        parent_title="II. TÁC PHẨM VỢ NHẶT"
    )
    chunks_l2 = chunker.chunk([sec_l2])
    assert chunks_l2[0].section_title == "II. TÁC PHẨM VỢ NHẶT"
    assert chunks_l2[0].subsection_title == "1. Hoàn cảnh sáng tác"
    assert chunks_l2[0].parent_section == "II. TÁC PHẨM VỢ NHẶT"


def test_overlap_generation(chunker):
    """
    Test that overlap is correctly generated and prepended between chunks in the same section.
    """
    # Create long paragraphs to trigger split.
    # To trigger a split on topic, we'll use topic changes in the paragraphs:
    # Paragraph 1 contains "giá trị hiện thực"
    # Paragraph 2 contains "giá trị nhân đạo"
    # We make Paragraph 1 have >50 words to allow overlap.
    p1 = "Giá trị hiện thực của tác phẩm được thể hiện qua bức tranh nạn đói khủng khiếp năm 1945. " * 10  # 150 words
    p2 = "Bên cạnh đó, giá trị nhân đạo của Vợ nhặt lại toả sáng qua tình yêu thương giữa những người nghèo."
    
    sections = [
        DocumentSection(
            title="Đánh giá giá trị",
            level=1,
            page_start=10,
            page_end=10,
            content=[p1, p2],
            parent_title=None
        )
    ]
    
    chunks = chunker.chunk(sections)
    assert len(chunks) == 2
    
    chunk_1 = chunks[0]
    chunk_2 = chunks[1]
    
    assert chunk_2.has_overlap is True
    assert chunk_2.overlap_from_chunk == chunk_1.chunk_id
    assert chunk_2.content.startswith(chunk_1.content.split()[-80:][0])
    assert p2 in chunk_2.content


def test_no_overlap_across_sections(chunker):
    """
    Test that overlap is NOT generated when transitioning to a new section.
    """
    p1 = "Giá trị hiện thực của tác phẩm được thể hiện qua bức tranh nạn đói khủng khiếp năm 1945. " * 10
    p2 = "Giá trị nhân đạo của tác phẩm thể hiện tấm lòng nhân hậu của Kim Lân."
    
    sections = [
        DocumentSection(
            title="Section 1",
            level=1,
            page_start=10,
            page_end=10,
            content=[p1],
            parent_title=None
        ),
        DocumentSection(
            title="Section 2",
            level=1,
            page_start=11,
            page_end=11,
            content=[p2],
            parent_title=None
        )
    ]
    
    chunks = chunker.chunk(sections)
    assert len(chunks) == 2
    assert chunks[0].has_overlap is False
    assert chunks[1].has_overlap is False
    assert chunks[1].overlap_from_chunk is None


def test_semantic_split_on_topic_change(chunker):
    """
    Test splitting a section when a different semantic topic (e.g. giá trị hiện thực vs nghệ thuật) is introduced.
    """
    sections = [
        DocumentSection(
            title="Phân tích",
            level=1,
            page_start=15,
            page_end=15,
            content=[
                "Đoạn văn này bàn về giá trị hiện thực của Vợ nhặt.",
                "Đoạn văn tiếp theo bàn về nghệ thuật đặc sắc của tác giả."
            ],
            parent_title=None
        )
    ]
    chunks = chunker.chunk(sections)
    assert len(chunks) == 2
    assert "gia_tri_hien_thuc" in chunks[0].tags
    assert "nghe_thuat" in chunks[1].tags


def test_keep_quotation_with_explanation(chunker):
    """
    Test that paragraphs ending with quotes are not split from the next explanation paragraph.
    """
    sections = [
        DocumentSection(
            title="Chi tiết trích dẫn",
            level=1,
            page_start=12,
            page_end=12,
            content=[
                'Nhân vật Tràng nói: "Thôi thì làm quen!"',  # Ends with quote
                "Câu nói ngẫu hứng này thể hiện sự khao khát hạnh phúc giản đơn."
            ],
            parent_title=None
        )
    ]
    # Even if there was a topic change or marker, ending with quote should prevent split.
    # Let's add a marker to the next paragraph to try and force a split, but it should be blocked.
    sections[0].content[1] = "Thứ nhất, câu nói này thể hiện khát khao hạnh phúc." # starts with marker
    
    chunks = chunker.chunk(sections)
    assert len(chunks) == 1  # Should remain in 1 chunk


def test_prompt_test_case(chunker):
    """
    Test case specified in prompt: "1. Nhân vật Tràng" with 5 paragraphs -> 1 chunk with 3 tags.
    """
    sections = [
        DocumentSection(
            title="1. Nhân vật Tràng",
            level=2,
            page_start=15,
            page_end=15,
            content=[
                "Nhân vật Tràng là nhân vật chính của truyện ngắn Vợ nhặt.",
                "Tràng hiện lên thô kệch, nghèo khổ giữa những ngày đói giáp hạt.",
                "Nhưng sâu bên trong tâm hồn Tràng là tấm lòng nhân hậu và niềm khao khát tổ ấm gia đình.",
                "Diễn biến tâm trạng nhân vật Tràng khi dẫn người vợ nhặt về nhà đầy bẽn lẽn mà vui sướng.",
                "Hình ảnh Tràng nghĩ đến lá cờ đỏ sao vàng ở cuối truyện mở ra tương lai tươi sáng."
            ],
            parent_title="II. TÁC PHẨM VỢ NHẶT"
        )
    ]
    chunks = chunker.chunk(sections)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "1_nhan_vat_trang_001"
    assert len(chunk.tags) == 3
    assert "nhan_vat" in chunk.tags
    assert "nhan_vat_trang" in chunk.tags
    assert "tam_ly_nhan_vat" in chunk.tags


def test_token_char_count(chunker):
    """
    Test token and character count calculations.
    """
    text = "Kim Lân là nhà văn xuất sắc viết về nông thôn."
    sections = [
        DocumentSection(
            title="Tiểu dẫn",
            level=1,
            page_start=2,
            page_end=2,
            content=[text],
            parent_title=None
        )
    ]
    chunks = chunker.chunk(sections)
    assert len(chunks) == 1
    assert chunks[0].token_count == 11  # 11 words
    assert chunks[0].char_count == len(text)
