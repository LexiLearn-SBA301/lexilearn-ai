import sys
import json
import uuid
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, "src")

from schemas.sync_schema import WorkSnapshot
from services.sync_service import SyncService

def build_mock_work_snapshot(has_sections=False) -> WorkSnapshot:
    """Builds a mock WorkSnapshot payload exactly as BE would send it."""
    
    # 1. Base Work
    work = {
        "id": str(uuid.uuid4()),
        "title": "TỎ LÒNG",
        "slug": "to_long",
        "original_title": "Thuật hoài",
        "genre": "tho_ca",
        "sub_genre": "that_ngon_tu_tuyet",
        "period": "trung_dai",
        "grade": 10,
        "semester": 1,
        "publish_year": None,
        "summary": "Tỏ lòng là bài thơ thể hiện chí nam nhi và khát vọng lập công đền nợ nước.",
        "cover_url": "https://example.com/cover.jpg",
        "is_published": True,
        "historical_context": "Sáng tác trong cuộc kháng chiến chống quân Mông - Nguyên lần thứ hai (1285).",
        "realistic_value": "Phản ánh khí thế hào hùng của quân đội nhà Trần.",
        "humanistic_value": "Ca ngợi vẻ đẹp của con người thời Trần với lý tưởng cao cả.",
        "artistic_value": "Thể thơ thất ngôn tứ tuyệt hàm súc, hình ảnh kỳ vĩ.",
        "famous_quote": "Công danh nam tử còn vương nợ / Luống thẹn tai nghe chuyện Vũ Hầu.",
        "quote_attribution": "Phạm Ngũ Lão"
    }

    # 2. Base Author
    author = {
        "id": str(uuid.uuid4()),
        "name": "Phạm Ngũ Lão",
        "pen_name": None,
        "slug": "pham_ngu_lao",
        "birth_year": 1255,
        "death_year": 1320,
        "period": "trung_dai",
        "bio": "Phạm Ngũ Lão là danh tướng đời Trần, người làng Phù Ủng, huyện Đường Hào, tỉnh Hưng Yên.",
        "portrait_url": None
    }

    # 3. Tags
    tags = [
        {"id": str(uuid.uuid4()), "name": "Thơ trung đại", "slug": "tho_trung_dai", "description": None},
        {"id": str(uuid.uuid4()), "name": "Văn học yêu nước", "slug": "van_hoc_yeu_nuoc", "description": None}
    ]

    # 4. Sections (Empty for Case 1, 3 Sections for Case 2)
    sections = []
    if has_sections:
        sections = [
            {
                "id": str(uuid.uuid4()),
                "number": 1,
                "title": "Phiên âm",
                "content": "Hoành sóc giang sơn cáp kỷ thu\nTam quân tỳ hổ khí thôn ngưu\nNam nhi vị liễu công danh trái\nTu thính nhân gian thuyết Vũ hầu.",
                "content_type": "POETRY",
                "word_count": 28
            },
            {
                "id": str(uuid.uuid4()),
                "number": 2,
                "title": "Dịch nghĩa",
                "content": "Cầm ngang ngọn giáo trấn giữ non sông đã mấy thu,\nBa quân như hổ báo, khí thế nuốt sao ngưu.\nThân nam nhi mà chưa trả xong nợ công danh,\nThì luống thẹn tai nghe người đời kể chuyện Vũ hầu.",
                "content_type": "PROSE",
                "word_count": 42
            },
            {
                "id": str(uuid.uuid4()),
                "number": 3,
                "title": "Dịch thơ",
                "content": "Múa giáo non sông trải mấy thu,\nBa quân khí mạnh nuốt trôi trâu.\nCông danh nam tử còn vương nợ,\nLuống thẹn tai nghe chuyện Vũ hầu.",
                "content_type": "POETRY",
                "word_count": 28
            }
        ]
        
    # 5. Commentaries
    commentaries = [
        {
            "id": str(uuid.uuid4()),
            "title": "Khí phách tuổi trẻ thời Trần",
            "content": "Bài thơ mang đậm hào khí Đông A, thể hiện sức mạnh của dân tộc trong kỷ nguyên chống ngoại xâm.",
            "commentator_name": "Lê Trí Viễn",
            "commentator_type": "EXPERT",
            "source_title": "Bình giảng văn học",
            "source_url": None,
            "published_year": 1998,
            "display_order": 1,
            "is_featured": True,
            "is_published": True
        }
    ]

    payload = {
        "schema_version": "literature_work_snapshot.v1",
        "synced_at": datetime.utcnow().isoformat() + "Z",
        "work": work,
        "author": author,
        "sections": sections,
        "commentaries": commentaries,
        "tags": tags
    }
    
    return WorkSnapshot(**payload)

def print_chunks(chunks):
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- CHUNK {i} ---")
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"category: {chunk.metadata.chunk_category}")
        print(f"content:\n{chunk.content}")
        print("metadata:")
        meta_dict = chunk.metadata.model_dump(exclude_none=True)
        print(json.dumps(meta_dict, ensure_ascii=False, indent=2))

def run_test():
    # Mock services so we don't need real Mongo/Ollama for testing the builder
    class MockWriter:
        def deactivate_by_work_slug(self, slug): return 0
        def upsert_chunk(self, chunk): pass
        
    class MockEmbedder:
        @property
        def model_name(self): return "mock_model"
        
    svc = SyncService(writer=MockWriter(), embedder=MockEmbedder())

    print("=" * 80)
    print("CASE 1: TÁC PHẨM VỪA TẠO (CHƯA CÓ SECTION)")
    print("=" * 80)
    payload_empty = build_mock_work_snapshot(has_sections=False)
    chunks_empty, _ = svc._build_all_chunks(payload_empty)
    print(f"Total chunks created: {len(chunks_empty)}")
    print_chunks(chunks_empty)
    
    print("\n" + "=" * 80)
    print("CASE 2: ĐẦY ĐỦ DỮ LIỆU (CÓ 3 SECTIONS & COMMENTARY)")
    print("=" * 80)
    payload_full = build_mock_work_snapshot(has_sections=True)
    chunks_full, _ = svc._build_all_chunks(payload_full)
    print(f"Total chunks created: {len(chunks_full)}")
    print_chunks(chunks_full)

if __name__ == "__main__":
    run_test()
