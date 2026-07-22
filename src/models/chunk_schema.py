from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime, timezone

class ChunkPosition(BaseModel):
    """
    Vị trí của chunk trong tài liệu gốc.
    """
    page: int = Field(..., description="Trang chứa chunk trong tài liệu gốc")
    chunk_index: int = Field(..., description="Thứ tự chunk trong file gốc (0-indexed)")
    total_chunks: int = Field(..., description="Tổng số chunks của tài liệu gốc")


class ChunkMetadata(BaseModel):
    """
    Thông tin siêu dữ liệu (metadata) của tác phẩm văn học.
    """
    schema_version: str = Field("literature_seed.v1", description="Phiên bản schema")
    source: Optional[str] = Field(None, description="Nguồn dữ liệu: 'docx_ingest' hoặc 'be_sync'")
    work_id: Optional[str] = Field(None, description="UUID tác phẩm từ BE Postgres")
    author_id: Optional[str] = Field(None, description="UUID tác giả từ BE Postgres")
    section_id: Optional[str] = Field(None, description="UUID section từ BE Postgres")

    work_title: Optional[str] = Field(None, description="Tên tác phẩm (ví dụ: Tỏ Lòng)")
    work_slug: Optional[str] = Field(None, description="Slug của tên tác phẩm")
    author_name: str = Field(..., description="Tên tác giả (ví dụ: Phạm Ngũ Lão)")
    author_slug: str = Field(..., description="Slug tên tác giả")

    author_period: Optional[str] = Field(None, description="Thời kỳ tác giả (dan_gian, trung_dai, hien_dai)")
    work_period: Optional[str] = Field(None, description="Thời kỳ tác phẩm (dan_gian, trung_dai, hien_dai)")

    genre: Optional[str] = Field(None, description="Thể loại văn học dạng snake_case (ví dụ: tho_ca, truyen_ngan, su_thi)")
    sub_genre: Optional[str] = Field(None, description="Tiểu thể loại dạng snake_case (ví dụ: that_ngon_tu_tuyet). Null nếu chưa xác định.")

    grade: Optional[int] = Field(None, description="Lớp học (ví dụ: 12)")
    semester: Optional[int] = Field(None, description="Học kì học tác phẩm này (ví dụ: 1)")
    publish_year: Optional[int] = Field(None, description="Năm xuất bản")

    chunk_category: str = Field(..., description="Phân loại chunk (ví dụ: text_section, author_bio, etc.)")

    section_slug: Optional[str] = Field(None, description="Slug của đoạn văn bản gốc")
    section_title: Optional[str] = Field(None, description="Tiêu đề đoạn văn bản gốc")
    section_order: Optional[int] = Field(None, description="Thứ tự của đoạn")

    content_type: str = Field(..., description="Kiểu nội dung trong metadata (PROSE, POETRY, MIXED)")


class ChunkSchema(BaseModel):
    """
    Đại diện cho lược đồ dữ liệu (schema) của một chunk ngữ nghĩa lưu trữ dưới database.
    """
    # Lưu ý: Trường _id sẽ được tự động sinh bởi MongoDB khi insert,
    # nên ta không cần định nghĩa ở đây hoặc có thể xử lý ở mức DB driver.
    
    chunk_id: str = Field(..., description="Mã định danh duy nhất của chunk (bắt buộc)")
    source_doc_id: str = Field(..., description="Mã định danh tài liệu nguồn chứa chunk này (bắt buộc)")

    content: str = Field(..., description="Nội dung văn bản của chunk (bắt buộc)")
    content_type: str = Field(..., description="Kiểu nội dung (ví dụ: prose, poem, table, list) (bắt buộc)")

    position: ChunkPosition = Field(..., description="Thông tin vị trí của chunk")
    metadata: ChunkMetadata = Field(..., description="Metadata tác phẩm tương ứng")

    token_count: int = Field(..., description="Số lượng tokens ước tính (tự sinh)")
    char_count: int = Field(..., description="Số lượng ký tự (tự sinh)")
    has_overlap: bool = Field(..., description="Đánh dấu có chứa đoạn overlap từ chunk trước hay không")

    embedding: Optional[List[float]] = Field(None, description="Vector nhúng biểu diễn nội dung (tự sinh, 1024 chiều)")
    search_text: str = Field(..., description="Nội dung đã được chuẩn hóa không dấu dùng cho Full-text search (tự sinh)")

    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Thời điểm lưu trữ vào DB (tự sinh)"
    )
    model_version: str = Field(..., description="Phiên bản mô hình embedding sử dụng (tự sinh)")
    is_active: bool = Field(True, description="Đánh dấu chunk có đang kích hoạt/sử dụng không (bắt buộc)")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "chunk_id": "to-long_c01",
                "source_doc_id": "to-long",
                "content": "Múa giáo non sông trải mấy thu...",
                "content_type": "POETRY",
                "position": {
                    "page": 1,
                    "chunk_index": 0,
                    "total_chunks": 5
                },
                "metadata": {
                    "schema_version": "literature_seed.v1",
                    "work_title": "Tỏ Lòng",
                    "work_slug": "to-long",
                    "author_name": "Phạm Ngũ Lão",
                    "author_slug": "pham-ngu-lao",
                    "author_period": "trung_dai",
                    "work_period": "trung_dai",
                    "genre": "tho_ca",
                    "sub_genre": "that_ngon_tu_tuyet",
                    "grade": 12,
                    "semester": 1,
                    "publish_year": None,
                    "chunk_category": "text_section",
                    "section_slug": "phien-am",
                    "section_title": "Phiên âm",
                    "section_order": 1,
                    "content_type": "POETRY"
                },
                "token_count": 45,
                "char_count": 120,
                "has_overlap": False,
                "embedding": [0.023, -0.145, 0.089],
                "search_text": "Mua giao non song trai may thu ba quan khi manh nuot troi trau",
                "model_version": "bge-m3-v1.0",
                "is_active": True
            }
        }
    )

# Alias để tương thích ngược nếu cần gọi bằng class name cũ
chunk_schema = ChunkSchema