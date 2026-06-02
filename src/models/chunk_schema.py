from pydantic import BaseModel, Field
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
    ten_tac_pham: str = Field(..., description="Tên tác phẩm (ví dụ: Vợ Nhặt)")
    tac_gia: str = Field(..., description="Tên tác giả (ví dụ: Kim Lân)")
    lop: int = Field(..., description="Lớp học (ví dụ: 12)")
    the_loai: str = Field(..., description="Thể loại văn học (ví dụ: truyen_ngan)")
    hoc_ki: int = Field(..., description="Học kì học tác phẩm này (ví dụ: 1)")
    nam_sang_tac: Optional[int] = Field(None, description="Năm sáng tác tác phẩm (ví dụ: 1962)")
    tags: List[str] = Field(default_factory=list, description="Các nhãn đặc tả nội dung")


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

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "chunk_id": "truyen-vo-nhat_p003_c02",
                "source_doc_id": "truyen-vo-nhat_p003",
                "content": "Tràng đi một mình giữa đường vắng...",
                "content_type": "prose",
                "position": {
                    "page": 12,
                    "chunk_index": 2,
                    "total_chunks": 18
                },
                "metadata": {
                    "ten_tac_pham": "Vợ Nhặt",
                    "tac_gia": "Kim Lân",
                    "lop": 12,
                    "the_loai": "truyen_ngan",
                    "hoc_ki": 1,
                    "nam_sang_tac": 1962,
                    "tags": ["nhan_vat_trang", "tam_ly", "nan_doi"]
                },
                "token_count": 312,
                "char_count": 687,
                "has_overlap": True,
                "embedding": [0.023, -0.145, 0.089],
                "search_text": "Trang di mot minh giua duong vang",
                "model_version": "bge-m3-v1.0",
                "is_active": True
            }
        }

# Alias để tương thích ngược nếu cần gọi bằng class name cũ
chunk_schema = ChunkSchema