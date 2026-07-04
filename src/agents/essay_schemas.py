from pydantic import BaseModel, Field

class EssaySectionOut(BaseModel):
    heading: str = Field(description="Tên phần, vd: 'Mở bài', 'Phân tích nghệ thuật', 'Kết bài'")
    body: str = Field(description="Nội dung phần này, viết đầy đủ và mạch lạc")

class EssayLLMOutput(BaseModel):
    """Schema mà LLM phải trả về. Trường `thinking` là chain-of-thought ẩn."""
    
    thinking: str = Field(
        description=(
            "Suy nghĩ nội bộ TRƯỚC khi viết bài. "
            "Hãy liệt kê: (1) luận điểm nào từ debate đáng dùng nhất, "
            "(2) luận điểm nào bị phản biện mạnh nên bỏ, "
            "(3) dàn ý sơ bộ (mở-thân-kết). "
            "Trường này KHÔNG hiển thị cho người đọc."
        )
    )
    title: str = Field(description="Tiêu đề bài nghị luận")
    sections: list[EssaySectionOut] = Field(
        description="Các phần của bài văn theo thứ tự: Mở bài, Thân bài (nhiều đoạn), Kết bài"
    )
