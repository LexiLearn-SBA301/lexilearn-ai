
from pydantic import BaseModel, Field
from typing import Optional, List

# 1. Schema cho Request (Dữ liệu Client gửi lên)
class ChatRequest(BaseModel):
    # Dùng Field để thêm validate (giống @NotNull, @Size bên Java)
    thread_id: Optional[str] = Field(default=None, description="Mã phiên hội thoại (= conversation_id do BE sinh) để lưu lịch sử. Nếu không truyền sẽ tự sinh.")
    message: str = Field(..., min_length=1, max_length=2000, description="Tin nhắn của người dùng")
    system: Optional[str] = Field(default=None, description="Prompt hệ thống (nếu có)")
    filters: Optional[dict] = Field(default=None, description="Bộ lọc siêu dữ liệu")
    limit: int = Field(default=5, description="Số lượng chunk tối đa cần lấy")


# 1b. Seed lịch sử vào checkpoint (BE gọi khi mở lại chat cũ / checkpoint nguội).
# Mỗi phần tử history: {"role": "user"|"assistant", "content": str}, cũ -> mới, tối đa 10 (5 cặp).
class SeedRequest(BaseModel):
    thread_id: str = Field(..., description="Conversation id do BE sinh")
    history: List[dict] = Field(default_factory=list, description="Các lượt đã hoàn tất, cũ -> mới, tối đa 10 (5 cặp)")

# 2. Schema cho Response (Dữ liệu Server trả về)
class ChatResponse(BaseModel):
    answer: str
    model: str
    sources: list = Field(default_factory=list)

class WorkflowResponse(BaseModel):
    answer: str
    route: str