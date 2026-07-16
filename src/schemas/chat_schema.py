
from pydantic import BaseModel, Field
from typing import Optional

# 1. Schema cho Request (Dữ liệu Client gửi lên)
class ChatRequest(BaseModel):
    # Dùng Field để thêm validate (giống @NotNull, @Size bên Java)
    thread_id: Optional[str] = Field(default=None, description="Mã phiên hội thoại để lưu lịch sử. Nếu không truyền sẽ tự sinh.")
    message: str = Field(..., min_length=1, max_length=2000, description="Tin nhắn của người dùng")
    system: Optional[str] = Field(default=None, description="Prompt hệ thống (nếu có)")
    filters: Optional[dict] = Field(default=None, description="Bộ lọc siêu dữ liệu")
    limit: int = Field(default=5, description="Số lượng chunk tối đa cần lấy")

# 2. Schema cho Response (Dữ liệu Server trả về)
class ChatResponse(BaseModel):
    answer: str
    model: str
    sources: list = Field(default_factory=list)

class WorkflowResponse(BaseModel):
    answer: str
    route: str