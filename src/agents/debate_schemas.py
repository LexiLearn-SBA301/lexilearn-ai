"""
Hợp đồng structured-output của LLM cho Tool 2 (critics_debate) — tách khỏi node để
critics_debate.py chỉ còn logic điều phối.

Đây là schema SLIM: CHỈ field do LLM viết (giảm gánh cho Qwen-3B). Server ráp
thành CriticTurn (state persist ở state/state_schema.py) sau.

Lưu ý: DebateSubState (state nội bộ của subgraph) vẫn ở critics_debate.py vì gắn
liền build_debate_subgraph — tách qua module khác làm type-checker mất nhận diện
TypedDict cho StateGraph.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class _ArgIn(BaseModel):
    point: str
    support: str = ""


class CriticR1Out(BaseModel):
    thesis: str
    arguments: list[_ArgIn] = Field(default_factory=list)


class _RebuttalIn(BaseModel):
    # CHỈ 1 field định danh mục tiêu: arg_id chép nguyên từ bảng tin (vd "lich_su-a2").
    # Server suy target_critic TỪ id này. Trước đây model điền tách rời (target_critic,
    # target_point) nên bắt bẻ luận điểm của critic A mà gắn nhãn critic B vẫn lọt: số
    # thứ tự luận điểm reset về 1 ở MỖI critic, nên "point 2" của ai cũng có -> id ghép
    # ra vẫn tồn tại, validate không bắt được. Gộp về 1 lựa chọn nguyên tử -> hết lệch.
    target_arg_id: str = ""
    stance: Literal["agree", "disagree", "qualify"] = "qualify"
    reason: str = ""


class CriticR2Out(BaseModel):
    rebuttals: list[_RebuttalIn] = Field(default_factory=list)
