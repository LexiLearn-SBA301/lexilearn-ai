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

from typing import Literal, Optional

from pydantic import BaseModel, Field

from state.state_schema import CriticRole


class _ArgIn(BaseModel):
    point: str
    support: str = ""


class CriticR1Out(BaseModel):
    thesis: str
    arguments: list[_ArgIn] = Field(default_factory=list)


class _RebuttalIn(BaseModel):
    target_critic: CriticRole
    target_point: Optional[int] = None   # số thứ tự luận điểm của critic đó (theo bảng tin); server map -> arg_id
    stance: Literal["agree", "disagree", "qualify"] = "qualify"
    reason: str = ""


class CriticR2Out(BaseModel):
    rebuttals: list[_RebuttalIn] = Field(default_factory=list)
