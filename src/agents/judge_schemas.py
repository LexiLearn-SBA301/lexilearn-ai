"""
Hợp đồng structured-output của LLM cho Supervisor judge — tách khỏi node để
supervisor_judge.py chỉ còn logic điều phối. Giống agents/debate_schemas.py.

Schema SLIM: LLM chỉ được chọn pass/retry. 'approve'/'reject' (state_schema.Verdict)
KHÔNG cho LLM tự chọn — reject là quyết định của server khi hết lượt retry
(xem agents/supervisor_judge.py), approve dành cho duyệt cuối ngoài phạm vi judge này.

`scores` là list[CriterionScore] chứ không phải dict[str, float]: Gemini Developer
API (không phải Vertex Enterprise) không hỗ trợ "additionalProperties" trong
response_schema -> dict tự do sẽ lỗi lúc gọi. Server ráp lại thành dict cho
state_schema.JudgeVerdict.scores sau khi nhận kết quả.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    criterion: str
    score: float


class JudgeOut(BaseModel):
    verdict: Literal["pass", "retry"]
    scores: list[CriterionScore] = Field(default_factory=list)
    reasoning: str = ""
    feedback: str = ""
