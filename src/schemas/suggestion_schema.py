from pydantic import BaseModel, Field

class SuggestionsResponse(BaseModel):
    work_title: str
    author_name: str
    suggested_questions: list[str] = Field(description="Danh sách đúng 3 câu hỏi gợi ý.")

class SuggestedQuestionsOut(BaseModel):
    questions: list[str] = Field(description="Danh sách đúng 3 câu hỏi gợi ý.")
