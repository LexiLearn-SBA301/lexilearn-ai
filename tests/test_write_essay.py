import asyncio
import os
import sys
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.write_essay import _render_debate_data, _render_nguyen_lieu, write_essay
from state.state_schema import (
    Stage, DebateState, PreparedContext, CriticRole, CriticTurn, Argument, Rebuttal, SourceChunk,
)
from agents.essay_schemas import EssayLLMOutput, EssaySectionOut

def test_render_nguyen_lieu_dua_van_ban_goc_vao_prompt():
    """Model được fine-tune để TRÍCH NGUYÊN VĂN từ [Ngữ liệu] -> phải thấy chunks thật.

    Trước đây node chỉ truyền summary + entities, chunks không bao giờ tới model, nên mọi
    câu trích dẫn trong bài đều là bịa.
    """
    context = PreparedContext(
        retrieval_query="query",
        chunks=[SourceChunk(chunk_id="c1", text="Hoành sóc giang sơn cáp kỷ thu"),
                SourceChunk(chunk_id="c2", text="Tam quân tỳ hổ khí thôn ngưu")],
        summary="Tóm tắt tác phẩm X",
    )
    res = _render_nguyen_lieu(context)
    assert "(1) Hoành sóc giang sơn cáp kỷ thu" in res    # đánh số đúng format fine-tune
    assert "(2) Tam quân tỳ hổ khí thôn ngưu" in res


def test_render_debate_data_khong_in_trung_phan_bien():
    """consensus/contested được dựng TỪ reason của rebuttal -> in cả 2 khối là lặp nguyên văn."""
    reason = "Câu kết là nỗi thẹn, không phải lý tưởng phục vụ"
    debate = DebateState(
        total_invocations=0,
        round1={
            CriticRole.TAM_LY: CriticTurn(
                round=1,
                critic=CriticRole.TAM_LY,
                thesis="Tâm lý phức tạp",
                arguments=[Argument(point="P1", support="S1", arg_id="tam_ly-a1")],
                parsed_ok=True
            )
        },
        round2={
            CriticRole.TAM_LY: CriticTurn(
                round=2, critic=CriticRole.TAM_LY, thesis="Tâm lý phức tạp",
                rebuttals=[Rebuttal(target_critic=CriticRole.LICH_SU, target_arg_id="lich_su-a1",
                                    stance="disagree", reason=reason)],
                parsed_ok=True,
            )
        },
        consensus_points=[],
        contested_points=[f"Nhà phê bình Tâm lý → Nhà phê bình Lịch sử: {reason}"],
    )
    res = _render_debate_data(debate)
    assert "Tâm lý phức tạp" in res
    assert "P1: S1" in res
    assert res.count(reason) == 1          # xuất hiện ĐÚNG 1 lần, không lặp
    assert "ĐIỂM TRANH CÃI" not in res


def test_write_essay_success():
    state = {
        "human_message": "Viết bài nghị luận",
        "context": PreparedContext(summary="Tóm tắt context", chunks=[]),
        "debate": DebateState(total_invocations=0, round1={}, round2={}, consensus_points=[], contested_points=[]),
        "last_feedback": {}
    }
    
    mock_structured_llm = MagicMock()
    # Trả về object đúng chuẩn Pydantic mà Langchain Structured Output sẽ trả ra
    mock_structured_llm.ainvoke = AsyncMock(return_value=EssayLLMOutput(
        thinking="Suy nghĩ lập dàn ý",
        title="Tiêu đề hay và nghệ thuật",
        sections=[
            EssaySectionOut(heading="Mở bài", body="Đoạn mở đầu có sức hút."),
            EssaySectionOut(heading="Thân bài", body="Nội dung chính phân tích đa chiều."),
        ]
    ))

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("agents.write_essay.ollama_provider.get_llm", return_value=mock_llm):
        out = asyncio.run(write_essay(state))
        
    assert out["current_stage"] == Stage.WRITE_ESSAY
    assert out["current_node"] == "write_essay"
    
    essay = out["essay"]
    assert essay.title == "Tiêu đề hay và nghệ thuật"
    assert "Mở bài" in essay.full_text
    assert "Đoạn mở đầu có sức hút." in essay.full_text
    assert "Thân bài" in essay.full_text
    assert essay.word_count > 0


def test_write_essay_fallback_on_error():
    state = {
        "human_message": "Viết bài nghị luận",
        "context": PreparedContext(summary="Tóm tắt context", chunks=[]),
        "debate": DebateState(total_invocations=0, round1={}, round2={}, consensus_points=[], contested_points=[])
    }
    
    mock_structured_llm = MagicMock()
    mock_structured_llm.ainvoke = AsyncMock(side_effect=Exception("Model JSON parse error"))

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("agents.write_essay.ollama_provider.get_llm", return_value=mock_llm):
        out = asyncio.run(write_essay(state))
        
    # Cần fallback trả về lỗi chứ không văng exception làm sập luồng
    essay = out["essay"]
    assert essay.title == "Lỗi sinh bài luận"
    assert "[LỖI TOOL 3]" in essay.full_text
