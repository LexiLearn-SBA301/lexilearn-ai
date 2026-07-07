import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.write_essay import _render_debate_data, _render_context_summary, write_essay
from state.state_schema import Stage, DebateState, PreparedContext, CriticRole, CriticTurn, Argument, Entity
from agents.essay_schemas import EssayLLMOutput, EssaySectionOut

def test_render_context_summary():
    context = PreparedContext(
        retrieval_query="query",
        chunks=[],
        summary="Tóm tắt tác phẩm X",
        entities=[Entity(name="A", type="character", description="desc A")]
    )
    res = _render_context_summary(context)
    assert "Tóm tắt tác phẩm X" in res
    assert "A (character): desc A" in res


def test_render_debate_data():
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
        round2={},
        consensus_points=["Đồng thuận 1"],
        contested_points=[]
    )
    res = _render_debate_data(debate)
    assert "ĐỒNG THUẬN" in res
    assert "Đồng thuận 1" in res
    assert "Tâm lý phức tạp" in res
    assert "P1: S1" in res


def test_write_essay_success():
    state = {
        "human_message": "Viết bài nghị luận",
        "context": PreparedContext(summary="Tóm tắt context", chunks=[]),
        "debate": DebateState(total_invocations=0, round1={}, round2={}, consensus_points=[], contested_points=[]),
        "last_feedback": {}
    }
    
    mock_structured_llm = MagicMock()
    # Trả về object đúng chuẩn Pydantic mà Langchain Structured Output sẽ trả ra
    mock_structured_llm.invoke.return_value = EssayLLMOutput(
        thinking="Suy nghĩ lập dàn ý",
        title="Tiêu đề hay và nghệ thuật",
        sections=[
            EssaySectionOut(heading="Mở bài", body="Đoạn mở đầu có sức hút."),
            EssaySectionOut(heading="Thân bài", body="Nội dung chính phân tích đa chiều."),
        ]
    )

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("agents.write_essay.ollama_provider.get_llm", return_value=mock_llm):
        out = write_essay(state)
        
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
    mock_structured_llm.invoke.side_effect = Exception("Model JSON parse error")
    
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("agents.write_essay.ollama_provider.get_llm", return_value=mock_llm):
        out = write_essay(state)
        
    # Cần fallback trả về lỗi chứ không văng exception làm sập luồng
    essay = out["essay"]
    assert essay.title == "Lỗi sinh bài luận"
    assert "[LỖI TOOL 3]" in essay.full_text
