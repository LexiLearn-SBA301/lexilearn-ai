import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from main import app
from api.chat_router import get_rag_service


from services.rag_service import RAGService

class MockRAGService(RAGService):
    def __init__(self):
        self.db = MagicMock()
        self.works_metadata = MagicMock()
        self.chunks = MagicMock()
        self.db.__getitem__.side_effect = self._get_collection

    def _get_collection(self, name):
        if name == "works_metadata":
            return self.works_metadata
        elif name == "chunks":
            return self.chunks
        return MagicMock()


@pytest.fixture
def mock_rag():
    return MockRAGService()


@pytest.fixture
def client(mock_rag):
    app.dependency_overrides[get_rag_service] = lambda: mock_rag
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_get_suggestions_cached(client, mock_rag):
    # 1. Mock DB cache hit
    mock_rag.works_metadata.find_one.return_value = {
        "work_title": "SÓNG",
        "author_name": "Xuân Quỳnh",
        "suggested_questions": [
            "Hoàn cảnh sáng tác bài thơ Sóng?",
            "Phân tích hình tượng sóng và em?",
            "Vẻ đẹp tâm hồn người phụ nữ ngày nay qua bài thơ?"
        ]
    }

    # 2. Call API
    response = client.get("/chat/works/suggestions?work_title=Sóng")
    
    # 3. Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["work_title"] == "SÓNG"
    assert data["author_name"] == "Xuân Quỳnh"
    assert len(data["suggested_questions"]) == 3
    assert data["suggested_questions"][0].startswith("Hoàn cảnh")
    
    # Verify cached branch was entered
    mock_rag.works_metadata.find_one.assert_called_once()
    mock_rag.chunks.find.assert_not_called()


def test_get_suggestions_lazy_cache(client, mock_rag):
    # 1. Mock DB cache miss
    mock_rag.works_metadata.find_one.return_value = None
    
    # Mock chunks found in DB
    mock_rag.chunks.find.return_value.limit.return_value = [
        {
            "content": "Sóng bắt đầu từ gió, Gió bắt đầu từ đâu...",
            "metadata": {
                "work_title": "SÓNG",
                "author_name": "Xuân Quỳnh",
                "grade": 12,
                "semester": 1
            }
        }
    ]

    # 2. Mock Gemini API
    mock_gemini_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "questions": [
            "Câu hỏi factual 1",
            "Câu hỏi analysis 2",
            "Câu hỏi comparative 3"
        ]
    })
    mock_gemini_client.models.generate_content.return_value = mock_resp

    with patch("providers.gemini_provider.gemini_provider.get_client", return_value=mock_gemini_client):
        # Call API
        response = client.get("/chat/works/suggestions?work_title=Sóng")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["work_title"] == "SÓNG"
        assert len(data["suggested_questions"]) == 3
        assert data["suggested_questions"][0] == "Câu hỏi factual 1"
        
        # Verify db was updated (cache saved)
        mock_rag.works_metadata.update_one.assert_called_once()


def test_get_suggestions_not_found(client, mock_rag):
    # DB cache miss
    mock_rag.works_metadata.find_one.return_value = None
    # No chunks found in DB either
    mock_rag.chunks.find.return_value.limit.return_value = []

    response = client.get("/chat/works/suggestions?work_title=KhôngTồnTại")
    
    assert response.status_code == 404
    assert "detail" in response.json()
    assert "Không tìm thấy tác phẩm" in response.json()["detail"]
