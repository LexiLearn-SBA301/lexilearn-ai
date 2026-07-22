"""
Test 2 endpoint tranh luận qua FastAPI THẬT (router + exception_handlers), không cần Ollama.

Dựng app tối giản chỉ gồm chat_router + handler thay vì import main.py: lifespan của main
mở Mongo + Redis, mà 2 endpoint này không đụng tới cái nào.

Điểm cần chốt: lỗi phải ra ĐÚNG mã HTTP. Đường LLM bịa id thì bỏ im lặng, còn id do FE gửi
sai là bug -> phải 400 chứ không được nuốt; hết phiên chờ là 409 chứ không phải 500.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.chat_router import router as chat_router
from api.exception_handlers import register_exception_handlers
from services.agent_service import debate_session

THREAD = "t-api-1"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(chat_router)
    register_exception_handlers(app)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean():
    debate_session.close_session(THREAD)
    debate_session.clear_optin(THREAD)
    yield
    debate_session.close_session(THREAD)
    debate_session.clear_optin(THREAD)


def test_optin_sets_flag(client):
    r = client.post("/chat/debate/optin", json={"thread_id": THREAD})
    assert r.status_code == 200
    assert r.json()["optin"] is True
    assert debate_session.has_optin(THREAD) is True


def test_reply_without_session_returns_409(client):
    r = client.post("/chat/debate/reply", json={"thread_id": THREAD, "message": "xin chào"})
    assert r.status_code == 409
    assert "khép lại" in r.json()["detail"]


def test_reply_round1_lands_in_queue(client):
    p = debate_session.open_session(THREAD, 1, set())
    r = client.post("/chat/debate/reply", json={"thread_id": THREAD, "message": "Ý của tôi"})
    assert r.status_code == 200
    assert r.json()["ended"] is False
    assert p.queue.get_nowait().message == "Ý của tôi"


def test_reply_round2_bad_arg_id_returns_400(client):
    debate_session.open_session(THREAD, 2, {"tam_ly-a1"})
    r = client.post("/chat/debate/reply", json={
        "thread_id": THREAD, "message": "x", "target_arg_id": "bia-a9", "stance": "agree"})
    assert r.status_code == 400
    assert "không có trong bảng tin" in r.json()["detail"]


def test_reply_round2_bad_stance_returns_422(client):
    """stance lạ bị Literal của Pydantic chặn ngay ở biên -> 422, chưa vào tới service."""
    debate_session.open_session(THREAD, 2, {"tam_ly-a1"})
    r = client.post("/chat/debate/reply", json={
        "thread_id": THREAD, "message": "x", "target_arg_id": "tam_ly-a1", "stance": "ghet"})
    assert r.status_code == 422


def test_reply_round2_ok(client):
    p = debate_session.open_session(THREAD, 2, {"tam_ly-a1"})
    r = client.post("/chat/debate/reply", json={
        "thread_id": THREAD, "message": "Không đúng", "target_arg_id": "tam_ly-a1",
        "stance": "disagree"})
    assert r.status_code == 200
    item = p.queue.get_nowait()
    assert (item.target_arg_id, item.stance) == ("tam_ly-a1", "disagree")


@pytest.mark.parametrize("body", [
    {"thread_id": THREAD},                      # Bỏ qua: không gửi field message
    {"thread_id": THREAD, "message": None},     # Kết thúc: message null
    {"thread_id": THREAD, "message": "   "},    # gõ toàn khoảng trắng -> coi như kết thúc
])
def test_empty_message_signals_end(client, body):
    """Bỏ qua / Kết thúc / gõ trắng đều là CÙNG một tín hiệu None cho node đang chờ."""
    p = debate_session.open_session(THREAD, 1, set())
    r = client.post("/chat/debate/reply", json=body)
    assert r.status_code == 200
    assert r.json()["ended"] is True
    assert p.queue.get_nowait() is None
