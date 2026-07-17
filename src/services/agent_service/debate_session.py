"""
Chỗ hẹn giữa node debate đang PAUSE và HTTP request mang lời của người học.

VÌ SAO CẦN: SSE chỉ chạy 1 chiều (server -> client), nên tin của người học BẮT BUỘC tới
bằng MỘT REQUEST KHÁC (POST /chat/debate/reply). Hai coroutine trong cùng process cần một
điểm hẹn -> dict theo thread_id, mỗi phiên chờ một asyncio.Queue.

VÌ SAO QUEUE MÀ KHÔNG PHẢI FUTURE: người học được gửi tối đa MAX_HUMAN_TURNS tin liên tiếp.
Future chỉ resolve 1 lần; nếu tin thứ 2 tới TRƯỚC lúc node kịp đăng ký Future mới thì tin
đó mất trắng. Queue buffer sẵn -> put_nowait() không bao giờ trượt, node đọc lúc nào cũng
được.

PHẠM VI: chỉ sống TRONG process, y như _PROD_APP của critics_debate — AI service chạy 1
instance (uvicorn 1 worker). Muốn scale nhiều replica thì cả file này lẫn cờ opt-in phải
chuyển sang Redis pub/sub, và đó là lúc nên cân nhắc lại LangGraph interrupt() thay vì
giữ kết nối SSE mở.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from exceptions import DebateInvalidReply, DebateNotWaiting

logger = logging.getLogger("rag-service.services.debate_session")

MAX_HUMAN_TURNS = 10       # trần số tin người học gửi trong MỘT vòng
IDLE_TIMEOUT_S = 180.0     # không nhận tin mới quá ngần này -> hội đồng tự đi tiếp

STANCES = ("agree", "disagree", "qualify")   # khớp Rebuttal.stance ở state_schema

# thread_id đã bấm "Tranh luận cùng AI" nhưng debate CHƯA chạy tới nơi.
_optin: set[str] = set()

# thread_id -> phiên đang chờ. CHỈ tồn tại trong lúc node debate pause.
_sessions: dict[str, "_Pending"] = {}


def normalize_arg_id(raw: Optional[str]) -> str:
    """Chuẩn hoá id trước khi tra bảng tin: chấp nhận cả "[lich_su-a1]" lẫn "lich_su-a1".
    """
    return (raw or "").strip().strip("[]").strip().lower()


@dataclass
class HumanReply:
    """1 tin của người học.

    Vòng 1 chỉ dùng `message` (nêu luận điểm của mình -> thành Argument có arg_id để 4
    critic bắt bẻ lại ở vòng 2). Vòng 2 dùng cả 3 field (-> thành Rebuttal).
    """
    message: str
    target_arg_id: Optional[str] = None
    stance: Optional[str] = None


@dataclass
class _Pending:
    round: int
    valid_arg_ids: set[str]                  # rỗng ở vòng 1 (chưa có gì để nhắm tới)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
#field(default_factory=asyncio.Queue) -> mỗi lần tạo Pending thì mới tạo 1 queue riêng chứ không sài chung
#queue: asyncio.Queue = asyncio.Queue() -> tạo queue chung biến static

# =============================================================================
# Opt-in — bấm nút "Tranh luận cùng AI" lúc luồng ĐANG chạy (cửa sổ bấm = lúc Tool 1
# chuẩn bị ngữ cảnh + judge chấm). Node debate đọc cờ này lúc vào.
# =============================================================================

def mark_optin(thread_id: str) -> None:
    _optin.add(thread_id)
    logger.info("Opt-in tranh luận cùng người học: thread=%s", thread_id)


def take_optin(thread_id: str) -> bool:
    """Đọc VÀ XOÁ cờ (one-shot). Gọi đúng 1 lần lúc vào node debate.

    Xoá ngay để cờ không dính sang LƯỢT CHAT SAU của cùng thread (state persist theo
    thread_id) -> người học sẽ bị hỏi ý kiến ở câu họ không hề bấm nút.
    """
    if thread_id in _optin:
        _optin.discard(thread_id)
        return True
    return False


def clear_optin(thread_id: str) -> None:
    """Dọn cờ khi lượt chat kết thúc mà debate KHÔNG chạy tới (vd route=factual).

    take_optin() lo ca thường; hàm này là lưới vét ở finalize cho ca cờ không ai đọc.
    """
    _optin.discard(thread_id)


# =============================================================================
# Phiên chờ
# =============================================================================

def open_session(thread_id: str, round_no: int, valid_arg_ids: set[str]) -> _Pending:
    """Mở phiên chờ cho 1 vòng. Node gọi TRƯỚC khi bắn event await_human ra FE."""
    p = _Pending(round=round_no, valid_arg_ids=set(valid_arg_ids))
    _sessions[thread_id] = p
    return p


def close_session(thread_id: str) -> None:
    """Đóng phiên. Sau lúc này mọi submit() đều 409 (đúng: không còn ai đọc queue)."""
    _sessions.pop(thread_id, None)


def is_waiting(thread_id: str) -> bool:
    return thread_id in _sessions


def waiting_round(thread_id: str) -> Optional[int]:
    """Đang chờ ở VÒNG nào (None = không chờ ai).

    Chỉ "có phiên hay không" là không đủ để phân biệt: giữa vòng 1 và vòng 2 có một
    khoảng phiên cũ ĐÃ đóng / phiên mới CHƯA mở, mà cả hai đều là "đang chờ" nếu chỉ
    nhìn is_waiting() -> dễ gửi nhầm tin vòng 2 vào queue vòng 1 vừa bị bỏ.
    """
    p = _sessions.get(thread_id)
    return p.round if p else None


def submit(thread_id: str, reply: Optional[HumanReply]) -> None:
    """Đẩy 1 tin (hoặc None = Kết thúc/Bỏ qua) cho node đang chờ.

    reply=None đi CHUNG đường với timeout: node thấy None thì thoát vòng lặp. Nhờ vậy
    "Bỏ qua" (chưa gửi tin nào) và "Kết thúc phản biện" (đã gửi n tin) là CÙNG một tín
    hiệu — khác nhau đúng ở chỗ đã có tin hay chưa, không cần thêm nhánh code nào.
    """
    p = _sessions.get(thread_id)
    if p is None:
        raise DebateNotWaiting(f"Không có phiên tranh luận nào đang chờ (thread={thread_id}).")
    if reply is not None:
        _validate(p, reply)
    p.queue.put_nowait(reply)


def _validate(p: _Pending, reply: HumanReply) -> None:
    if not (reply.message or "").strip():
        raise DebateInvalidReply("Nội dung phát biểu không được để trống.")
    if p.round != 2:
        return
    # Vòng 2 = phản biện -> bắt buộc nhắm vào một luận điểm CÓ THẬT trong bảng tin.
    if reply.stance not in STANCES:
        raise DebateInvalidReply(
            f"stance phải là một trong {STANCES}, nhận được {reply.stance!r}."
        )
    aid = normalize_arg_id(reply.target_arg_id)
    if aid not in p.valid_arg_ids:
        raise DebateInvalidReply(
            f"target_arg_id {reply.target_arg_id!r} không có trong bảng tin "
            f"({sorted(p.valid_arg_ids)})."
        )
    # Tự phản biện luận điểm của chính mình là vô nghĩa. 4 critic AI bị chặn ở _speak_r2
    # (`target == role` -> bỏ); người học đi đường khác nên phải chặn ở đây.
    if aid.startswith("human-"):
        raise DebateInvalidReply("Không thể phản biện chính luận điểm của bạn.")
