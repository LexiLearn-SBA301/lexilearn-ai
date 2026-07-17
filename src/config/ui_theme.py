"""
UI theme cho StreamEvent — trả kèm màu / severity / lane để FE render timeline.

Nguyên tắc (xem block_diagram/streaming_realtime_plan.md §4.6):
- `variant` / `severity` / `group` là TOKEN NGỮ NGHĨA — nguồn sự thật để FE tự map palette
  (đổi theme / dark mode / rebrand KHÔNG phải đụng BE).
- `color` chỉ là màu MẶC ĐỊNH tiện dụng; FE có palette riêng thì override theo `variant`.

File này CỐ TÌNH không import gì từ state_schema (chỉ key bằng chuỗi `.value`) để tránh
circular import với EventEmitter — nơi gọi `ui_meta(...)`.
"""
from __future__ import annotations

from typing import Any, Optional

# 4 màu critic = bộ CATEGORICAL, hue tách nhau, cố tình TRÁNH dải green/amber/red (dành
# cho verdict judge) để không nhầm "màu critic" với "màu phán quyết"; đều mid-tone nên
# đọc được trên cả nền sáng lẫn tối.
_CRITIC: dict[str, str] = {
    "hinh_thuc": "#14b8a6",   # teal
    "lich_su":   "#3b82f6",   # blue
    "tam_ly":    "#8b5cf6",   # violet
    "tiep_nhan": "#ec4899",   # pink
    # Người học — CỐ TÌNH lấy đỏ-gạch thương hiệu (#ab3429), cùng màu bong bóng user ở
    # box chat: xuyên suốt app màu này đã có nghĩa "bạn". Khác hẳn 4 hue lạnh của critic
    # nên nhìn phát ra ngay ai là người thật; cũng đủ trầm để không lẫn với đỏ cảnh báo
    # (#ef4444) của verdict.
    "human":     "#ab3429",   # brick (brand)
}

# severity -> màu mặc định
_SEVERITY_COLOR: dict[str, str] = {
    "info":    "#64748b",   # slate
    "success": "#22c55e",   # green
    "warning": "#f59e0b",   # amber
    "error":   "#ef4444",   # red
}

# event_type.value -> khung UI (khi KHÔNG phải critic_turn / judge — 2 cái đó tính riêng)
_EVENT: dict[str, dict[str, str]] = {
    "intent":         {"variant": "supervisor", "severity": "info",    "group": "intent"},
    "route":          {"variant": "supervisor", "severity": "info",    "group": "intent"},
    "status":         {"variant": "status",     "severity": "info",    "group": "intent"},
    "thinking":       {"variant": "status",     "severity": "info",    "group": "intent"},
    "retrieval":      {"variant": "retrieval",  "severity": "info",    "group": "retrieval"},
    "bulletin":       {"variant": "bulletin",   "severity": "info",    "group": "debate"},
    "await_human":    {"variant": "await_human", "severity": "info",   "group": "debate"},
    "debate_lock":    {"variant": "status",     "severity": "info",    "group": "debate"},
    "retry":          {"variant": "judge",      "severity": "warning", "group": "judge"},
    "citation_check": {"variant": "citation",   "severity": "info",    "group": "final"},
    "token":          {"variant": "essay",      "severity": "info",    "group": "final"},
    "error":          {"variant": "error",      "severity": "error",   "group": "final"},
    "done":           {"variant": "done",       "severity": "success", "group": "final"},
}

# verdict.value -> severity (cho event JUDGE)
_VERDICT_SEVERITY: dict[str, str] = {
    "pass": "success", "approve": "success", "retry": "warning", "reject": "error",
}

_DEFAULT = {"variant": "status", "severity": "info", "group": "intent"}


def _val(x: Any) -> str:
    """Lấy .value nếu là Enum, còn lại ép str."""
    return x.value if hasattr(x, "value") else str(x)


def ui_meta(event_type: Any, *, role: Optional[Any] = None,
            verdict: Optional[Any] = None) -> dict[str, str]:
    """Trả khối `payload["ui"]` = {variant, color, severity, group}.

    - role != None  -> event của 1 critic (màu theo role).
    - event_type == "judge" -> màu/severity theo `verdict` (pass/approve=success,
      retry=warning, reject=error).
    - còn lại -> tra `_EVENT`.
    """
    if role is not None:
        rv = _val(role)
        return {"variant": rv, "color": _CRITIC.get(rv, _SEVERITY_COLOR["info"]),
                "severity": "info", "group": "debate"}

    et = _val(event_type)
    if et == "judge":
        sev = _VERDICT_SEVERITY.get(_val(verdict) if verdict is not None else "pass", "info")
        return {"variant": "judge", "color": _SEVERITY_COLOR[sev], "severity": sev,
                "group": "judge"}

    base = _EVENT.get(et, _DEFAULT)
    sev = base["severity"]
    return {"variant": base["variant"], "color": _SEVERITY_COLOR.get(sev, "#64748b"),
            "severity": sev, "group": base["group"]}
