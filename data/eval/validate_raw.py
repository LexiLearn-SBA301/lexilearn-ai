import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal

# In tiếng Việt ra console không bị lỗi encoding trên Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ==========================================
# 1. CONSTANTS & UTILS (Logic chuẩn hóa)
# ==========================================
SYSTEM_FIXED = "Bạn là trợ lý phân tích văn học tiếng Việt. Chỉ phân tích dựa trên ngữ liệu được cung cấp trong [Ngữ liệu]. Mọi dẫn chứng phải trích nguyên văn từ ngữ liệu, đặt trong ngoặc kép. Nếu ngữ liệu không đủ căn cứ để trả lời, hãy nói rõ điều đó thay vì suy đoán."

def norm(s: str) -> str:
    """đưa về chuẩn NFC và lowercase, xóa khoảng trắng thừa"""
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"\s+", " ", s).strip()

def check_hallucinated_quotes(context: str, output: str) -> List[str]:
    """Tìm tất cả ngoặc kép trong output, kiểm tra xem có nằm trong context không
        EX: Context: "Thân em như tấm lụa đào..."
            Output: "Tác giả ví "thân em" như "tấm lụa đào""
        -> ['thân em', 'tấm lụa đào']
    """
    ctx_n = norm(context)
    quotes = re.findall(r'"([^"]+)"', output) # quotes là list các đoạn nằm trong ngoặc kép
    return [q for q in quotes if norm(q) not in ctx_n]  # trả về list các data không clean

# ==========================================
# 2. DTO SCHEMAS (Định nghĩa cấu trúc)
# ==========================================
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class SftSchema(BaseModel):
    """Schema dành cho S0, S1, S2, S3, S4 (Dạng hội thoại)"""
    id: str = Field(pattern=r"^S[0-4]_\d{4}$") # EX: S2_0001
    skill: Literal["S0", "S1", "S2", "S3", "S4"] #Literal == @Pattern (java)
    type: Literal["normal", "robustness"]
    messages: List[Message]

    @model_validator(mode="after") # mode="after" sau khi validation các fields ở trên mới chạy hàm này
    def validate_sft_logic(self) -> "SftSchema":
        msgs = self.messages
        if len(msgs) != 3:
            raise ValueError("Phải có chính xác 3 lượt: system, user, assistant")

        sys_msg, user_msg, ast_msg = msgs[0], msgs[1], msgs[2]

        # 1. Check thứ tự role
        if sys_msg.role != "system" or user_msg.role != "user" or ast_msg.role != "assistant":
            raise ValueError("Thứ tự role sai. Phải là: system -> user -> assistant")

        # 2. Check System Prompt có bị sai chữ nào không
        if sys_msg.content != SYSTEM_FIXED:
            raise ValueError("System prompt bị sai khác so với quy định cố định!")

        # 3. Check định dạng User prompt
        if "[Ngữ liệu]" not in user_msg.content or "[Đề]" not in user_msg.content:
            raise ValueError("User content thiếu thẻ [Ngữ liệu] hoặc [Đề]")

        # 4. Check "0 bịa dẫn chứng"
        bia_quotes = check_hallucinated_quotes(user_msg.content, ast_msg.content)
        if bia_quotes:
            raise ValueError(f"PHÁT HIỆN BỊA DẪN CHỨNG: {bia_quotes}")

        return self

class DpoSchema(BaseModel):
    """Schema dành riêng cho S5 (Cặp tốt/xấu)"""
    id: str = Field(pattern=r"^S5_\d{4}$") # EX: S5_0001
    skill: Literal["S5"]
    type: Literal["preference"]
    prompt: str
    chosen: str
    rejected: str

    @model_validator(mode="after")
    def validate_dpo_logic(self) -> "DpoSchema":
        # 1. Check định dạng Prompt
        if "[Ngữ liệu]" not in self.prompt or "[Đề]" not in self.prompt:
            raise ValueError("Prompt thiếu thẻ [Ngữ liệu] hoặc [Đề]")

        # 2. Check "0 bịa dẫn chứng" CHỈ TRÊN bài CHOSEN (bài rejected cố tình bịa nên bỏ qua)
        bia_quotes = check_hallucinated_quotes(self.prompt, self.chosen)
        if bia_quotes:
            raise ValueError(f"BÀI CHOSEN BỊA DẪN CHỨNG: {bia_quotes}")

        return self

# ==========================================
# 3. PHÂN LOẠI LỖI (Diagnostics cho report)
# ==========================================
# Mức độ ưu tiên gán nhãn cho cả dòng: càng đứng trước càng "nặng"
PRIORITY = ["OTHER", "SYSTEM", "ABSENT", "PAIRING", "TEMPLATE", "LINEBREAK"]
LABEL = {
    "OTHER":    "Lỗi cấu trúc/định dạng (id, role, thiếu thẻ...)",
    "SYSTEM":   "System prompt sai khác",
    "ABSENT":   "Cụm/diễn giải KHÔNG có trong ngữ liệu (sửa nội dung)",
    "PAIRING":  "Ghép cặp bằng '/' hoặc '-' (diễn giải, bỏ ngoặc kép)",
    "TEMPLATE": "Mẫu khái quát có dấu '...' (bỏ ngoặc kép)",
    "LINEBREAK":"Thơ đúng nhưng nối dòng bằng ' / ' (chỉ sửa format)",
}
SEV = {
    "OTHER": "Cấu trúc", "SYSTEM": "Cấu trúc",
    "ABSENT": "Nội dung (nặng)", "PAIRING": "Nội dung",
    "TEMPLATE": "Format", "LINEBREAK": "Format (nhẹ)",
}
FIX = {
    "OTHER":    "Xem chi tiết lỗi ở cột bên; sửa đúng schema (id ^S[0-5]_\\d{4}$, đủ thẻ [Ngữ liệu]/[Đề], 3 role).",
    "SYSTEM":   "Dán lại đúng SYSTEM_FIXED (sai/thừa/thiếu ký tự).",
    "ABSENT":   "Bỏ ngoặc kép (diễn giải), hoặc trích đúng nguyên văn từ ngữ liệu.",
    "PAIRING":  "Bỏ ngoặc kép phần ghép cặp, hoặc trích riêng từng vế đúng nguyên văn.",
    "TEMPLATE": "Bỏ ngoặc kép cho công thức/mẫu; chỉ dùng ngoặc kép khi trích nguyên văn.",
    "LINEBREAK":"Đổi ' / ' thành xuống dòng thật (\\n) đúng như trong ngữ liệu.",
}
TAG = {"LINEBREAK": "⟂", "TEMPLATE": "…", "PAIRING": "/", "ABSENT": "✗"}

def classify_quote(q: str, ctx_n: str) -> str:
    """Phân loại lý do 1 quote bị cờ (chỉ dùng cho report)."""
    if norm(q) in ctx_n:
        return "OK"
    if "..." in q or "…" in q:
        return "TEMPLATE"                 # mẫu khái quát, không phải trích nguyên văn
    if norm(q.replace("/", " ")) in ctx_n:
        return "LINEBREAK"                # nội dung ĐÚNG, chỉ thay xuống dòng bằng ' / '
    if "/" in q or "-" in q:
        return "PAIRING"                  # ghép cặp đối -> diễn giải
    return "ABSENT"                       # cụm thực sự không có trong ngữ liệu

def diagnose(raw: dict, is_s5: bool, exc: Exception) -> dict:
    """Từ exception + dữ liệu thô, suy ra loại lỗi để đưa vào report."""
    msg = str(exc)
    low = msg.lower()
    if "system prompt" in low:
        return {"cat": "SYSTEM", "quotes": [], "detail": "System prompt lệch SYSTEM_FIXED"}
    if "bịa" in low:  # lỗi dẫn chứng (BỊA DẪN CHỨNG / CHOSEN BỊA DẪN CHỨNG)
        if is_s5:
            ctx, out = raw["prompt"], raw["chosen"]
        else:
            ctx, out = raw["messages"][1]["content"], raw["messages"][2]["content"]
        ctx_n = norm(ctx)
        bad = check_hallucinated_quotes(ctx, out)
        cats = {q: classify_quote(q, ctx_n) for q in bad}
        line_cat = next((c for c in PRIORITY if c in cats.values()), "ABSENT")
        return {"cat": line_cat, "quotes": [(q, cats[q]) for q in bad], "detail": ""}
    # còn lại: lỗi cấu trúc/schema -> lấy gọn message của Pydantic
    parts = [l.strip() for l in msg.split("\n")[1:] if l.strip() and "further information" not in l]
    detail = " | ".join(parts)[:200] if parts else msg.split("\n")[0]
    return {"cat": "OTHER", "quotes": [], "detail": detail}

# ==========================================
# 4. MAIN RUNNER (Quét thư mục + xuất report)
# ==========================================
def validate_all_files():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    report_path = base_dir / "data" / "eval" / "validation_errors_report.md"

    if not raw_dir.exists():
        print(f"-> Không tìm thấy thư mục {raw_dir}")
        return

    total_passed = 0
    rows = []  # (file, line, id, info_dict)

    print("-> BẮT ĐẦU VALIDATE DỮ LIỆU THÔ...")
    print("-" * 50)

    for file_path in sorted(raw_dir.glob("*.jsonl")):
        file_name = file_path.name
        is_s5 = file_name.startswith("s5")
        SchemaClass = DpoSchema if is_s5 else SftSchema

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue

                try:
                    raw_dict = json.loads(line)
                    SchemaClass(**raw_dict)  # ** unpack map fields vào schema, lỗi thì throw
                    total_passed += 1
                except Exception as e:
                    try:
                        raw_dict = json.loads(line)
                    except Exception:
                        raw_dict = {}
                    info = diagnose(raw_dict, is_s5, e)
                    rows.append((file_name, line_num, raw_dict.get("id", "?"), info))

    total_failed = len(rows)
    total = total_passed + total_failed

    # ---------- REPORTING (console) ----------
    print(f"\n -> TỔNG KẾT VALIDATION:")
    print(f" -> Hợp lệ: {total_passed} mẫu")
    print(f" -> Lỗi:    {total_failed} mẫu")

    cnt = Counter(r[3]["cat"] for r in rows)
    if total_failed > 0:
        print(f"\n -> PHÂN LOẠI LỖI:")
        for c in PRIORITY:
            if cnt.get(c):
                print(f"    - {c:9s}: {cnt[c]:>3}  ({LABEL[c]})")
        print(f"\n -> CHI TIẾT: xem {report_path}")
        print(" -> CẢNH BÁO: Không được chạy file mix_s0_s4.py khi vẫn còn lỗi!")
    else:
        print("\n -> TUYỆT VỜI! 100% dữ liệu đạt chuẩn. Bạn có thể tiến hành trộn data.")

    # ---------- REPORTING (markdown) ----------
    write_report(report_path, rows, total_passed, total_failed, total, cnt)


def write_report(report_path: Path, rows: list, passed: int, failed: int, total: int, cnt: Counter):
    md = []
    md.append("# Báo cáo validate dữ liệu thô\n")
    md.append(f"Tổng **{total}** mẫu · Hợp lệ **{passed}** · Lỗi **{failed}**. "
              "Sinh tự động bởi `data/eval/validate_raw.py`.\n")

    if failed == 0:
        md.append("> ✅ 100% dữ liệu đạt chuẩn — có thể chạy `mix_s0_s4.py`.\n")
        report_path.write_text("\n".join(md), encoding="utf-8")
        return

    md.append("## 1. Tổng hợp theo loại lỗi\n")
    md.append("| Loại lỗi | Số dòng | Mức độ | Cách sửa |")
    md.append("|---|---|---|---|")
    for c in PRIORITY:
        if cnt.get(c):
            md.append(f"| {LABEL[c]} | {cnt[c]} | {SEV[c]} | {FIX[c]} |")
    md.append("")
    md.append("> Quy tắc gốc (`check_hallucinated_quotes`): mọi chuỗi trong `\"...\"` ở câu trả lời "
              "phải là chuỗi con (sau khi NFC + lowercase + gọn khoảng trắng) của `[Ngữ liệu]`. "
              "Checker là so khớp chuỗi thuần — không hiểu thành ngữ, không hiểu xuống dòng.\n")

    md.append("## 2. Chi tiết từng dòng\n")
    md.append("| File | Dòng | ID | Loại | Quote bị cờ / Chi tiết | Hướng xử lý |")
    md.append("|---|---|---|---|---|---|")
    for name, ln, rid, info in rows:
        cat = info["cat"]
        if info["quotes"]:
            parts = []
            for q, qc in info["quotes"]:
                parts.append(f"`{q.replace('|', chr(92) + '|')}` {TAG.get(qc, '?')}")
            cell = "<br>".join(parts)
        else:
            cell = "_" + info["detail"].replace("|", chr(92) + "|") + "_"
        md.append(f"| {name} | {ln} | {rid} | {LABEL[cat]} | {cell} | {FIX[cat]} |")
    md.append("")
    md.append("Ký hiệu cột Quote: `⟂` = thơ đúng chữ chỉ sai dấu nối dòng · `…` = mẫu khái quát có dấu ... · "
              "`/` = ghép cặp bằng /,- · `✗` = cụm không có trong ngữ liệu.\n")

    report_path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    validate_all_files()
