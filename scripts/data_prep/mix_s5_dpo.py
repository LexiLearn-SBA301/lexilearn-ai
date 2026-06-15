import json
import random
from pathlib import Path

# python scripts/data_prep/mix_s5_dpo.py
#
# Mix data cho phase DPO (S5). KHÁC mix_s0_s4.py ở schema:
#   - SFT (s0-s4): mỗi mẫu là {messages: [...]}        -> train/dev/test.jsonl
#   - DPO (s5)   : mỗi mẫu là {prompt, chosen, rejected} -> dpo_train/dev/test.jsonl
# Vì khác schema nên KHÔNG gộp chung được; phải có script split riêng.
# Việc gắn system prompt + render ChatML để parity được làm Ở BƯỚC TRAIN
# (trong train_s5_dpo_kaggle.py), KHÔNG đụng vào nội dung ở đây — chỉ shuffle + split.
def process_and_split_dpo_data():
    # 1. Đường dẫn tương đối (clone về là chạy được)
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    # 2. Đọc s5.jsonl (toàn bộ là type="preference")
    s5_path = raw_dir / "s5.jsonl"
    if not s5_path.exists():
        print(f" -> Lỗi: Không tìm thấy {s5_path}. Hãy kiểm tra lại data/raw/.")
        return

    all_data = []
    print(" -> Đang đọc dữ liệu DPO từ s5.jsonl...")
    with open(s5_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f" -> Lỗi parse JSON tại s5.jsonl, dòng {line_number}")
                continue
            # Chỉ nhận đúng cặp preference (phòng dòng lạ lẫn vào).
            if {"prompt", "chosen", "rejected"} <= obj.keys():
                all_data.append(obj)
            else:
                print(f" -> Bỏ qua dòng {line_number}: thiếu prompt/chosen/rejected")

    total_samples = len(all_data)
    if total_samples == 0:
        print(" -> Không có cặp preference hợp lệ nào. Dừng.")
        return

    # 3. Shuffle (CÙNG seed 42 như mix SFT để nhất quán quy ước repo)
    print(f" -> Đã nạp {total_samples} cặp preference. Đang trộn ngẫu nhiên...")
    random.seed(42)
    random.shuffle(all_data)

    # 4. Split 80% train / 10% dev / 10% test (giống tỷ lệ SFT)
    train_end_idx = int(total_samples * 0.8)
    val_end_idx = int(total_samples * 0.9)

    train_data = all_data[:train_end_idx]
    val_data = all_data[train_end_idx:val_end_idx]
    test_data = all_data[val_end_idx:]

    # 5. Helper ghi JSONL
    def save_jsonl(data_list, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. Xuất file — TÊN RIÊNG (dpo_*) để KHÔNG đè file SFT (train/dev/test.jsonl)
    print(" -> Đang lưu các file DPO đã chia vào thư mục processed...")
    save_jsonl(train_data, processed_dir / "dpo_train.jsonl")
    save_jsonl(val_data, processed_dir / "dpo_dev.jsonl")
    save_jsonl(test_data, processed_dir / "dpo_test.jsonl")

    # 7. Báo cáo
    print("\n" + "=" * 40)
    print(" -> XỬ LÝ DATA DPO HOÀN TẤT!")
    print(f"Tổng số cặp: {total_samples}")
    print(f" - Train (80%): {len(train_data)} cặp -> data/processed/dpo_train.jsonl")
    print(f" - Dev   (10%): {len(val_data)} cặp -> data/processed/dpo_dev.jsonl")
    print(f" - Test  (10%): {len(test_data)} cặp -> data/processed/dpo_test.jsonl")
    print("=" * 40)


if __name__ == "__main__":
    process_and_split_dpo_data()
