import json
import random
from pathlib import Path

#python scripts/data_prep/mix_s0_s4.py
def process_and_split_data():
    # 1. Cấu hình đường dẫn tương đối (để ai clone code về cũng chạy được)
    # Lấy thư mục gốc của project (chứa file thư mục scripts)
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    # Tạo thư mục processed nếu chưa có
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    
    # 2. Đọc và gom dữ liệu từ S0 đến S4
    print(" -> Đang đọc dữ liệu từ thư mục raw...")
    # Lặp qua các file từ s0.jsonl đến s4.jsonl
    for i in range(5):
        file_path = raw_dir / f"s{i}.jsonl"
        
        if not file_path.exists():
            print(f" -> Cảnh báo: Không tìm thấy file {file_path.name}. Bỏ qua.")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data_obj = json.loads(line)
                    all_data.append(data_obj)
                except json.JSONDecodeError:
                    print(f" -> Lỗi parse JSON tại {file_path.name}, dòng {line_number}")

    total_samples = len(all_data)
    if total_samples == 0:
        print(" -> Không có dữ liệu nào được nạp. Vui lòng kiểm tra lại thư mục raw.")
        return

    # 3. Shuffle
    print(f" -> Đã gộp thành công {total_samples} mẫu. Đang trộn ngẫu nhiên...")
    # Fix seed 
    random.seed(42) 
    random.shuffle(all_data)

    # 4. Tính toán index cắt mảng (Tỷ lệ 80% Train, 10% Val, 10% Test)
    train_end_idx = int(total_samples * 0.8)
    val_end_idx = int(total_samples * 0.9)

    train_data = all_data[:train_end_idx]
    val_data = all_data[train_end_idx:val_end_idx]
    test_data = all_data[val_end_idx:]

    # 5. Hàm helper để ghi file JSONL
    def save_jsonl(data_list, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. Xuất file
    print(" -> Đang lưu các file đã chia vào thư mục processed...")
    save_jsonl(train_data, processed_dir / "train.jsonl")
    save_jsonl(val_data, processed_dir / "dev.jsonl")
    save_jsonl(test_data, processed_dir / "test.jsonl")

    # 7. In báo cáo
    print("\n" + "="*40)
    print(" -> XỬ LÝ DỮ LIỆU HOÀN TẤT!")
    print(f"Tổng số mẫu: {total_samples}")
    print(f" - Train (80%): {len(train_data)} mẫu -> data/processed/train.jsonl")
    print(f" - Val   (10%): {len(val_data)} mẫu -> data/processed/dev.jsonl")
    print(f" - Test  (10%): {len(test_data)} mẫu -> data/processed/test.jsonl")
    print("="*40)

if __name__ == "__main__":
    process_and_split_data()

