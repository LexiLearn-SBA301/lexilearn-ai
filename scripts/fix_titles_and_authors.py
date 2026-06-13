import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

project_root = r"d:\Project\SBA\rag-service"
sys.path.insert(0, os.path.join(project_root, "src"))

from db.mongo_client import connect_to_mongo, get_database

def main():
    connect_to_mongo()
    db = get_database()
    col = db["document_chunks"]
    
    # Fix 1: UY-LÍT-XƠ TRỞ VỀ HÔ-ME-RƠ
    res1 = col.update_many(
        {"is_active": True, "metadata.ten_tac_pham": "UY-LÍT-XƠ TRỞ VỀ HÔ-ME-RƠ"},
        {"$set": {
            "metadata.ten_tac_pham": "UY-LÍT-XƠ TRỞ VỀ",
            "metadata.tac_gia": "Hô-me-rơ"
        }}
    )
    print(f"Updated {res1.modified_count} chunks of 'UY-LÍT-XƠ TRỞ VỀ HÔ-ME-RƠ' to 'UY-LÍT-XƠ TRỞ VỀ' with author 'Hô-me-rơ'.")
    
    # Fix 2: CẢNH NGÀY HÈ_ NGUYỄN TRÃI
    res2 = col.update_many(
        {"is_active": True, "metadata.ten_tac_pham": "CẢNH NGÀY HÈ_ NGUYỄN TRÃI"},
        {"$set": {
            "metadata.ten_tac_pham": "CẢNH NGÀY HÈ",
            "metadata.tac_gia": "Nguyễn Trãi"
        }}
    )
    print(f"Updated {res2.modified_count} chunks of 'CẢNH NGÀY HÈ_ NGUYỄN TRÃI' to 'CẢNH NGÀY HÈ' with author 'Nguyễn Trãi'.")
    
    # Fix 3: Garbage title Z ^ Z% ^ F4 -> CHỌN SỰ VIỆC, CHI TIẾT TIÊU BIỀU TRONG BÀI VĂN TỰ SỰ
    res3 = col.update_many(
        {"is_active": True, "metadata.ten_tac_pham": "Z ^ Z% ^ F4"},
        {"$set": {
            "metadata.ten_tac_pham": "CHỌN SỰ VIỆC, CHI TIẾT TIÊU BIỀU TRONG BÀI VĂN TỰ SỰ"
        }}
    )
    print(f"Updated {res3.modified_count} chunks of 'Z ^ Z% ^ F4' to 'CHỌN SỰ VIỆC, CHI TIẾT TIÊU BIỀU TRONG BÀI VĂN TỰ SỰ'.")

if __name__ == "__main__":
    main()
