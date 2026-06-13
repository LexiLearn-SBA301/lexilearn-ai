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
    
    # 1. Update chunks with title "T" on pages 54-55
    res1 = col.update_many(
        {
            "is_active": True, 
            "metadata.ten_tac_pham": "T",
            "position.page": {"$in": [54, 55]}
        },
        {"$set": {
            "metadata.ten_tac_pham": "TRẢ BÀI LÀM VĂN SỐ 1",
            "metadata.tac_gia": "Bộ Giáo Dục và Đào Tạo"
        }}
    )
    print(f"Updated {res1.modified_count} chunks on page 54-55 to 'TRẢ BÀI LÀM VĂN SỐ 1'.")
    
    # 2. Update chunks with title "T" on pages >= 56
    res2 = col.update_many(
        {
            "is_active": True,
            "metadata.ten_tac_pham": "T",
            "position.page": {"$gte": 56}
        },
        {"$set": {
            "metadata.ten_tac_pham": "RA-MA BUỘC TỘI",
            "metadata.tac_gia": "Van-mi-ki"
        }}
    )
    print(f"Updated {res2.modified_count} chunks on page >= 56 to 'RA-MA BUỘC TỘI' with author 'Van-mi-ki'.")

if __name__ == "__main__":
    main()
