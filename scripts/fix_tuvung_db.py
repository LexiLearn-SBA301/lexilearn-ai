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
    
    # 1. Update ten_tac_pham from "TRONG BÀI VĂN TỰ SỰ" to "CHỌN SỰ VIỆC, CHI TIẾT TIÊU BIỀU TRONG BÀI VĂN TỰ SỰ"
    res1 = col.update_many(
        {"is_active": True, "metadata.ten_tac_pham": "TRONG BÀI VĂN TỰ SỰ"},
        {"$set": {"metadata.ten_tac_pham": "CHỌN SỰ VIỆC, CHI TIẾT TIÊU BIỀU TRONG BÀI VĂN TỰ SỰ"}}
    )
    print(f"Updated {res1.modified_count} chunks with title 'TRONG BÀI VĂN TỰ SỰ' -> 'CHỌN SỰ VIỆC, CHI TIẾT TIÊU BIỀU TRONG BÀI VĂN TỰ SỰ'.")
    
    # 2. Update chunk_ids to fix spelling errors (gaeh_chun -> cach_chon, ghi_tiet -> chi_tiet)
    # Let's find chunks with chunk_id containing "gaeh_chun" or "ghi_tiet"
    chunks = list(col.find(
        {"is_active": True, "chunk_id": {"$regex": "gaeh_chun|ghi_tiet", "$options": "i"}}
    ))
    
    print(f"Found {len(chunks)} chunks with spelling errors in chunk_id.")
    
    updated_count = 0
    for chunk in chunks:
        old_id = chunk["chunk_id"]
        # Replace spelling errors
        new_id = old_id.replace("gaeh_chun", "cach_chon").replace("ghi_tiet", "chi_tiet")
        
        col.update_one(
            {"_id": chunk["_id"]},
            {"$set": {"chunk_id": new_id}}
        )
        print(f"  Updated chunk_id: '{old_id}' -> '{new_id}'")
        updated_count += 1
        
    print(f"Successfully fixed spelling in {updated_count} chunk_ids.")

if __name__ == "__main__":
    main()
