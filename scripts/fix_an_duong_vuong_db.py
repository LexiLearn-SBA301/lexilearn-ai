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
    
    # 1. Update ten_tac_pham in metadata
    old_title = "VÀ MÌ CHÂU - TRỌNG THUỶ"
    new_title = "TRUYỆN AN DƯƠNG VƯƠNG VÀ MỊ CHÂU - TRỌNG THỦY"
    
    # Find active chunks with this title
    query = {
        "is_active": True,
        "metadata.ten_tac_pham": old_title
    }
    
    chunks = list(col.find(query))
    print(f"Found {len(chunks)} chunks with title '{old_title}'.")
    
    if len(chunks) == 0:
        print("No chunks need updating.")
        return
        
    result = col.update_many(
        query,
        {"$set": {"metadata.ten_tac_pham": new_title}}
    )
    
    print(f"Successfully updated {result.modified_count} chunks to '{new_title}'.")

if __name__ == "__main__":
    main()
