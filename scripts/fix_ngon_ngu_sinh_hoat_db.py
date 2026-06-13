import os
import sys
import re
import unicodedata
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

project_root = r"d:\Project\SBA\rag-service"
sys.path.insert(0, os.path.join(project_root, "src"))

# Load config relative to project root
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

from db.mongo_client import connect_to_mongo, get_database

def remove_vietnamese_accents(text: str) -> str:
    """Helper to convert Vietnamese text to clean lowercase ASCII-like text for FTS."""
    if not text:
        return ""
    normalized = unicodedata.normalize('NFD', text)
    stripped = "".join([c for c in normalized if not unicodedata.combining(c)])
    stripped = stripped.replace('Đ', 'D').replace('đ', 'd')
    return re.sub(r'\s+', ' ', stripped).strip().lower()

def count_tokens(text: str) -> int:
    """Estimates the token count of a text using underthesea word tokenization."""
    if not text:
        return 0
    try:
        from underthesea import word_tokenize
        return len(word_tokenize(text))
    except Exception:
        return len(text.split())

def clean_ocr_typos(text: str) -> str:
    """Specific OCR and Vietnamese typography corrections."""
    if not text:
        return text
    # 1. H0ạt -> Hoạt
    text = re.sub(r'\bh0ạt\b', 'hoạt', text, flags=re.IGNORECASE)
    text = re.sub(r'\bH0ẠT\b', 'HOẠT', text)
    # 2. NBôn -> Ngôn
    text = re.sub(r'\bnbôn\b', 'ngôn', text, flags=re.IGNORECASE)
    text = re.sub(r'\bNBÔN\b', 'NGÔN', text)
    # 3. Ngứữ -> Ngữ
    text = re.sub(r'\bngứữ\b', 'ngữ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bNGỨỮ\b', 'NGỮ', text)
    return text

def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("=== DRY RUN MODE - No changes will be saved ===\n")
    else:
        print("=== LIVE MODE - Changes WILL be saved to database ===\n")
        
    connect_to_mongo()
    db = get_database()
    col = db["document_chunks"]
    
    # Target chunks
    query = {
        "is_active": True,
        "source_doc_id": "sach-giao-khoa-ngu-van-10-tap-1-co-ban",
        "metadata.ten_tac_pham": {"$in": ["I- NBÔN NGỮ SINH H0ẠT", "PHONG CÁCH NGÔN NGỨỮ SINH HOẠT"]}
    }
    
    chunks = list(col.find(query))
    print(f"Found {len(chunks)} chunks to update.")
    
    updated_count = 0
    for chunk in chunks:
        old_title = chunk["metadata"]["ten_tac_pham"]
        new_title = "PHONG CÁCH NGÔN NGỮ SINH HOẠT"
        
        old_chunk_id = chunk["chunk_id"]
        # Fix misspelled chunk ID if present
        new_chunk_id = old_chunk_id
        if "phung_bach_nbon_n6u_sinh_huat" in old_chunk_id:
            new_chunk_id = old_chunk_id.replace("phung_bach_nbon_n6u_sinh_huat", "phong_cach_ngon_ngu_sinh_hoat")
            
        old_content = chunk["content"]
        new_content = clean_ocr_typos(old_content)
        
        print(f"\nChunk ID: '{old_chunk_id}' -> '{new_chunk_id}'")
        print(f"  Title: '{old_title}' -> '{new_title}'")
        if old_content != new_content:
            print("  [Content corrected]")
            print(f"    Before: {repr(old_content[:100])}")
            print(f"    After:  {repr(new_content[:100])}")
            
        if not dry_run:
            update_fields = {
                "metadata.ten_tac_pham": new_title,
                "chunk_id": new_chunk_id,
                "content": new_content,
                "char_count": len(new_content),
                "token_count": count_tokens(new_content),
                "search_text": remove_vietnamese_accents(new_content)
            }
            col.update_one({"_id": chunk["_id"]}, {"$set": update_fields})
            updated_count += 1
            
    if not dry_run:
        print(f"\nSuccessfully updated {updated_count} chunks.")
    else:
        print("\nDry run finished. No updates applied.")

if __name__ == "__main__":
    main()
