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
from core.embedder import Embedder

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

def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("=== DRY RUN MODE - No changes will be saved ===\n")
    else:
        print("=== LIVE MODE - Changes WILL be saved to database ===\n")
        
    connect_to_mongo()
    db = get_database()
    col = db["document_chunks"]
    
    # 1. Find the chunk to split
    target_chunk_id = "2_suu_tam_nhung_bai_ca_dao_hai_huoc_phe_phan_thoi_luoi_nhac_le_la_an_qua_001"
    query = {
        "chunk_id": target_chunk_id,
        "is_active": True,
        "source_doc_id": "sach-giao-khoa-ngu-van-10-tap-1-co-ban"
    }
    
    chunk = col.find_one(query)
    if not chunk:
        print(f"Error: Chunk '{target_chunk_id}' not found in database!")
        return
        
    content = chunk["content"]
    print("Found original chunk content:")
    print("-" * 50)
    print(content)
    print("-" * 50)
    
    # Locate split point
    split_marker = "0ọc THÊM"
    if split_marker not in content:
        split_marker = "Đọc thêm"
        if "đọc thêm" in content.lower():
            # case insensitive split pattern
            parts = re.split(r'(?i)0ọc\s+thêm|đọc\s+thêm', content, 1)
        else:
            print("Error: Split marker not found in chunk content!")
            return
    else:
        parts = content.split(split_marker, 1)
        
    if len(parts) < 2:
        print("Error: Could not split content into two parts!")
        return
        
    chunk1_content = parts[0].strip()
    
    # Define clean heading content for chunk 2
    chunk2_content = "ĐỌC THÊM: LỜI TIỄN DẶN\n(Trích Tiễn dặn người yêu — truyện thơ dân tộc Thái)"
    
    print("\nProposed Split:")
    print(f"Part 1 (Page {chunk['position']['page']}, index {chunk['position']['chunk_index']}):")
    print("-" * 30)
    print(chunk1_content)
    print("-" * 30)
    print(f"Part 2 (Page 93, new index {chunk['position']['chunk_index'] + 1}):")
    print("-" * 30)
    print(chunk2_content)
    print("-" * 30)
    
    if dry_run:
        print("\nDry run completed. No database updates performed.")
        return
        
    print("\nInitializing Embedder to calculate embeddings...")
    embedder = Embedder()
    
    print("Embedding Part 1...")
    emb_part1 = embedder.embed_query(chunk1_content)
    print("Embedding Part 2...")
    emb_part2 = embedder.embed_query(chunk2_content)
    
    # Step 1: Shift existing indexes and update total_chunks
    original_idx = chunk["position"]["chunk_index"]
    target_doc = chunk["source_doc_id"]
    
    print(f"Shifting index >= {original_idx + 1} for document '{target_doc}'...")
    shift_res = col.update_many(
        {
            "source_doc_id": target_doc,
            "is_active": True,
            "position.chunk_index": {"$gte": original_idx + 1}
        },
        {"$inc": {"position.chunk_index": 1}}
    )
    print(f"Shifted {shift_res.modified_count} chunks.")
    
    print(f"Incrementing total_chunks for document '{target_doc}'...")
    total_res = col.update_many(
        {
            "source_doc_id": target_doc,
            "is_active": True
        },
        {"$inc": {"position.total_chunks": 1}}
    )
    print(f"Updated total_chunks for {total_res.modified_count} chunks.")
    
    # Step 2: Update original chunk (Part 1)
    print(f"Updating original chunk '{target_chunk_id}'...")
    col.update_one(
        {"_id": chunk["_id"]},
        {"$set": {
            "content": chunk1_content,
            "char_count": len(chunk1_content),
            "token_count": count_tokens(chunk1_content),
            "search_text": remove_vietnamese_accents(chunk1_content),
            "embedding": emb_part1
        }}
    )
    print("Updated original chunk.")
    
    # Step 3: Insert new chunk (Part 2)
    new_chunk_id = "doc_them_loi_tien_dan_000"
    print(f"Inserting new chunk '{new_chunk_id}' at index {original_idx + 1}...")
    
    # Copy original metadata and update fields
    new_metadata = chunk["metadata"].copy()
    new_metadata["ten_tac_pham"] = "ĐỌC THÊM: LỜI TIỄN DẶN"
    
    new_position = {
        "page": 93,
        "chunk_index": original_idx + 1,
        "total_chunks": chunk["position"]["total_chunks"] + 1
    }
    
    new_chunk = {
        "chunk_id": new_chunk_id,
        "source_doc_id": target_doc,
        "content": chunk2_content,
        "content_type": "prose",
        "position": new_position,
        "metadata": new_metadata,
        "token_count": count_tokens(chunk2_content),
        "char_count": len(chunk2_content),
        "has_overlap": False,
        "embedding": emb_part2,
        "search_text": remove_vietnamese_accents(chunk2_content),
        "model_version": chunk["model_version"],
        "is_active": True
    }
    
    col.insert_one(new_chunk)
    print("Inserted new chunk.")
    
    # Step 4: Update all chunks in pages 93-95 to have 'ten_tac_pham' = "ĐỌC THÊM: LỜI TIỄN DẶN"
    print("Updating metadata.ten_tac_pham for 'Đọc thêm' section chunks (indexes 360 to 373)...")
    title_res = col.update_many(
        {
            "source_doc_id": target_doc,
            "is_active": True,
            "position.chunk_index": {"$gte": original_idx + 1, "$lte": original_idx + 14}
        },
        {"$set": {"metadata.ten_tac_pham": "ĐỌC THÊM: LỜI TIỄN DẶN"}}
    )
    print(f"Updated ten_tac_pham for {title_res.modified_count} chunks.")
    print("\nDatabase remediation completed successfully!")

if __name__ == "__main__":
    main()
