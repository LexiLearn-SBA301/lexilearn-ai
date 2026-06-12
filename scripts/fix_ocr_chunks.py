"""
Fix OCR v->u errors in existing database chunks.
This script corrects the chunks that were already ingested with Tesseract OCR errors.
It also regenerates the search_text field.

Usage: .venv\Scripts\python scripts\fix_ocr_chunks.py [--dry-run]
"""
import sys
import os
import re
import unicodedata

# Set up python path relative to the script location to allow execution from any Cwd
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
# Load config relative to project root
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

from db.mongo_client import connect_to_mongo, get_database
from core.pdf_reader import PDFReader


def remove_vietnamese_accents(text: str) -> str:
    """Helper to convert Vietnamese text to clean lowercase ASCII-like text for FTS."""
    if not text:
        return ""
    normalized = unicodedata.normalize('NFD', text)
    stripped = "".join([c for c in normalized if not unicodedata.combining(c)])
    stripped = stripped.replace('Đ', 'D').replace('đ', 'd')
    return re.sub(r'\s+', ' ', stripped).strip().lower()


def main():
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        print("=== DRY RUN MODE - No changes will be saved ===\n")
    else:
        print("=== LIVE MODE - Changes WILL be saved to database ===\n")
    
    connect_to_mongo()
    db = get_database()
    collection = db["document_chunks"]
    
    # Create a PDFReader instance to use its _fix_ocr_vietnamese method
    reader = PDFReader()
    
    # Find all active chunks
    chunks = list(collection.find(
        {"is_active": True},
        {"_id": 1, "chunk_id": 1, "content": 1, "source_doc_id": 1}
    ))
    
    print(f"Total active chunks to scan: {len(chunks)}")
    
    fixed_count = 0
    total_corrections = 0
    
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content:
            continue
        
        # Apply OCR fix
        corrected = reader._fix_ocr_vietnamese(content)
        
        if corrected != content:
            fixed_count += 1
            # Count the number of differences (rough estimate)
            diff_count = sum(1 for a, b in zip(content, corrected) if a != b)
            total_corrections += diff_count
            
            print(f"\n  [{chunk['chunk_id']}] ({chunk.get('source_doc_id', 'N/A')})")
            
            # Show first diff context
            for i, (a, b) in enumerate(zip(content, corrected)):
                if a != b:
                    start = max(0, i - 15)
                    end = min(len(content), i + 15)
                    print(f"    Before: ...{content[start:end]}...")
                    print(f"    After:  ...{corrected[start:end]}...")
                    break
            
            if not dry_run:
                # Regenerate search_text from corrected content
                search_text = remove_vietnamese_accents(corrected)
                
                collection.update_one(
                    {"_id": chunk["_id"]},
                    {"$set": {
                        "content": corrected,
                        "search_text": search_text,
                    }}
                )
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Chunks scanned: {len(chunks)}")
    print(f"  Chunks with corrections: {fixed_count}")
    print(f"  Total character corrections: {total_corrections}")
    if dry_run:
        print(f"\n  (DRY RUN - no changes saved. Run without --dry-run to apply.)")
    else:
        print(f"\n  All corrections have been saved to the database.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
