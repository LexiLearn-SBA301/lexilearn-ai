import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Add src folder to sys.path
sys.path.append("src")
from db.mongo_client import connect_to_mongo, close_mongo_connection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Establish MongoDB connection when starting FastAPI server
    connect_to_mongo()
    yield
    # Close connection when stopping
    close_mongo_connection()

app = FastAPI(
    title="RAG Service",
    description="API for Retrieval-Augmented Generation Service",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"message": "Welcome to RAG Service API"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "rag-service"}


if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser(description="RAG Service API and CLI tool.")
    parser.add_argument(
        "--serve", 
        action="store_true", 
        help="Khởi chạy máy chủ FastAPI Server (chạy mặc định nếu không truyền tham số)."
    )
    parser.add_argument(
        "--ingest", 
        action="store_true", 
        help="Kích hoạt dịch vụ nạp dữ liệu IngestService bất đồng bộ."
    )
    parser.add_argument(
        "--pdf", 
        type=str, 
        default="docs", 
        help="Đường dẫn đến file PDF đơn lẻ hoặc thư mục chứa các file PDF cần nạp (mặc định: 'docs')."
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Gửi câu hỏi truy vấn hệ thống RAG để nhận câu trả lời từ LLM."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Chạy chương trình đánh giá tự động dựa trên bộ Ground Truth."
    )
    parser.add_argument(
        "--lop",
        type=int,
        help="Bộ lọc lớp học (10, 11, 12) khi truy vấn."
    )
    parser.add_argument(
        "--work",
        type=str,
        help="Bộ lọc tên tác phẩm văn học khi truy vấn."
    )

    args = parser.parse_args()

    if args.ingest:
        from services.ingest_service import IngestService
        
        print("=" * 80)
        print(f"KHỞI ĐỘNG TIẾN TRÌNH INGESTION CHO: {args.pdf}")
        print("=" * 80)
        
        try:
            service = IngestService()
            job_id = service.start_ingestion(args.pdf)
            print(f"Đã khởi tạo Job nạp dữ liệu bất đồng bộ thành công.")
            print(f"Mã Job ID: {job_id}")
            print("Đang chạy ngầm và theo dõi tiến độ, vui lòng không tắt CMD...")
            print("-" * 80)

            last_status = None
            last_processed = -1
            last_errors = []

            while True:
                job = service.get_job_status(job_id)
                if not job:
                    print("Lỗi: Không tìm thấy thông tin Job trong database.")
                    break

                status = str(job.get("status") or "pending")
                total = job.get("total_files", 0)
                processed = job.get("processed_files", 0)
                errors = job.get("errors", [])

                # Print progress update on change
                if status != last_status or processed != last_processed or len(errors) > len(last_errors):
                    time_str = datetime.now().strftime("%H:%M:%S")
                    print(f"[{time_str}] Trạng thái: {status.upper()} | Tiến trình: Đã xử lý {processed}/{total} tệp PDF.")
                    
                    # Print new errors
                    if len(errors) > len(last_errors):
                        for err in errors[len(last_errors):]:
                            print(f"   [LỖI] {err}")
                    
                    last_status = status
                    last_processed = processed
                    last_errors = list(errors)

                if status in ["done", "error"]:
                    print("=" * 80)
                    if status == "done":
                        print(f"HOÀN THÀNH JOB '{job_id}'! Nạp thành công {processed}/{total} tệp.")
                    else:
                        print(f"THẤT BẠI JOB '{job_id}'! Không có tệp nào được nạp thành công.")
                    
                    if errors:
                        print(f"Tổng hợp các lỗi xảy ra ({len(errors)} lỗi):")
                        for idx, err in enumerate(errors):
                            print(f"  {idx + 1}. {err}")
                    print("=" * 80)
                    sys.exit(0)

                time.sleep(1.5)

        except Exception as e:
            print(f"Lỗi nghiêm trọng khi khởi chạy tiến trình nạp: {e}")
            sys.exit(1)

    elif args.query:
        from services.rag_service import RAGService
        
        # Build filter dict
        filters = {}
        if args.lop:
            filters["lop"] = args.lop
        if args.work:
            filters["ten_tac_pham"] = args.work
            
        print("=" * 80)
        print(f"TRUY VẤN HỆ THỐNG RAG: {args.query}")
        if filters:
            print(f"Bộ lọc: {filters}")
        print("=" * 80)
        
        try:
            rag_service = RAGService()
            result = rag_service.query(args.query, filters=filters)
            
            print("\nCÂU TRẢ LỜI TỪ HỆ THỐNG LLM:")
            print("-" * 80)
            print(result["answer"])
            print("-" * 80)
            
            print(f"\nTÀI LIỆU THAM KHẢO TRÍCH XUẤT ({len(result['sources'])} chunks):")
            for idx, src in enumerate(result["sources"]):
                metadata = src.get("metadata", {})
                title = metadata.get("ten_tac_pham", "Không rõ tác phẩm")
                author = metadata.get("tac_gia", "Không rõ tác giả")
                page = src.get("position", {}).get("page", "?")
                score = src.get("rrf_score", 0.0)
                print(f"\n[{idx + 1}] {title} - {author} (Trang {page}) | Điểm RRF: {score:.5f}")
                print(f"    Nội dung: {src.get('content', '')[:250]}...")
            print("=" * 80)
            sys.exit(0)
            
        except Exception as e:
            print(f"Lỗi khi thực hiện truy vấn RAG: {e}")
            sys.exit(1)
            
    elif args.evaluate:
        from services.rag_service import RAGService
        
        print("=" * 80)
        print("BẮT ĐẦU ĐÁNH GIÁ HỆ THỐNG RAG TRÊN BỘ DỮ LIỆU GROUND TRUTH")
        print("=" * 80)
        
        try:
            rag_service = RAGService()
            result = rag_service.evaluate()
            
            print("\n" + "=" * 30 + " KẾT QUẢ ĐÁNH GIÁ " + "=" * 30)
            print(f" * Tổng số câu hỏi đánh giá: {result['total_queries']}")
            print(f" * Số lần truy vấn trúng đích (Hits): {result['hits']}")
            print(f" * Tỉ lệ Hit Rate@{result['limit']}: {result['hit_rate'] * 100:.2f}%")
            print(f" * Điểm số Mean Reciprocal Rank (MRR): {result['mrr']:.4f}")
            print("=" * 80)
            sys.exit(0)
            
        except Exception as e:
            print(f"Lỗi khi chạy đánh giá RAG: {e}")
            sys.exit(1)

    else:
        # Default behavior: run uvicorn server
        import uvicorn
        # Disable hot reload when executing via other command arguments to avoid loop triggers
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
