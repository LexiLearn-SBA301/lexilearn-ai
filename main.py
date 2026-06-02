import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI

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
                    break

                time.sleep(1.5)

        except Exception as e:
            print(f"Lỗi nghiêm trọng khi khởi chạy tiến trình nạp: {e}")
            sys.exit(1)

    else:
        # Default behavior: run uvicorn server
        import uvicorn
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
