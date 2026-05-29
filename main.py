import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Thêm thư mục src vào sys.path để import dễ dàng hơn
sys.path.append("src")
from db.mongo_client import connect_to_mongo, close_mongo_connection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi chạy kết nối khi server bắt đầu (không có await)
    connect_to_mongo()
    yield
    # Đóng kết nối khi server tắt (không có await)
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
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

