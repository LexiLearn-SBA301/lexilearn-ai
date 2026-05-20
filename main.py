from fastapi import FastAPI

app = FastAPI(
    title="RAG Service",
    description="API for Retrieval-Augmented Generation Service",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to RAG Service API"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "rag-service"}
