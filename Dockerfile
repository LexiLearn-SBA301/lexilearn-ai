# Dockerfile — RAG Service (FastAPI).
# Embeddings + LLM đều chạy qua Ollama (container riêng) nên image này KHÔNG cần
# torch/sentence-transformers -> nhẹ.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential: build python-crfsuite (dep của underthesea). curl: cho healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

# main.py chỉ bind 127.0.0.1 trong khối __main__. Chạy uvicorn trực tiếp + bind
# 0.0.0.0 để truy cập được từ ngoài container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
