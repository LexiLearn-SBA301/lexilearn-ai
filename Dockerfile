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
# torch/torchvision bản CPU (~200MB) thay vì bản CUDA (~2.5GB): container app
# KHÔNG được cấp GPU trong docker-compose (chỉ ollama có) nên CUDA torch là tải thừa.
# Cài trước để vietocr/torchvision dùng lại, không kéo nvidia-* về.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

# main.py chỉ bind 127.0.0.1 trong khối __main__. Chạy uvicorn trực tiếp + bind
# 0.0.0.0 để truy cập được từ ngoài container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
