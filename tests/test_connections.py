import sys
import os

# Đảm bảo import được các module từ thư mục src
sys.path.append("src")

import pytest
from db.mongo_client import connect_to_mongo, close_mongo_connection
from providers.ollama_provider import ollama_provider

def test_mongodb_connection():
    """Kiểm tra kết nối tới MongoDB"""
    print("\n--- Testing MongoDB Connection ---")
    try:
        client = connect_to_mongo()
        assert client is not None
        client.admin.command('ping')
        print("[SUCCESS] MongoDB connection established successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to connect to MongoDB: {e}")
        pytest.fail(f"MongoDB connection failed: {e}")
    finally:
        close_mongo_connection()

def test_ollama_llm_connection():
    """Kiểm tra kết nối tới Ollama LLM Model"""
    print("\n--- Testing Ollama LLM Connection ---")
    try:
        llm = ollama_provider.get_llm()
        response = llm.invoke("Say 'OK'")
        assert response is not None
        assert response.content is not None
        print(f"[SUCCESS] Ollama LLM connected successfully!")
        print(f"Response: {response.content.strip()}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Ollama LLM: {e}")
        pytest.fail(f"Ollama LLM connection failed: {e}")

def test_ollama_embeddings_connection():
    """Kiểm tra kết nối tới Ollama Embeddings Model"""
    print("\n--- Testing Ollama Embeddings Connection ---")
    try:
        embeddings = ollama_provider.get_embeddings()
        vector = embeddings.embed_query("Test connection")
        assert vector is not None
        assert len(vector) > 0
        print(f"[SUCCESS] Ollama Embeddings connected successfully!")
        print(f"Embedding vector dimension: {len(vector)}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Ollama Embeddings: {e}")
        pytest.fail(f"Ollama Embeddings connection failed: {e}")

if __name__ == "__main__":
    print("Running connection tests directly...")
    try:
        test_mongodb_connection()
    except Exception:
        pass
        
    try:
        test_ollama_llm_connection()
    except Exception:
        pass
        
    try:
        test_ollama_embeddings_connection()
    except Exception:
        pass
