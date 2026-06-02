"""
Unit tests for the Embedder module (LangChain HuggingFaceEmbeddings).

All tests use mocks to avoid downloading the actual model.
The singleton is reset before each test to ensure isolation.
"""

import json
import os
import sys
import threading

import pytest
from unittest.mock import patch, MagicMock

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.embedder import Embedder


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the Embedder singleton before and after each test."""
    Embedder._reset()
    yield
    Embedder._reset()


@pytest.fixture
def config_path(tmp_path):
    """Create a temporary embedder config file."""
    config = {
        "model_name": "BAAI/bge-m3",
        "dimension": 1024,
        "batch_size": 32,
        "max_length": 8192,
        "normalize_embeddings": True,
        "device_priority": ["cuda", "mps", "cpu"],
        "show_progress_bar": False,
    }
    path = tmp_path / "embedder_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def _make_mock_model(dimension=1024):
    """Create a mock LangChain HuggingFaceEmbeddings model."""
    mock_model = MagicMock()

    def mock_embed_documents(texts):
        return [[0.1] * dimension for _ in texts]

    def mock_embed_query(text):
        return [0.1] * dimension

    mock_model.embed_documents = MagicMock(side_effect=mock_embed_documents)
    mock_model.embed_query = MagicMock(side_effect=mock_embed_query)
    return mock_model


# ── Singleton Tests ─────────────────────────────────────────────────

class TestSingleton:
    """Tests for the singleton pattern."""

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_singleton_returns_same_instance(self, mock_device, mock_load, config_path):
        """Two instantiations should return the exact same object."""
        e1 = Embedder(config_path)
        e2 = Embedder(config_path)
        assert e1 is e2

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_model_loaded_once(self, mock_device, mock_load, config_path):
        """The model should only be loaded once across multiple instantiations."""
        Embedder(config_path)
        Embedder(config_path)
        Embedder(config_path)
        mock_load.assert_called_once()

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_singleton_thread_safety(self, mock_device, mock_load, config_path):
        """Multiple threads creating Embedder should all get the same instance."""
        instances = []

        def create_instance():
            instances.append(Embedder(config_path))

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        assert all(inst is instances[0] for inst in instances)

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_reset_allows_reinitialization(self, mock_device, mock_load, config_path):
        """After _reset(), a new instance should be created."""
        e1 = Embedder(config_path)
        Embedder._reset()
        e2 = Embedder(config_path)
        assert e1 is not e2


# ── Configuration Tests ─────────────────────────────────────────────

class TestConfiguration:
    """Tests for configuration loading and properties."""

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_config_properties(self, mock_device, mock_load, config_path):
        """Properties should reflect the loaded config."""
        embedder = Embedder(config_path)
        assert embedder.model_name == "BAAI/bge-m3"
        assert embedder.dimension == 1024
        assert embedder.batch_size == 32
        assert embedder.max_length == 8192
        assert embedder.device == "cpu"

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model(512))
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_custom_config(self, mock_device, mock_load, tmp_path):
        """Custom config values should be loaded correctly."""
        config = {
            "model_name": "custom/model-v2",
            "dimension": 512,
            "batch_size": 16,
            "max_length": 4096,
            "normalize_embeddings": False,
            "device_priority": ["cpu"],
            "show_progress_bar": True,
        }
        path = tmp_path / "custom_config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        embedder = Embedder(str(path))
        assert embedder.model_name == "custom/model-v2"
        assert embedder.dimension == 512
        assert embedder.batch_size == 16
        assert embedder.max_length == 4096

    def test_missing_config_file(self, tmp_path):
        """Should raise FileNotFoundError for a non-existent config."""
        with pytest.raises(FileNotFoundError):
            Embedder(str(tmp_path / "nonexistent.json"))


# ── Device Detection Tests ──────────────────────────────────────────

class TestDeviceDetection:
    """Tests for auto device detection logic."""

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    def test_cuda_selected_when_available(self, mock_load, config_path):
        """CUDA should be selected when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            embedder = Embedder(config_path)
            assert embedder.device == "cuda"

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    def test_cpu_fallback_when_no_gpu(self, mock_load, config_path):
        """CPU should be the fallback when no GPU is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            embedder = Embedder(config_path)
            assert embedder.device == "cpu"

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    def test_mps_selected_when_cuda_unavailable(self, mock_load, config_path):
        """MPS should be selected when CUDA is unavailable but MPS is."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            embedder = Embedder(config_path)
            assert embedder.device == "mps"


# ── LangChain Embedding Tests ──────────────────────────────────────

class TestEmbedding:
    """Tests for embed_documents and embed_query (LangChain API)."""

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_embed_documents(self, mock_device, mock_load, config_path):
        """embed_documents should return list of lists with correct shape."""
        embedder = Embedder(config_path)
        texts = ["Nhân vật Tràng", "Giá trị nhân đạo", "Bối cảnh nạn đói"]
        result = embedder.embed_documents(texts)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, list) for v in result)
        assert all(len(v) == 1024 for v in result)

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_embed_query(self, mock_device, mock_load, config_path):
        """embed_query should return a single embedding vector."""
        embedder = Embedder(config_path)
        result = embedder.embed_query("Phân tích nhân vật Mị trong Vợ chồng A Phủ")
        assert isinstance(result, list)
        assert len(result) == 1024

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_embed_documents_delegates_to_langchain(self, mock_device, mock_load, config_path):
        """embed_documents should delegate to the underlying LangChain model."""
        embedder = Embedder(config_path)
        texts = ["test"]
        embedder.embed_documents(texts)
        embedder._model.embed_documents.assert_called_once_with(texts)  # type: ignore[attr-defined]

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_embed_query_delegates_to_langchain(self, mock_device, mock_load, config_path):
        """embed_query should delegate to the underlying LangChain model."""
        embedder = Embedder(config_path)
        embedder.embed_query("test query")
        embedder._model.embed_query.assert_called_once_with("test query")  # type: ignore[attr-defined]

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_langchain_embeddings_property(self, mock_device, mock_load, config_path):
        """langchain_embeddings property should return the underlying model."""
        embedder = Embedder(config_path)
        lc_model = embedder.langchain_embeddings
        assert lc_model is embedder._model


# ── Input Validation Tests ──────────────────────────────────────────

class TestInputValidation:
    """Tests for input validation in embed_documents and embed_query."""

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_empty_list_raises(self, mock_device, mock_load, config_path):
        """Embedding an empty list should raise ValueError."""
        embedder = Embedder(config_path)
        with pytest.raises(ValueError, match="must not be empty"):
            embedder.embed_documents([])

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_non_list_input_raises(self, mock_device, mock_load, config_path):
        """Embedding a non-list should raise ValueError."""
        embedder = Embedder(config_path)
        with pytest.raises(ValueError, match="Expected list"):
            embedder.embed_documents("not a list")  # type: ignore[arg-type]

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_non_string_item_raises(self, mock_device, mock_load, config_path):
        """Embedding a list with non-string items should raise ValueError."""
        embedder = Embedder(config_path)
        with pytest.raises(ValueError, match="expected str"):
            embedder.embed_documents(["valid", 123])  # type: ignore[arg-type]

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_whitespace_only_item_raises(self, mock_device, mock_load, config_path):
        """Embedding a list with whitespace-only strings should raise ValueError."""
        embedder = Embedder(config_path)
        with pytest.raises(ValueError, match="empty or whitespace"):
            embedder.embed_documents(["valid", "   "])

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_embed_query_empty_string_raises(self, mock_device, mock_load, config_path):
        """Embedding an empty query should raise ValueError."""
        embedder = Embedder(config_path)
        with pytest.raises(ValueError, match="must not be empty"):
            embedder.embed_query("")

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_embed_query_non_string_raises(self, mock_device, mock_load, config_path):
        """Embedding a non-string query should raise ValueError."""
        embedder = Embedder(config_path)
        with pytest.raises(ValueError, match="Expected str"):
            embedder.embed_query(42)  # type: ignore[arg-type]


# ── Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_large_batch(self, mock_device, mock_load, config_path):
        """Should handle batches larger than batch_size without error."""
        embedder = Embedder(config_path)
        texts = [f"Sample text {i}" for i in range(100)]
        result = embedder.embed_documents(texts)
        assert len(result) == 100
        assert all(len(v) == 1024 for v in result)

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_single_item_list(self, mock_device, mock_load, config_path):
        """Should handle a single-item list correctly."""
        embedder = Embedder(config_path)
        result = embedder.embed_documents(["Only one text"])
        assert len(result) == 1
        assert len(result[0]) == 1024

    @patch("core.embedder.Embedder._load_model", return_value=_make_mock_model())
    @patch("core.embedder.Embedder._detect_device", return_value="cpu")
    def test_unicode_vietnamese_text(self, mock_device, mock_load, config_path):
        """Should handle Vietnamese Unicode text without errors."""
        embedder = Embedder(config_path)
        texts = [
            "Phân tích nhân vật Tràng trong truyện ngắn Vợ nhặt của Kim Lân",
            "Giá trị nhân đạo trong tác phẩm Chí Phèo của Nam Cao",
            "Bút pháp nghệ thuật của Nguyễn Tuân qua tùy bút Người lái đò sông Đà",
        ]
        result = embedder.embed_documents(texts)
        assert len(result) == 3
        assert all(isinstance(v, list) for v in result)
