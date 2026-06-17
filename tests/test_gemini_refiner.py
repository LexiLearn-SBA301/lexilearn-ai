import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.gemini_refiner import GeminiRefiner
from core.pdf_reader import ExtractedElement


class TestGeminiRefiner:
    
    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_init_with_api_key(self, mock_getenv, mock_genai):
        """Test initialization when API key is provided."""
        mock_getenv.side_effect = lambda key, default="": "dummy_key" if key == "GEMINI_API_KEY" else "gemini-2.0-flash"
        
        refiner = GeminiRefiner()
        
        assert refiner.gemini_api_key == "dummy_key"
        assert refiner.gemini_model == "gemini-2.0-flash"
        assert refiner.is_available() is True
        mock_genai.Client.assert_called_once_with(api_key="dummy_key")

    @patch("core.gemini_refiner.os.getenv")
    def test_init_without_api_key(self, mock_getenv):
        """Test initialization when API key is NOT provided."""
        mock_getenv.return_value = None
        
        refiner = GeminiRefiner()
        
        assert refiner.gemini_api_key is None
        assert refiner.is_available() is False
        assert refiner.client is None

    @patch("core.gemini_refiner.os.getenv")
    def test_init_with_empty_api_key(self, mock_getenv):
        """Test initialization when API key is an empty string or whitespace."""
        mock_getenv.side_effect = lambda key, default="": "   " if key == "GEMINI_API_KEY" else "gemini-2.0-flash"
        
        refiner = GeminiRefiner()
        
        assert refiner.gemini_api_key is None or refiner.gemini_api_key == ""
        assert refiner.is_available() is False
        assert refiner.client is None

    @patch("core.gemini_refiner.genai", None)
    @patch("core.gemini_refiner.os.getenv")
    def test_init_without_genai_library(self, mock_getenv):
        """Test initialization when google-genai is not installed."""
        mock_getenv.side_effect = lambda key, default="": "some_key" if key == "GEMINI_API_KEY" else "model"
        
        refiner = GeminiRefiner()
        
        assert refiner.is_available() is False
        assert refiner.client is None

    @patch("core.gemini_refiner.os.getenv")
    def test_refine_bypass_when_unavailable(self, mock_getenv):
        """Test that refine returns original elements if API is unavailable."""
        mock_getenv.return_value = None
        refiner = GeminiRefiner()
        
        elements = [ExtractedElement(page=1, type="paragraph", raw_text="test", source_file="doc.pdf")]
        result = refiner.refine(elements)
        
        assert result == elements
        assert result[0].raw_text == "test"

    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_refine_bypass_empty_list(self, mock_getenv, mock_genai):
        """Test that refine handles empty element list gracefully."""
        mock_getenv.side_effect = lambda key, default="": "key" if key == "GEMINI_API_KEY" else "model"
        refiner = GeminiRefiner()
        
        result = refiner.refine([])
        assert result == []

    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_group_by_page(self, mock_getenv, mock_genai):
        """Test the internal grouping logic."""
        mock_getenv.side_effect = lambda key, default="": "dummy_key" if key == "GEMINI_API_KEY" else "model"
        refiner = GeminiRefiner()
        
        elements = [
            ExtractedElement(page=1, type="p", raw_text="A", source_file="d"),
            ExtractedElement(page=1, type="p", raw_text="B", source_file="d"),
            ExtractedElement(page=2, type="p", raw_text="C", source_file="d")
        ]
        
        pages = refiner._group_by_page(elements)
        assert len(pages) == 2
        assert len(pages[1]) == 2
        assert len(pages[2]) == 1

    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_parse_response_with_markers(self, mock_getenv, mock_genai):
        """Test parsing logic with standard markers."""
        mock_getenv.side_effect = lambda key, default="": "dummy_key" if key == "GEMINI_API_KEY" else "model"
        refiner = GeminiRefiner()
        
        fake_response = (
            "===PAGE_1===\n"
            "This is corrected text for page 1.\n"
            "Line two.\n\n"
            "===PAGE_3===\n"
            "Corrected text for page 3.\n"
        )
        
        expected_pages = [1, 3]
        parsed = refiner._parse_response(fake_response, expected_pages)
        
        assert 1 in parsed
        assert 3 in parsed
        assert "This is corrected text for page 1.\nLine two." in parsed[1]
        assert "Corrected text for page 3." in parsed[3]

    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_parse_response_wrapped_in_code_block(self, mock_getenv, mock_genai):
        """Test parsing when Gemini wraps the entire response in a markdown code block."""
        mock_getenv.side_effect = lambda key, default="": "key" if key == "GEMINI_API_KEY" else "model"
        refiner = GeminiRefiner()
        
        fake_response = (
            "```text\n"
            "===PAGE_5===\n"
            "Corrected page 5 text.\n"
            "===PAGE_6===\n"
            "Corrected page 6 text.\n"
            "```\n"
        )
        
        parsed = refiner._parse_response(fake_response, [5, 6])
        
        assert 5 in parsed
        assert 6 in parsed
        assert parsed[5] == "Corrected page 5 text."
        assert parsed[6] == "Corrected page 6 text."

    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_parse_response_no_markers(self, mock_getenv, mock_genai):
        """Test parsing when Gemini returns text without any markers at all."""
        mock_getenv.side_effect = lambda key, default="": "key" if key == "GEMINI_API_KEY" else "model"
        refiner = GeminiRefiner()
        
        fake_response = "Here is some corrected text without any page markers."
        
        parsed = refiner._parse_response(fake_response, [1, 2])
        
        # Should return empty dict, preserving original text
        assert parsed == {}

    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_apply_refined_text_exact_match(self, mock_getenv, mock_genai):
        """Test applying text when line counts match elements."""
        mock_getenv.side_effect = lambda key, default="": "dummy_key" if key == "GEMINI_API_KEY" else "model"
        refiner = GeminiRefiner()
        
        elements = [
            ExtractedElement(page=1, type="heading", raw_text="Err0r He4ding", source_file="d"),
            ExtractedElement(page=1, type="paragraph", raw_text="S0me t3xt.", source_file="d")
        ]
        
        refined_text = "Error Heading\nSome text."
        
        refiner._apply_refined_text_to_elements(elements, refined_text)
        
        assert elements[0].raw_text == "Error Heading"
        assert elements[1].raw_text == "Some text."

    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_apply_refined_text_mismatch_fewer_lines(self, mock_getenv, mock_genai):
        """Test applying when Gemini returns fewer lines than original elements."""
        mock_getenv.side_effect = lambda key, default="": "key" if key == "GEMINI_API_KEY" else "model"
        refiner = GeminiRefiner()
        
        elements = [
            ExtractedElement(page=1, type="heading", raw_text="Heading", source_file="d"),
            ExtractedElement(page=1, type="paragraph", raw_text="Para 1", source_file="d"),
            ExtractedElement(page=1, type="paragraph", raw_text="Para 2", source_file="d")
        ]
        
        # Gemini only returned 2 lines instead of 3
        refined_text = "Heading corrected\nPara 1 corrected"
        refiner._apply_refined_text_to_elements(elements, refined_text)
        
        assert elements[0].raw_text == "Heading corrected"
        assert elements[1].raw_text == "Para 1 corrected"
        # Third element should keep its original text
        assert elements[2].raw_text == "Para 2"

    @patch("core.gemini_refiner.time.sleep")
    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_call_gemini_retry_logic(self, mock_getenv, mock_genai, mock_sleep):
        """Test that API call retries on failure."""
        mock_getenv.side_effect = lambda key, default="": "dummy_key" if key == "GEMINI_API_KEY" else "model"
        refiner = GeminiRefiner()
        
        # Mock client to fail twice then succeed
        mock_response = MagicMock()
        mock_response.text = "Success"
        
        refiner.client.models.generate_content.side_effect = [
            Exception("Network Error"),
            Exception("Timeout"),
            mock_response
        ]
        
        result = refiner._call_gemini("Test prompt")
        
        assert result == "Success"
        assert refiner.client.models.generate_content.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("core.gemini_refiner.genai")
    @patch("core.gemini_refiner.os.getenv")
    def test_call_gemini_empty_response(self, mock_getenv, mock_genai):
        """Test that empty Gemini response is handled gracefully."""
        mock_getenv.side_effect = lambda key, default="": "key" if key == "GEMINI_API_KEY" else "model"
        refiner = GeminiRefiner()
        
        mock_response = MagicMock()
        mock_response.text = None
        refiner.client.models.generate_content.return_value = mock_response
        
        result = refiner._call_gemini("Test prompt")
        assert result is None

    def test_strip_markdown_wrapper(self):
        """Test that markdown wrappers are correctly stripped."""
        text_with_wrapper = "```text\nsome content here\nmore lines\n```"
        result = GeminiRefiner._strip_markdown_wrapper(text_with_wrapper)
        assert result == "some content here\nmore lines"
        
        # Test no wrapper
        text_no_wrapper = "===PAGE_1===\nsome content"
        result2 = GeminiRefiner._strip_markdown_wrapper(text_no_wrapper)
        assert result2 == "===PAGE_1===\nsome content"
        
        # Test just ``` with no lang
        text_plain_wrapper = "```\nplain block\n```"
        result3 = GeminiRefiner._strip_markdown_wrapper(text_plain_wrapper)
        assert result3 == "plain block"

    @patch("core.gemini_refiner.os.getenv")
    @patch("httpx.post")
    def test_ollama_backend_success(self, mock_post, mock_getenv):
        """Test that Ollama backend is used and succeeds."""
        mock_getenv.side_effect = lambda key, default="": {
            "REFINER_BACKEND": "ollama",
            "OLLAMA_URL": "http://localhost:11434",
            "OLLAMA_LLM_MODEL": "qwen2.5:3b"
        }.get(key, default)
        
        refiner = GeminiRefiner()
        assert refiner.refiner_backend == "ollama"
        assert refiner.is_available() is True
        
        # Mock httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": "Refined text from Ollama"
            }
        }
        mock_post.return_value = mock_response
        
        result = refiner._call_gemini("Test prompt")
        assert result == "Refined text from Ollama"
        mock_post.assert_called_once()
        
    @patch("core.gemini_refiner.os.getenv")
    @patch("httpx.post")
    def test_ollama_backend_failure(self, mock_post, mock_getenv):
        """Test that Ollama backend handles failure and retries."""
        mock_getenv.side_effect = lambda key, default="": {
            "REFINER_BACKEND": "ollama"
        }.get(key, default)
        
        refiner = GeminiRefiner()
        
        # Mock httpx to raise exception
        mock_post.side_effect = Exception("Ollama connection error")
        
        result = refiner._call_gemini("Test prompt")
        assert result is None
