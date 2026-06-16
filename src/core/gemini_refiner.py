import os
import json
import time
import logging
from typing import List, Optional
from core.pdf_reader import ExtractedElement

logger = logging.getLogger("rag-service.gemini-refiner")

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class GeminiRefiner:
    """
    Refines extracted text using Google Gemini API before chunking.
    Batches multiple pages to reduce API calls and costs.
    """

    def __init__(self, config_path: Optional[str] = None):
        raw_key = os.getenv("GEMINI_API_KEY", "")
        self.api_key = raw_key.strip() if raw_key else None
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        
        self.client = None
        if not genai:
            logger.warning("google-genai library is not installed. GeminiRefiner will be disabled. Run: pip install google-genai")
        elif not self.api_key:
            logger.info("GEMINI_API_KEY is not set in .env. GeminiRefiner will be disabled.")
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"GeminiRefiner initialized with model '{self.model_name}'")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Client: {e}. GeminiRefiner will be disabled.")

        # Load configuration
        if not config_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(current_dir, "..", "config", "gemini_refiner_config.json"))

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load Gemini config: {e}. Using defaults.")
            self.config = {
                "prompt_template": "VĂN BẢN CẦN SỬA:\n{batch_text}",
                "batch_size_pages": 5,
                "max_retries": 2,
                "retry_delay_seconds": 2,
                "temperature": 0.1
            }

    def is_available(self) -> bool:
        """Check if Gemini API is configured and available."""
        return self.client is not None

    def refine(self, elements: List[ExtractedElement]) -> List[ExtractedElement]:
        """
        Batches elements by page, calls Gemini API to correct OCR errors,
        and updates the raw_text of each element.
        """
        if not self.is_available() or not elements:
            return elements

        logger.info(f"Starting Gemini text refinement for {len(elements)} elements...")
        
        # Group elements by page to maintain structure and batch efficiently
        pages_data = self._group_by_page(elements)
        if not pages_data:
            return elements

        page_numbers = sorted(pages_data.keys())
        batch_size = self.config.get("batch_size_pages", 5)
        
        for i in range(0, len(page_numbers), batch_size):
            batch_pages = page_numbers[i:i + batch_size]
            logger.info(f"Refining batch: pages {batch_pages[0]} to {batch_pages[-1]}")
            
            # Prepare batch text
            batch_texts = []
            for p in batch_pages:
                page_content = "\n".join([el.raw_text for el in pages_data[p] if el.raw_text])
                batch_texts.append(f"===PAGE_{p}===\n{page_content}")
            
            full_batch_text = "\n\n".join(batch_texts)
            
            # Build prompt and call API
            prompt = self.config["prompt_template"].replace("{batch_text}", full_batch_text)
            refined_text = self._call_gemini(prompt)
            
            if refined_text:
                # Parse response and map back to pages
                refined_pages = self._parse_response(refined_text, batch_pages)
                
                # Apply refined text back to elements
                for p in batch_pages:
                    if p in refined_pages and refined_pages[p]:
                        self._apply_refined_text_to_elements(pages_data[p], refined_pages[p])
                    else:
                        logger.warning(f"Failed to parse refined text for page {p}. Keeping original.")

        return elements

    def _group_by_page(self, elements: List[ExtractedElement]) -> dict:
        """Groups elements by their page number."""
        pages = {}
        for el in elements:
            if el.page not in pages:
                pages[el.page] = []
            pages[el.page].append(el)
        return pages

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Calls Gemini API with retry logic."""
        if self.client is None:
            logger.warning("Gemini Client is not initialized. Skipping API call.")
            return None

        max_retries = self.config.get("max_retries", 2)
        delay = self.config.get("retry_delay_seconds", 2)
        temp = self.config.get("temperature", 0.1)

        config_kwargs = {"temperature": temp}
        gen_config = types.GenerateContentConfig(**config_kwargs) if types else None

        for attempt in range(max_retries + 1):
            try:
                if gen_config:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=gen_config
                    )
                else:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt
                    )
                if response.text:
                    return response.text
                logger.warning(f"Gemini returned empty response for batch. Skipping.")
                return None
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Gemini API call failed after {max_retries + 1} attempts: {e}")
                    return None
        return None

    @staticmethod
    def _strip_markdown_wrapper(text: str) -> str:
        """Remove markdown code block wrappers that Gemini sometimes adds around the entire response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split('\n')
            # Remove opening ``` line (e.g. ```text, ```markdown, ```)
            if len(lines) > 1:
                lines = lines[1:]
            # Remove closing ``` line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = '\n'.join(lines)
        return text.strip()

    def _parse_response(self, response: str, expected_pages: List[int]) -> dict:
        """
        Parses the Gemini response to extract text for each page marker.
        Returns a dictionary mapping page_number to its refined text.
        """
        refined_pages = {}
        
        # Standardize newlines
        response = response.replace('\r\n', '\n')
        
        # Strip markdown code block wrapper from entire response BEFORE parsing markers
        response = self._strip_markdown_wrapper(response)

        # Check if any markers exist at all
        first_marker = f"===PAGE_{expected_pages[0]}==="
        if first_marker not in response:
            logger.warning(f"Gemini response does not contain expected markers. Entire response will be discarded.")
            return refined_pages
        
        for i, page_num in enumerate(expected_pages):
            marker = f"===PAGE_{page_num}==="
            next_marker = f"===PAGE_{expected_pages[i+1]}===" if i + 1 < len(expected_pages) else None
            
            start_idx = response.find(marker)
            if start_idx == -1:
                logger.warning(f"Marker for page {page_num} not found in Gemini response. Keeping original.")
                continue
                
            start_content = start_idx + len(marker)
            
            if next_marker:
                end_idx = response.find(next_marker, start_content)
                if end_idx == -1:
                    end_idx = len(response)
            else:
                end_idx = len(response)
                
            page_text = response[start_content:end_idx].strip()
            refined_pages[page_num] = page_text
            
        return refined_pages

    def _apply_refined_text_to_elements(self, page_elements: List[ExtractedElement], refined_page_text: str):
        """
        Re-distributes the refined text back to the individual ExtractedElements.
        Since Gemini might alter line counts slightly, we do a best-effort 
        line-by-line or paragraph-by-paragraph matching.
        """
        refined_lines = [line.strip() for line in refined_page_text.split('\n') if line.strip()]
        
        # If the number of non-empty lines matches exactly, it's a simple 1:1 mapping
        valid_elements = [el for el in page_elements if el.raw_text and el.raw_text.strip()]
        
        if len(refined_lines) == len(valid_elements):
            for i, el in enumerate(valid_elements):
                el.raw_text = refined_lines[i]
            return

        # Best effort mapping if counts differ:
        # Just update elements sequentially until we run out of refined lines
        # In practice, with strict prompting, Gemini should preserve structure.
        logger.warning(f"Structure mismatch on page {page_elements[0].page}. Original: {len(valid_elements)} elements, Refined: {len(refined_lines)} lines.")
        
        min_len = min(len(valid_elements), len(refined_lines))
        for i in range(min_len):
            valid_elements[i].raw_text = refined_lines[i]
            
        # If Gemini truncated, leave remaining elements as they were
        # If Gemini added lines, append them to the last element
        if len(refined_lines) > len(valid_elements) and valid_elements:
            extra_text = "\n".join(refined_lines[len(valid_elements):])
            valid_elements[-1].raw_text += "\n" + extra_text
