import os
import re
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("rag-service.security.injection-guard")
logging.basicConfig(level=logging.INFO)


class InjectionGuard:
    """
    InjectionGuard is responsible for inspecting user queries for potential prompt injection attacks
    including jailbreaks, prompt leaking, high density of special characters, and code/script blocks.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Load configuration and compile jailbreak regex patterns.
        """
        # Default configuration values
        self.max_query_length = 1000
        self.max_special_char_ratio = 0.15
        self.block_code_blocks = True
        self.jailbreak_patterns = [
            r"ignore\s+previous\s+instructions",
            r"ignore\s+system\s+instructions",
            r"forget\s+all\s+instructions",
            r"you\s+are\s+now",
            r"act\s+as\s+a",
            r"system\s+prompt",
            r"bỏ\s+qua\s+chỉ\s+thị",
            r"bỏ\s+qua\s+hướng\s+dẫn",
            r"quên\s+đi\s+các\s+chỉ\s+thị",
            r"đóng\s+vai",
            r"trở\s+thành\s+một",
            r"lệnh\s+hệ\s+thống",
            r"bạn\s+là\s+một"
        ]

        # Resolve config path
        if not config_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(
                os.path.join(current_dir, "..", "config", "injection_config.json")
            )

        # Attempt to load configuration
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.max_query_length = config.get("max_query_length", self.max_query_length)
                self.max_special_char_ratio = config.get("max_special_char_ratio", self.max_special_char_ratio)
                self.block_code_blocks = config.get("block_code_blocks", self.block_code_blocks)
                self.jailbreak_patterns = config.get("jailbreak_patterns", self.jailbreak_patterns)
                logger.info(f"Loaded InjectionGuard configuration from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load InjectionGuard config ({e}). Using default values.")

        # Compile combined regex pattern for speed
        try:
            combined_pattern = "|".join(self.jailbreak_patterns)
            self.jailbreak_rx = re.compile(combined_pattern, re.IGNORECASE)
        except Exception as e:
            logger.error(f"Failed to compile jailbreak regex patterns ({e}). Compiling fallback.")
            self.jailbreak_rx = re.compile("ignore\\s+previous", re.IGNORECASE)

    def check_query(self, query: str) -> Dict[str, Any]:
        """
        Sanitize and check query against security rules.
        Returns:
            dict containing:
                - safe: bool
                - reason: Optional[str]
        """
        if not query or not query.strip():
            return {"safe": True, "reason": None}

        # Rule 1: Check maximum length
        if len(query) > self.max_query_length:
            return {
                "safe": False,
                "reason": f"Độ dài câu hỏi vượt quá giới hạn tối đa ({len(query)}/{self.max_query_length} ký tự)."
            }

        # Rule 2: Check for code blocks or script tags
        if self.block_code_blocks:
            if "```" in query or re.search(r"<script\b[^>]*>", query, re.IGNORECASE):
                return {
                    "safe": False,
                    "reason": "Câu hỏi không được chứa khối mã nguồn (code blocks) hoặc thẻ kịch bản (script tags)."
                }

        # Rule 3: Check special character ratio
        special_chars = set("!@#$%^&*()_+={}[]|\\:;\"'<>,.?/~`")
        special_count = sum(1 for char in query if char in special_chars)
        ratio = special_count / len(query) if len(query) > 0 else 0.0

        if ratio > self.max_special_char_ratio:
            return {
                "safe": False,
                "reason": f"Mật độ ký tự đặc biệt trong câu hỏi quá cao ({ratio * 100:.1f}% > {self.max_special_char_ratio * 100:.1f}%)."
            }

        # Rule 4: Match compiled regex patterns for Jailbreak / Prompt Leaking
        if self.jailbreak_rx.search(query):
            return {
                "safe": False,
                "reason": "Phát hiện mẫu câu hỏi mang tính chất can thiệp hệ thống (Prompt Injection/Jailbreak)."
            }

        return {"safe": True, "reason": None}
