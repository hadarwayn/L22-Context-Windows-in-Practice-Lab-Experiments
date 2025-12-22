"""
Claude API model implementation.

Supports Claude models via the Anthropic API.
Use this when running from Claude Code or any environment with Anthropic access.
"""

import os
from typing import Optional

import tiktoken

from .base_model import BaseModel


class ClaudeModel(BaseModel):
    """Claude model via Anthropic API."""

    def __init__(
        self,
        model_name: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None
    ):
        """
        Initialize Claude model.

        Args:
            model_name: Claude model identifier
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
        """
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._encoding = tiktoken.get_encoding("cl100k_base")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

    def query(
        self,
        context: str,
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Query Claude model via Anthropic API.

        Uses the Messages API for Claude 3+ models.
        """
        import requests

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        # Build user message with context and question
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": user_content}
            ]
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
        except requests.RequestException as e:
            return f"Error calling Claude API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Claude response: {str(e)}"

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (approximate for Claude)."""
        return len(self._encoding.encode(text))
