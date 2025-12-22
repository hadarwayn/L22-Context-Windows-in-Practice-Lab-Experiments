"""
LLM Interface implementation supporting Ollama and API providers.

Provides concrete implementation of BaseModel for various LLM backends.
"""

import time
from typing import Optional

import requests
import tiktoken

from .base_model import BaseModel


class OllamaModel(BaseModel):
    """Ollama-based local LLM model."""

    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama model.

        Args:
            model_name: Name of the Ollama model
            base_url: Ollama server URL
        """
        super().__init__(model_name)
        self.base_url = base_url.rstrip("/")
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def query(
        self,
        context: str,
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Query Ollama model."""
        prompt = self._build_prompt(context, question, system_prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            return f"Error: {str(e)}"

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self._encoding.encode(text))

    def _build_prompt(
        self,
        context: str,
        question: str,
        system_prompt: Optional[str]
    ) -> str:
        """Build the full prompt."""
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")
        parts.append(f"Context:\n{context}\n")
        parts.append(f"Question: {question}\n")
        parts.append("Answer:")
        return "\n".join(parts)


class MockModel(BaseModel):
    """Mock model for testing without API calls."""

    def __init__(self, model_name: str = "mock"):
        """Initialize mock model."""
        super().__init__(model_name)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def query(
        self,
        context: str,
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Return mock response based on context content."""
        # Simulate processing time
        time.sleep(0.1)

        # Simple mock: look for quoted values in context
        if "CEO" in question.upper():
            for line in context.split("\n"):
                if "CEO" in line or "מנכ" in line:
                    return line
        return "Information not found in context."

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self._encoding.encode(text))


def get_model(
    backend: str = "ollama",
    model_name: Optional[str] = None,
    **kwargs
) -> BaseModel:
    """
    Factory function to get appropriate model instance.

    Args:
        backend: Backend type ('ollama', 'mock', 'claude', 'gemini')
        model_name: Model name to use
        **kwargs: Additional arguments for model initialization

    Returns:
        BaseModel instance
    """
    if backend == "mock":
        return MockModel(model_name or "mock")
    elif backend == "ollama":
        return OllamaModel(model_name or "llama2", **kwargs)
    elif backend == "claude":
        from .claude_model import ClaudeModel
        return ClaudeModel(model_name or "claude-3-haiku-20240307", **kwargs)
    elif backend == "gemini":
        from .gemini_model import GeminiModel
        return GeminiModel(model_name or "gemini-2.0-flash-exp", **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
