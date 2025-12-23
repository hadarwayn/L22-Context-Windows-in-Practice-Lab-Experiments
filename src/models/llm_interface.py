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
    """Mock model simulating realistic LLM behavior patterns."""

    def __init__(self, model_name: str = "mock"):
        """Initialize mock model."""
        super().__init__(model_name)
        self._encoding = tiktoken.get_encoding("cl100k_base")
        import random
        self._random = random

    def query(
        self,
        context: str,
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Return mock response simulating realistic LLM behavior.

        Simulates: Lost in the Middle, context size impact, RAG benefit.
        """
        tokens = self.count_tokens(context)

        # Simulate processing time based on context size
        time.sleep(0.05 + tokens * 0.00002)

        # Find answer patterns - map question keywords to expected answers
        patterns = {
            "CEO": "David Cohen", "revenue": "$50 million",
            "target": "$50 million", "deadline": "March 15, 2025",
            "budget": "$2.5 million", "side effect": "dizziness and nausea",
            "תופעות": "סחרחורת ובחילה",
            "finding": "records", "record": "processed",
        }

        answer = None
        for key, ans in patterns.items():
            if key.lower() in question.lower():
                answer = ans
                break

        if not answer:
            # Try to find any quoted or key values in context
            if "record" in context.lower():
                answer = "records"
            else:
                return "Information not found in context."

        # Check if answer or related content exists in context
        answer_found = answer.lower() in context.lower()
        if not answer_found:
            return "The requested information is not in the context."

        # Calculate position in context (0.0 = start, 1.0 = end)
        answer_pos = context.lower().find(answer.lower())
        if answer_pos == -1:
            answer_pos = len(context) // 2  # Default to middle if not found
        relative_pos = answer_pos / max(len(context), 1)

        # Position-based accuracy (Lost in the Middle effect)
        if relative_pos < 0.15:  # Start
            position_prob = 0.95
        elif relative_pos > 0.85:  # End
            position_prob = 0.92
        else:  # Middle - U-shaped curve
            middle_distance = abs(relative_pos - 0.5)
            position_prob = 0.45 + (middle_distance * 0.6)

        # Context size penalty: accuracy drops with more tokens
        if tokens < 1000:
            size_factor = 1.0
        elif tokens < 5000:
            size_factor = 0.85
        elif tokens < 15000:
            size_factor = 0.65
        else:
            size_factor = 0.45

        success_prob = position_prob * size_factor

        if self._random.random() < success_prob:
            return f"Based on the context, the answer is: {answer}"
        else:
            return "Unable to locate the specific information requested."

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
