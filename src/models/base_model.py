"""
Abstract base class for LLM models.

Provides a unified interface for different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseModel(ABC):
    """Abstract base class for all LLM models."""

    def __init__(self, model_name: str):
        """
        Initialize the model.

        Args:
            model_name: Name/identifier of the model
        """
        self.model_name = model_name

    @abstractmethod
    def query(
        self,
        context: str,
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Query the model with context and question.

        Args:
            context: The context text to provide to the model
            question: The question to ask about the context
            system_prompt: Optional system prompt

        Returns:
            Model's response as a string
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        pass

    def get_model_name(self) -> str:
        """
        Get the model name.

        Returns:
            Model name string
        """
        return self.model_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
