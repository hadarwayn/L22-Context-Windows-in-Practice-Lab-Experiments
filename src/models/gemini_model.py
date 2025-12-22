"""
Gemini API model implementation.

Supports Gemini models via the Google AI API.
Use this when running from Gemini CLI or Google AI Studio.
"""

import os
from typing import Optional

import tiktoken

from .base_model import BaseModel
import google.auth
from google.auth.transport.requests import Request


class GeminiModel(BaseModel):
    """Gemini model via Google AI API."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None
    ):
        """
        Initialize Gemini model.

        Args:
            model_name: Gemini model identifier
            api_key: Google AI API key (or from GOOGLE_API_KEY env var)
        """
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self.credentials = None

        if not self.api_key:
            # Try Application Default Credentials (ADC)
            try:
                self.credentials, _ = google.auth.default()
                print("Using Google Application Default Credentials")
            except Exception as e:
                # Raise original error if ADC also fails
                raise ValueError(
                    "Google API key required (GOOGLE_API_KEY) or Application Default Credentials."
                ) from e

    def query(
        self,
        context: str,
        question: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Query Gemini model via Google AI API.

        Uses the generateContent endpoint.
        """
        import requests

        # Build prompt with context and question
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"Instructions: {system_prompt}\n\n")
        prompt_parts.append(f"Context:\n{context}\n\n")
        prompt_parts.append(f"Question: {question}\n\nAnswer:")

        full_prompt = "".join(prompt_parts)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

        payload = {
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "temperature": 0.1
            }
        }

        try:
            headers = {}
            params = {}
            
            if self.api_key:
                params["key"] = self.api_key
            elif self.credentials:
                if not self.credentials.valid:
                    self.credentials.refresh(Request())
                headers["Authorization"] = f"Bearer {self.credentials.token}"
                # For ADC/Vertex AI, the endpoint might differ, but assuming raw Gemini API via
                # generativelanguage.googleapis.com supports OAuth tokens (which it does).
            
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except requests.RequestException as e:
            return f"Error calling Gemini API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing Gemini response: {str(e)}"

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (approximate for Gemini)."""
        return len(self._encoding.encode(text))
