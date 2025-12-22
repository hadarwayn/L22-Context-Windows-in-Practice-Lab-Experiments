"""
Common utility functions for Context Windows Lab.

Provides helper functions for text processing, file I/O, and timing.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken


def count_words(text: str) -> int:
    """
    Count the number of words in a text.

    Args:
        text: Input text string

    Returns:
        Number of words
    """
    return len(text.split())


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text using tiktoken.

    Args:
        text: Input text string
        model: Model name for tokenizer selection

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save dictionary data to a JSON file.

    Args:
        data: Dictionary to save
        path: Output file path
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load dictionary data from a JSON file.

    Args:
        path: Input file path

    Returns:
        Loaded dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """
    Get current timestamp as formatted string.

    Returns:
        Timestamp string in format YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_date_string() -> str:
    """
    Get current date as formatted string.

    Returns:
        Date string in format YYYY-MM-DD
    """
    return datetime.now().strftime("%Y-%m-%d")


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed * 1000


def calculate_average(values: List[float]) -> float:
    """
    Calculate the average of a list of values.

    Args:
        values: List of numeric values

    Returns:
        Average value, or 0.0 if list is empty
    """
    if not values:
        return 0.0
    return sum(values) / len(values)
