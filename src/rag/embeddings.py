"""
Text embedding module using sentence transformers.

Provides functions for converting text to vector representations.
"""

from typing import List, Optional

import numpy as np

# Lazy loading to avoid import time
_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get or create the sentence transformer model.

    Args:
        model_name: Name of the sentence transformer model

    Returns:
        SentenceTransformer model instance
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
    return _model


def embed_text(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Convert a single text to embedding vector.

    Args:
        text: Text to embed
        model_name: Model to use for embedding

    Returns:
        Embedding vector as numpy array
    """
    model = get_embedding_model(model_name)
    return model.encode(text, convert_to_numpy=True)


def embed_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = False
) -> np.ndarray:
    """
    Convert multiple texts to embedding vectors.

    Args:
        texts: List of texts to embed
        model_name: Model to use for embedding
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar

    Returns:
        Array of embedding vectors
    """
    model = get_embedding_model(model_name)
    return model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=show_progress
    )


def compute_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
