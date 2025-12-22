"""RAG (Retrieval-Augmented Generation) components."""

from .embeddings import (
    get_embedding_model,
    embed_text,
    embed_texts,
    compute_similarity,
)
from .vector_store import (
    create_vector_store,
    add_documents,
    similarity_search,
    split_documents,
    clear_collection,
)

__all__ = [
    "get_embedding_model",
    "embed_text",
    "embed_texts",
    "compute_similarity",
    "create_vector_store",
    "add_documents",
    "similarity_search",
    "split_documents",
    "clear_collection",
]
