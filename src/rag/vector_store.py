"""
Vector store module using ChromaDB.

Provides functions for storing and retrieving documents by similarity.
"""

from typing import List, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings


def create_vector_store(
    collection_name: str = "documents",
    persist_directory: Optional[Path] = None
) -> chromadb.Collection:
    """
    Create or get a ChromaDB collection.

    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist data (None for in-memory)

    Returns:
        ChromaDB collection
    """
    if persist_directory:
        client = chromadb.PersistentClient(path=str(persist_directory))
    else:
        client = chromadb.Client()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_documents(
    collection: chromadb.Collection,
    documents: List[str],
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[dict]] = None
) -> None:
    """
    Add documents to the vector store.

    Args:
        collection: ChromaDB collection
        documents: List of document texts
        ids: Optional list of document IDs
        metadatas: Optional list of metadata dicts
    """
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )


def similarity_search(
    collection: chromadb.Collection,
    query: str,
    k: int = 3
) -> List[Tuple[str, float]]:
    """
    Search for similar documents.

    Args:
        collection: ChromaDB collection
        query: Query text
        k: Number of results to return

    Returns:
        List of (document, score) tuples
    """
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "distances"]
    )

    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Convert distances to similarity scores (ChromaDB returns distances)
    return [
        (doc, 1 - dist)
        for doc, dist in zip(documents, distances)
    ]


def split_documents(
    documents: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split documents into smaller chunks.

    Args:
        documents: List of documents to split
        chunk_size: Target size of each chunk (in characters)
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """
    chunks = []

    for doc in documents:
        if len(doc) <= chunk_size:
            chunks.append(doc)
            continue

        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunk = doc[start:end]

            # Try to break at sentence boundary
            if end < len(doc):
                last_period = chunk.rfind(".")
                if last_period > chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - chunk_overlap

    return chunks


def clear_collection(collection: chromadb.Collection) -> None:
    """
    Clear all documents from a collection.

    Args:
        collection: ChromaDB collection to clear
    """
    # Get all IDs and delete them
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
