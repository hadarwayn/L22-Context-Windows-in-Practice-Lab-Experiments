"""Tests for RAG components."""

import pytest
import numpy as np

from src.rag import (
    embed_text,
    embed_texts,
    compute_similarity,
    create_vector_store,
    add_documents,
    similarity_search,
    split_documents,
    clear_collection,
)


class TestEmbeddings:
    """Tests for embedding functions."""

    def test_embed_single_text(self):
        """Test embedding a single text."""
        embedding = embed_text("Hello world")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        texts = ["Hello", "World", "Test"]
        embeddings = embed_texts(texts)
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 3

    def test_similar_texts_have_high_similarity(self):
        """Test that similar texts have high cosine similarity."""
        emb1 = embed_text("The cat sat on the mat")
        emb2 = embed_text("The cat is sitting on the mat")
        similarity = compute_similarity(emb1, emb2)
        assert similarity > 0.8  # Should be quite similar


class TestVectorStore:
    """Tests for vector store functions."""

    def test_create_collection(self):
        """Test creating a vector store collection."""
        collection = create_vector_store("test_collection")
        assert collection is not None

    def test_add_and_search(self):
        """Test adding documents and searching."""
        collection = create_vector_store("test_search")
        clear_collection(collection)

        docs = [
            "Python is a programming language",
            "Java is also a programming language",
            "Cats are cute animals"
        ]
        add_documents(collection, docs)

        results = similarity_search(collection, "programming languages", k=2)
        assert len(results) == 2

        # Top results should be about programming
        top_doc, _ = results[0]
        assert "programming" in top_doc.lower()


class TestDocumentSplitting:
    """Tests for document splitting."""

    def test_split_short_document(self):
        """Test that short documents aren't split."""
        doc = "This is a short document."
        chunks = split_documents([doc], chunk_size=1000)
        assert len(chunks) == 1

    def test_split_long_document(self):
        """Test that long documents are split."""
        doc = "Word " * 500  # ~500 words
        chunks = split_documents([doc], chunk_size=100)
        assert len(chunks) > 1

    def test_multiple_documents(self):
        """Test splitting multiple documents."""
        docs = ["Short doc.", "Another short doc."]
        chunks = split_documents(docs, chunk_size=1000)
        assert len(chunks) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
