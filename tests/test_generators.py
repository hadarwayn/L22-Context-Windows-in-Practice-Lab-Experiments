"""Tests for document generators."""

import pytest

from src.generators import (
    generate_filler_text,
    embed_fact_at_position,
    generate_document,
    generate_document_set,
    generate_hebrew_documents,
)


class TestFillerText:
    """Tests for filler text generation."""

    def test_generates_text(self):
        """Test that filler text is generated."""
        text = generate_filler_text(100)
        assert len(text) > 0

    def test_approximate_word_count(self):
        """Test that word count is approximately correct."""
        text = generate_filler_text(100)
        word_count = len(text.split())
        # Should be at least the requested amount
        assert word_count >= 100

    def test_different_topics(self):
        """Test different topics generate different text."""
        business = generate_filler_text(50, "business")
        tech = generate_filler_text(50, "technology")
        # Both should generate text
        assert len(business) > 0
        assert len(tech) > 0


class TestFactEmbedding:
    """Tests for fact embedding."""

    def test_embed_at_start(self):
        """Test fact embedded at start."""
        text = "This is some text."
        fact = "Important fact."
        result, idx = embed_fact_at_position(text, fact, "start")
        assert result.startswith(fact)
        assert idx == 0

    def test_embed_at_end(self):
        """Test fact embedded at end."""
        text = "This is some text."
        fact = "Important fact."
        result, idx = embed_fact_at_position(text, fact, "end")
        assert result.endswith(fact)

    def test_embed_at_middle(self):
        """Test fact embedded in middle."""
        text = "Word one two three four five six."
        fact = "FACT"
        result, _ = embed_fact_at_position(text, fact, "middle")
        assert fact in result
        # Should not be at very start or end
        assert not result.startswith(fact)
        assert not result.endswith(fact)


class TestDocumentGeneration:
    """Tests for document generation."""

    def test_generate_document(self):
        """Test document generation with fact."""
        doc = generate_document(200, "CEO is John", "middle")
        assert "CEO is John" in doc
        assert len(doc.split()) >= 50  # Should have substantial content

    def test_generate_document_set(self):
        """Test document set generation."""
        docs = generate_document_set(5, 100)
        assert len(docs) == 5
        for doc in docs:
            assert len(doc) > 0


class TestHebrewDocuments:
    """Tests for Hebrew document generation."""

    def test_generates_hebrew_docs(self):
        """Test Hebrew document generation."""
        docs = generate_hebrew_documents(5, ["technology", "law"])
        assert len(docs) == 5

    def test_returns_tuples(self):
        """Test that tuples of (doc, topic) are returned."""
        docs = generate_hebrew_documents(3, ["medicine"])
        for doc, topic in docs:
            assert isinstance(doc, str)
            assert isinstance(topic, str)
            assert len(doc) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
