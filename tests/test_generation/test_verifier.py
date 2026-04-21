"""Tests for generation/verifier.py

Tests:
- verify_citations
"""

import pytest
from rag_lab.generation.verifier import verify_citations


class TestVerifyCitations:
    def test_valid_citation(self):
        response = "See [DOC: Test, Section 1] for details."
        chunks = [
            {
                "chunk_id": "1",
                "text": "Test content",
                "heading_path": "Section 1",
            }
        ]
        result = verify_citations(response, chunks)
        assert "See" in result

    def test_missing_citation(self):
        response = "Some answer without citation."
        chunks = [
            {
                "chunk_id": "1",
                "text": "Test content",
                "heading_path": "Section 1",
            }
        ]
        result = verify_citations(response, chunks)
        # Should handle missing citation
        assert len(result) > 0

    def test_empty_response(self):
        chunks = [
            {
                "chunk_id": "1",
                "text": "Test content",
                "heading_path": "Section 1",
            }
        ]
        result = verify_citations("", chunks)
        assert len(result) == 0

    def test_multiple_citations(self):
        response = "See [DOC: A, Section 1] and [DOC: B, Section 2]."
        chunks = [
            {
                "chunk_id": "1",
                "text": "Content A",
                "heading_path": "Section 1",
            },
            {
                "chunk_id": "2",
                "text": "Content B",
                "heading_path": "Section 2",
            }
        ]
        result = verify_citations(response, chunks)
        assert "See" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
