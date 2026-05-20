"""Tests for retrieval/query_processor.py

Tests:
- process_query
- _generate_hypothetical_answer
- _generate_query_variant
"""

import pytest
from rag_lab.retrieval.query_processor import (
    process_query,
    _generate_hypothetical_answer,
    _generate_query_variant,
)


class TestProcessQuery:
    def test_original_query(self):
        queries = process_query("What is SDMX?", use_hyde=False)
        assert len(queries) >= 1
        assert queries[0]["text"] == "What is SDMX?"

    def test_with_hyde(self):
        queries = process_query("What is SDMX?", use_hyde=True)
        assert len(queries) >= 2  # original + hyde
        assert queries[0]["type"] == "original"
        assert queries[1]["type"] == "hyde"

    def test_with_expansion(self):
        queries = process_query("What is SDMX?", use_hyde=False)
        # Should have original + at least 2 variants
        assert len(queries) >= 1

    def test_empty_query(self):
        queries = process_query("", use_hyde=False)
        assert len(queries) >= 1

    def test_query_with_stop_words(self):
        queries = process_query("What is the meaning of life?", use_hyde=False)
        # Should have original + variants
        assert len(queries) >= 1


class TestGenerateHypotheticalAnswer:
    def test_basic(self):
        result = _generate_hypothetical_answer("What is SDMX?")
        assert len(result) > 0
        assert "sdmx" in result.lower()

    @pytest.mark.llm_required
    def test_empty(self):
        result = _generate_hypothetical_answer("")
        assert len(result) > 0


class TestGenerateQueryVariant:
    def test_remove_stop_words(self):
        result = _generate_query_variant("What is the meaning of life?", 0)
        assert "What" in result or "meaning" in result

    def test_tail_terms(self):
        result = _generate_query_variant("What is the meaning of life?", 1)
        assert "life" in result

    def test_no_stop_words(self):
        result = _generate_query_variant("Hello world", 0)
        assert result == "hello world"

    def test_all_stop_words(self):
        result = _generate_query_variant("What is the", 0)
        assert result == "What is the"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
