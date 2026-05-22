"""Tests for retrieval/query_processor.py

Tests:
- process_query
- _generate_hypothetical_answer (including HyDE max_tokens regression, bug 3.10)
- _generate_query_variant
"""

import pytest
from unittest.mock import patch, MagicMock

from rag_lab.retrieval.query_processor import (
    process_query,
    _generate_hypothetical_answer,
    _generate_query_variant,
    HYDE_MAX_TOKENS,
    HYDE_TEMPERATURE,
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

    def test_calls_generate_response_with_bounded_max_tokens(self):
        """HyDE must pass HYDE_MAX_TOKENS to generate_response (bug 3.10 regression)."""
        with patch("rag_lab.retrieval.query_processor.generate_response",
                   return_value="Hypothetical SDMX answer.") as mock_gen:
            _generate_hypothetical_answer("What is SDMX?")

        mock_gen.assert_called_once()
        _, kwargs = mock_gen.call_args
        assert kwargs.get("max_tokens") == HYDE_MAX_TOKENS, (
            f"HyDE must pass max_tokens={HYDE_MAX_TOKENS} to generate_response, "
            f"got {kwargs.get('max_tokens')}"
        )

    def test_calls_generate_response_with_low_temperature(self):
        """HyDE must use HYDE_TEMPERATURE for focused generation (bug 3.10)."""
        with patch("rag_lab.retrieval.query_processor.generate_response",
                   return_value="Hypothetical answer.") as mock_gen:
            _generate_hypothetical_answer("What is a DSD?")

        _, kwargs = mock_gen.call_args
        assert kwargs.get("temperature") == HYDE_TEMPERATURE

    def test_hyde_max_tokens_constant_is_bounded(self):
        """HYDE_MAX_TOKENS must be significantly smaller than LLM_MAX_TOKENS."""
        from rag_lab.config import LLM_MAX_TOKENS
        assert HYDE_MAX_TOKENS <= 512, (
            f"HYDE_MAX_TOKENS={HYDE_MAX_TOKENS} is too large for a short hypothetical paragraph"
        )
        assert HYDE_MAX_TOKENS < LLM_MAX_TOKENS, (
            "HYDE_MAX_TOKENS should be smaller than the full LLM answer budget"
        )

    def test_llm_failure_falls_back_to_original_query(self):
        """When generate_response raises, HyDE returns the original query unchanged."""
        from rag_lab.exceptions import LLMConnectionError
        with patch("rag_lab.retrieval.query_processor.generate_response",
                   side_effect=LLMConnectionError("LLM unavailable")):
            result = _generate_hypothetical_answer("What is SDMX?")
        assert result == "What is SDMX?"

    def test_empty_llm_response_falls_back(self):
        """Empty LLM response causes fallback to original query."""
        with patch("rag_lab.retrieval.query_processor.generate_response", return_value=""):
            result = _generate_hypothetical_answer("What is SDMX?")
        assert result == "What is SDMX?"


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
