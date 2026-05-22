"""Tests for retrieval/query_processor.py

Tests:
- process_query (original always present, variants config-controlled)
- _generate_hypothetical_answer (including HyDE max_tokens regression, bug 3.10)
- _generate_stopword_variant / _generate_last_terms_variant / _generate_query_variant (legacy)
"""

import pytest
from unittest.mock import patch, MagicMock

from rag_lab.retrieval.query_processor import (
    process_query,
    _generate_hypothetical_answer,
    _generate_query_variant,
    _generate_stopword_variant,
    _generate_last_terms_variant,
    _filtered_terms,
    HYDE_MAX_TOKENS,
    HYDE_TEMPERATURE,
)


# ---------------------------------------------------------------------------
# process_query — core invariants
# ---------------------------------------------------------------------------

class TestProcessQuery:
    def test_original_always_first(self):
        queries = process_query("What is SDMX?", use_hyde=False)
        assert queries[0]["text"] == "What is SDMX?"
        assert queries[0]["type"] == "original"

    def test_original_never_absent(self):
        for q in ["", "   ", "What is SDMX?", "¿Qué es un DSD?", "DSD"]:
            result = process_query(q, use_hyde=False)
            assert len(result) >= 1
            assert result[0]["type"] == "original"

    def test_no_duplicates_in_output(self):
        with (
            patch("rag_lab.retrieval.query_processor.QUERY_VARIANT_STOPWORD_ENABLED", True),
            patch("rag_lab.retrieval.query_processor.QUERY_VARIANT_LAST_TERMS_ENABLED", True),
        ):
            queries = process_query("What is SDMX?", use_hyde=False)
        texts = [q["text"] for q in queries]
        assert len(texts) == len(set(texts)), f"Duplicate texts: {texts}"

    def test_stopword_variant_disabled_by_default(self):
        """With default config, no variant_stopword type should appear."""
        queries = process_query("What is the role of SDMX in data exchange?", use_hyde=False)
        types = [q["type"] for q in queries]
        assert "variant_stopword" not in types

    def test_last_terms_variant_disabled_by_default(self):
        """With default config, no variant_last_terms type should appear."""
        queries = process_query("What is the role of SDMX in data exchange?", use_hyde=False)
        types = [q["type"] for q in queries]
        assert "variant_last_terms" not in types

    def test_stopword_variant_appears_when_enabled(self):
        with patch("rag_lab.retrieval.query_processor.QUERY_VARIANT_STOPWORD_ENABLED", True):
            queries = process_query("What is the role of SDMX?", use_hyde=False)
        types = [q["type"] for q in queries]
        assert "variant_stopword" in types

    def test_last_terms_variant_appears_when_enabled(self):
        with patch("rag_lab.retrieval.query_processor.QUERY_VARIANT_LAST_TERMS_ENABLED", True):
            queries = process_query("What is the role of SDMX in data exchange formats?", use_hyde=False)
        types = [q["type"] for q in queries]
        assert "variant_last_terms" in types

    def test_stopword_variant_not_added_when_identical_to_original(self):
        """If stop-word filtering leaves the full original text, no variant is added."""
        # "SDMX DSD" has no stop words → filtered = "sdmx dsd" ≠ "SDMX DSD" (different case)
        # but "DSD" alone → filtered = "dsd" (lowercased), which != "DSD"
        # True test: if everything in query is a stop word, variant fallback = original → not added
        with patch("rag_lab.retrieval.query_processor.QUERY_VARIANT_STOPWORD_ENABLED", True):
            # "what is the" → filtered = [] → fallback = "what is the" = original → not added
            queries = process_query("what is the", use_hyde=False)
        texts = [q["text"] for q in queries]
        assert texts.count("what is the") == 1  # only original, not duplicated

    def test_empty_query_produces_one_result(self):
        queries = process_query("", use_hyde=False)
        assert len(queries) == 1
        assert queries[0]["type"] == "original"

    def test_with_hyde(self):
        queries = process_query("What is SDMX?", use_hyde=True)
        assert len(queries) >= 2
        assert queries[0]["type"] == "original"
        assert queries[1]["type"] == "hyde"

    def test_both_variants_enabled_no_duplicates(self):
        with (
            patch("rag_lab.retrieval.query_processor.QUERY_VARIANT_STOPWORD_ENABLED", True),
            patch("rag_lab.retrieval.query_processor.QUERY_VARIANT_LAST_TERMS_ENABLED", True),
        ):
            # Short query: stopword and last_terms may produce same result
            queries = process_query("What is SDMX?", use_hyde=False)
        texts = [q["text"] for q in queries]
        assert len(texts) == len(set(texts)), "Duplicate variant texts detected"


# ---------------------------------------------------------------------------
# _filtered_terms
# ---------------------------------------------------------------------------

class TestFilteredTerms:
    def test_removes_english_stop_words(self):
        result = _filtered_terms("What is the meaning of life?")
        assert "what" not in result
        assert "is" not in result
        assert "the" not in result
        assert "of" not in result
        assert "meaning" in result
        assert "life" in result

    def test_removes_spanish_stop_words(self):
        result = _filtered_terms("¿Qué es un esquema de conceptos en SDMX?")
        assert "qué" not in result
        assert "es" not in result
        assert "un" not in result
        assert "de" not in result
        assert "en" not in result
        assert "sdmx" in result
        assert "esquema" in result
        assert "conceptos" in result

    def test_preserves_acronyms(self):
        result = _filtered_terms("What is DSD in SDMX?")
        assert "dsd" in result
        assert "sdmx" in result

    def test_strips_trailing_punctuation(self):
        # strip() removes leading/trailing punctuation from each space-separated token
        result = _filtered_terms("SDMX. DSD,")
        assert "sdmx" in result
        assert "dsd" in result

    def test_empty_returns_empty(self):
        assert _filtered_terms("") == []
        assert _filtered_terms("   ") == []

    def test_all_stop_words_returns_empty(self):
        assert _filtered_terms("what is the") == []


# ---------------------------------------------------------------------------
# _generate_stopword_variant
# ---------------------------------------------------------------------------

class TestGenerateStopwordVariant:
    def test_key_terms_extracted(self):
        result = _generate_stopword_variant("What is the role of SDMX in data exchange?")
        assert "sdmx" in result
        assert "role" in result
        assert "data" in result
        assert "exchange" in result

    def test_short_query(self):
        result = _generate_stopword_variant("What is SDMX?")
        assert "sdmx" in result

    def test_long_technical_query(self):
        q = "What are the technical specifications for the SDMX GenericData format?"
        result = _generate_stopword_variant(q)
        assert "technical" in result
        assert "specifications" in result
        assert "sdmx" in result
        assert "genericdata" in result

    def test_spanish_query(self):
        q = "¿Cómo se utilizan las restricciones en SDMX para limitar los valores permitidos?"
        result = _generate_stopword_variant(q)
        assert "sdmx" in result
        # stop words removed
        assert "cómo" not in result
        assert "para" not in result

    def test_acronyms_sdmx_dsd_msd_preserved(self):
        for acronym in ["sdmx", "dsd", "msd", "cl", "cs"]:
            q = f"What is {acronym.upper()} in the standard?"
            result = _generate_stopword_variant(q)
            assert acronym in result, f"Acronym {acronym} should be preserved"

    def test_all_stop_words_falls_back_to_original(self):
        q = "what is the"
        result = _generate_stopword_variant(q)
        assert result == q

    def test_lowercased_output(self):
        result = _generate_stopword_variant("What is SDMX?")
        assert result == result.lower()


# ---------------------------------------------------------------------------
# _generate_last_terms_variant
# ---------------------------------------------------------------------------

class TestGenerateLastTermsVariant:
    def test_tail_of_long_query(self):
        q = "What is the role of the Maintenance Agency in SDMX?"
        result = _generate_last_terms_variant(q)
        # filtered: ["role", "maintenance", "agency", "sdmx"]
        # last 5 (all 4): same as stopword variant for this query
        assert "sdmx" in result

    def test_tail_focus_on_specific_topic(self):
        q = "What are the rules for Data Structure Definition key families in SDMX?"
        result = _generate_last_terms_variant(q)
        # Should end with the most specific terms
        assert "sdmx" in result

    def test_short_query_same_as_stopword(self):
        q = "What is SDMX?"
        sw = _generate_stopword_variant(q)
        lt = _generate_last_terms_variant(q)
        # For short queries (≤5 key terms), both should be the same
        assert lt == sw

    def test_all_stop_words_falls_back(self):
        assert _generate_last_terms_variant("what is the") == "what is the"

    def test_last_terms_is_suffix_of_stopword_variant(self):
        q = "What are the technical specifications for SDMX GenericData format structure?"
        sw = _generate_stopword_variant(q).split()
        lt = _generate_last_terms_variant(q).split()
        assert lt == sw[max(0, len(sw)-5):]


# ---------------------------------------------------------------------------
# _generate_query_variant (legacy dispatcher — backward compat)
# ---------------------------------------------------------------------------

class TestGenerateQueryVariantLegacy:
    def test_idx_0_calls_stopword(self):
        q = "What is the meaning of life?"
        assert _generate_query_variant(q, 0) == _generate_stopword_variant(q)

    def test_idx_1_calls_last_terms(self):
        q = "What is the meaning of life?"
        assert _generate_query_variant(q, 1) == _generate_last_terms_variant(q)

    def test_idx_unknown_returns_original(self):
        q = "What is SDMX?"
        assert _generate_query_variant(q, 99) == q

    def test_remove_stop_words(self):
        result = _generate_query_variant("What is the meaning of life?", 0)
        assert "meaning" in result
        assert "life" in result

    def test_tail_terms(self):
        result = _generate_query_variant("What is the meaning of life?", 1)
        assert "life" in result

    def test_no_stop_words_in_query(self):
        result = _generate_query_variant("Hello world", 0)
        assert result == "hello world"

    def test_all_stop_words(self):
        result = _generate_query_variant("What is the", 0)
        assert result == "What is the"


# ---------------------------------------------------------------------------
# HyDE tests (unchanged from v1.10)
# ---------------------------------------------------------------------------

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
        assert kwargs.get("max_tokens") == HYDE_MAX_TOKENS

    def test_calls_generate_response_with_low_temperature(self):
        with patch("rag_lab.retrieval.query_processor.generate_response",
                   return_value="Hypothetical answer.") as mock_gen:
            _generate_hypothetical_answer("What is a DSD?")
        _, kwargs = mock_gen.call_args
        assert kwargs.get("temperature") == HYDE_TEMPERATURE

    def test_hyde_max_tokens_constant_is_bounded(self):
        from rag_lab.config import LLM_MAX_TOKENS
        assert HYDE_MAX_TOKENS <= 512
        assert HYDE_MAX_TOKENS < LLM_MAX_TOKENS

    def test_llm_failure_falls_back_to_original_query(self):
        from rag_lab.exceptions import LLMConnectionError
        with patch("rag_lab.retrieval.query_processor.generate_response",
                   side_effect=LLMConnectionError("LLM unavailable")):
            result = _generate_hypothetical_answer("What is SDMX?")
        assert result == "What is SDMX?"

    def test_empty_llm_response_falls_back(self):
        with patch("rag_lab.retrieval.query_processor.generate_response", return_value=""):
            result = _generate_hypothetical_answer("What is SDMX?")
        assert result == "What is SDMX?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
