"""Tests for the full_hyde benchmark variant (v1.12).

Covers:
- full_hyde registered in ALL_VARIANT_NAMES but NOT VARIANT_NAMES
- full_hyde falls back gracefully when LLM is unavailable
- run_variant dispatcher correctly routes to run_full_hyde
- generate_response force_no_thinking suppresses token multiplier
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from rag_lab.benchmark.pipeline_variants import (
    VARIANT_NAMES,
    ALL_VARIANT_NAMES,
    HYDE_VARIANT_NAMES,
    run_variant,
)
from rag_lab.generation.llm_client import generate_response


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

class TestHydeVariantRegistry:
    def test_full_hyde_in_all_variant_names(self):
        assert "full_hyde" in ALL_VARIANT_NAMES

    def test_full_hyde_not_in_default_variant_names(self):
        """full_hyde must not run by default — it requires a live LLM."""
        assert "full_hyde" not in VARIANT_NAMES

    def test_full_hyde_in_hyde_variant_names(self):
        assert "full_hyde" in HYDE_VARIANT_NAMES

    def test_standard_variants_unchanged(self):
        """v1.12 must not remove or rename any existing variant."""
        for v in ["dense", "bm25", "dense_bm25", "hybrid", "full"]:
            assert v in VARIANT_NAMES
            assert v in ALL_VARIANT_NAMES


# ---------------------------------------------------------------------------
# run_full_hyde behaviour
# ---------------------------------------------------------------------------

class TestRunFullHyde:
    def _make_stores(self):
        """Create lightweight mocked stores for unit tests."""
        vector_store = MagicMock()
        doc_store = MagicMock()
        fts_store = MagicMock()

        # hybrid_search will be patched, so store mocks are mostly structural
        return vector_store, doc_store, fts_store

    def _dummy_dense(self):
        return np.zeros(1024, dtype=np.float32)

    def _dummy_sparse(self):
        return {42: 0.5, 100: 0.3}

    def test_fallback_when_llm_unavailable(self):
        """When LLM fails, full_hyde must fall back to original query_dense."""
        from rag_lab.exceptions import LLMConnectionError

        vs, ds, fts = self._make_stores()
        dummy_dense = self._dummy_dense()

        with patch("rag_lab.retrieval.query_processor.generate_response",
                   side_effect=LLMConnectionError("server down")):
            with patch("rag_lab.benchmark.pipeline_variants.hybrid_search",
                       return_value=([], {"candidate_pool_size": 0, "n_dense": 0,
                                         "n_bm25": 0, "n_sparse": 0,
                                         "sparse_used": False})) as mock_hs:
                run_variant(
                    "full_hyde", "What is SDMX?",
                    dummy_dense, self._dummy_sparse(),
                    vs, ds, fts,
                    top_k=5, rrf_k=20, rerank_device="cpu",
                    embedding_device="cpu",
                )

        # hybrid_search must have been called (with fallback dense = original)
        mock_hs.assert_called_once()
        call_kwargs = mock_hs.call_args[1]
        # When LLM fails, hyde_used=False → same dense as input
        np.testing.assert_array_equal(call_kwargs["query_dense"], dummy_dense)

    def test_hyde_used_flag_in_stats(self):
        """Stats dict must include hyde_used indicating whether LLM was used."""
        from rag_lab.exceptions import LLMConnectionError

        vs, ds, fts = self._make_stores()

        with patch("rag_lab.retrieval.query_processor.generate_response",
                   side_effect=LLMConnectionError("server down")):
            with patch("rag_lab.benchmark.pipeline_variants.hybrid_search",
                       return_value=([], {"candidate_pool_size": 0, "n_dense": 0,
                                         "n_bm25": 0, "n_sparse": 0,
                                         "sparse_used": False})):
                _, stats = run_variant(
                    "full_hyde", "What is SDMX?",
                    self._dummy_dense(), self._dummy_sparse(),
                    vs, ds, fts,
                    top_k=5, rrf_k=20, rerank_device="cpu",
                    embedding_device="cpu",
                )

        assert "hyde_used" in stats
        assert stats["hyde_used"] is False

    def test_original_query_used_for_bm25(self):
        """BM25 always receives the original query text, never the hypothetical."""
        vs, ds, fts = self._make_stores()

        with patch("rag_lab.retrieval.query_processor.generate_response",
                   return_value="SDMX is a standard for structured data exchange."):
            with patch("rag_lab.benchmark.pipeline_variants.hybrid_search",
                       return_value=([], {"candidate_pool_size": 0, "n_dense": 0,
                                         "n_bm25": 0, "n_sparse": 0,
                                         "sparse_used": False})) as mock_hs:
                with patch("rag_lab.embedding.encoder.encode_chunks",
                           return_value=(np.zeros((1, 1024), dtype=np.float32), {})):
                    run_variant(
                        "full_hyde", "What is SDMX?",
                        self._dummy_dense(), self._dummy_sparse(),
                        vs, ds, fts,
                        top_k=5, rrf_k=20, rerank_device="cpu",
                        embedding_device="cpu",
                    )

        # First positional arg to hybrid_search is the query text (for BM25)
        call_args = mock_hs.call_args[0]
        assert call_args[0] == "What is SDMX?"  # original, not hypothetical

    def test_unknown_variant_raises(self):
        vs, ds, fts = self._make_stores()
        with pytest.raises(ValueError, match="Unknown variant"):
            run_variant(
                "nonexistent_variant", "query",
                self._dummy_dense(), {}, vs, ds, fts,
                top_k=5, rrf_k=20, rerank_device="cpu",
            )


# ---------------------------------------------------------------------------
# generate_response: force_no_thinking skips token multiplier
# ---------------------------------------------------------------------------

class TestGenerateResponseForceNoThinking:
    def test_force_no_thinking_skips_multiplier(self):
        """With force_no_thinking=True, actual_max_tokens == requested max_tokens."""
        with patch("rag_lab.generation.llm_client._get_client") as mock_client_factory:
            mock_choice = MagicMock()
            mock_choice.message.content = "response"
            mock_choice.message.reasoning_content = ""
            mock_choice.finish_reason = "stop"
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage = None
            mock_client_factory.return_value.chat.completions.create.return_value = mock_resp

            generate_response("sys", "user", max_tokens=300, force_no_thinking=True)

        _, call_kwargs = mock_client_factory.return_value.chat.completions.create.call_args
        assert call_kwargs["max_tokens"] == 300  # not 300*4=1200

    def test_default_uses_multiplier(self):
        """Without force_no_thinking, actual_max_tokens = requested * multiplier."""
        from rag_lab.generation.llm_client import _THINKING_TOKEN_MULTIPLIER

        with patch("rag_lab.generation.llm_client._get_client") as mock_client_factory:
            mock_choice = MagicMock()
            mock_choice.message.content = "response"
            mock_choice.message.reasoning_content = ""
            mock_choice.finish_reason = "stop"
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage = None
            mock_client_factory.return_value.chat.completions.create.return_value = mock_resp

            generate_response("sys", "user", max_tokens=300, force_no_thinking=False)

        _, call_kwargs = mock_client_factory.return_value.chat.completions.create.call_args
        assert call_kwargs["max_tokens"] == 300 * _THINKING_TOKEN_MULTIPLIER

    def test_timeout_forwarded(self):
        """timeout parameter must be forwarded to the API create call."""
        with patch("rag_lab.generation.llm_client._get_client") as mock_client_factory:
            mock_choice = MagicMock()
            mock_choice.message.content = "response"
            mock_choice.message.reasoning_content = ""
            mock_choice.finish_reason = "stop"
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage = None
            mock_client_factory.return_value.chat.completions.create.return_value = mock_resp

            generate_response("sys", "user", timeout=15.0)

        _, call_kwargs = mock_client_factory.return_value.chat.completions.create.call_args
        assert call_kwargs["timeout"] == 15.0

    def test_none_timeout_forwarded(self):
        """timeout=None must be passed through (no timeout)."""
        with patch("rag_lab.generation.llm_client._get_client") as mock_client_factory:
            mock_choice = MagicMock()
            mock_choice.message.content = "response"
            mock_choice.message.reasoning_content = ""
            mock_choice.finish_reason = "stop"
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            mock_resp.usage = None
            mock_client_factory.return_value.chat.completions.create.return_value = mock_resp

            generate_response("sys", "user", timeout=None)

        _, call_kwargs = mock_client_factory.return_value.chat.completions.create.call_args
        assert call_kwargs["timeout"] is None
