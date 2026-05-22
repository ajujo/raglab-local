"""Regression tests for reranker device cache handling (bug 3.9).

Before the fix, load_reranker(device) returned the cached model regardless of
the requested device — a call with device="cpu" after a device="cuda" load
would silently return the cuda model. These tests verify correct behaviour.
"""

from unittest.mock import MagicMock, patch, call

import pytest

from rag_lab.retrieval.reranker import (
    load_reranker,
    reset_reranker_cache,
    _reranker_cache,
    _reranker_cache_device,
)


class TestRerankerDeviceCache:
    def setup_method(self):
        reset_reranker_cache()

    def teardown_method(self):
        reset_reranker_cache()

    def test_reset_clears_both_cache_and_device(self):
        """reset_reranker_cache clears model AND device tracker."""
        import rag_lab.retrieval.reranker as _mod
        _mod._reranker_cache = MagicMock()
        _mod._reranker_cache_device = "cuda"

        reset_reranker_cache()

        assert _mod._reranker_cache is None
        assert _mod._reranker_cache_device is None

    def test_same_device_returns_cached_model(self):
        """Second call with the same device returns the cached model without reloading."""
        mock_model = MagicMock()

        # FlagReranker is imported locally inside load_reranker, so we patch at source.
        with patch("FlagEmbedding.FlagReranker", return_value=mock_model) as mock_cls:
            m1 = load_reranker("cpu")
            m2 = load_reranker("cpu")

        assert m1 is m2
        mock_cls.assert_called_once()  # only one instantiation

    def test_different_device_reloads_model(self):
        """Switching device forces a new model load instead of returning stale cache."""
        mock_cpu = MagicMock(name="cpu_model")
        mock_cuda = MagicMock(name="cuda_model")

        with patch("FlagEmbedding.FlagReranker", side_effect=[mock_cpu, mock_cuda]) as mock_cls:
            m_cpu = load_reranker("cpu")
            m_cuda = load_reranker("cuda")

        assert m_cpu is mock_cpu
        assert m_cuda is mock_cuda
        assert mock_cls.call_count == 2

    def test_device_tracked_after_load(self):
        """_reranker_cache_device reflects the device of the loaded model."""
        import rag_lab.retrieval.reranker as _mod
        mock_model = MagicMock()

        with patch("FlagEmbedding.FlagReranker", return_value=mock_model):
            load_reranker("cpu")

        assert _mod._reranker_cache_device == "cpu"
        assert _mod._reranker_cache is mock_model

    def test_none_device_uses_config_default(self):
        """Calling load_reranker() without device uses RERANKER_DEVICE from config."""
        import rag_lab.retrieval.reranker as _mod
        mock_model = MagicMock()

        with (
            patch("rag_lab.retrieval.reranker.RERANKER_DEVICE", "cpu"),
            patch("FlagEmbedding.FlagReranker", return_value=mock_model),
        ):
            load_reranker(None)

        assert _mod._reranker_cache_device == "cpu"

    def test_rerank_passes_device_to_loader(self):
        """rerank() passes its device argument through to load_reranker."""
        from rag_lab.retrieval.reranker import rerank

        mock_model = MagicMock()
        mock_model.compute_score.return_value = [0.9, 0.7]

        with patch("rag_lab.retrieval.reranker.load_reranker", return_value=mock_model) as mock_load:
            rerank("query", [{"text": "a"}, {"text": "b"}], top_k=2, device="cpu")

        mock_load.assert_called_once_with("cpu")
