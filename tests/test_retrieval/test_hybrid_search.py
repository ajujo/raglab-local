"""Tests for retrieval/hybrid_search.py (updated for v2 two-stage architecture).

The old two-way RRF tests (_reciprocal_rank_fusion) have moved to test_fusion.py.
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from rag_lab.retrieval.hybrid_search import hybrid_search


class TestHybridSearch:
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store.query.return_value = {
            "ids": ["chunk1", "chunk2", "chunk3"],
            "distances": [0.1, 0.2, 0.3],
        }
        return store

    @pytest.fixture
    def mock_fts_store(self):
        store = Mock()
        store.query.return_value = [
            {"id": "chunk2", "bm25_score": 9.5},
            {"id": "chunk1", "bm25_score": 7.2},
        ]
        return store

    @pytest.fixture
    def mock_doc_store(self):
        store = Mock()
        store.get_by_ids.return_value = [
            {"chunk_id": "chunk1", "doc_id": "d1", "text": "Text 1"},
            {"chunk_id": "chunk2", "doc_id": "d1", "text": "Text 2"},
            {"chunk_id": "chunk3", "doc_id": "d1", "text": "Text 3"},
        ]
        store._conn = None  # no sparse BLOBs in this mock
        return store

    def test_hybrid_search(self, mock_vector_store, mock_fts_store, mock_doc_store):
        result = hybrid_search(
            "test question",
            mock_vector_store,
            mock_doc_store,
            mock_fts_store,
            query_dense=np.random.rand(1024),
            query_sparse={1: 1.0},
            top_k=5,
        )
        assert len(result) > 0
        # Five-score fields must be present
        for chunk in result:
            assert "rrf_score" in chunk

    def test_empty_dense_results(self, mock_fts_store, mock_doc_store):
        mock_vs = Mock()
        mock_vs.query.return_value = {"ids": [], "distances": []}

        result = hybrid_search(
            "test question",
            mock_vs,
            mock_doc_store,
            mock_fts_store,
            query_dense=None,
            query_sparse=None,
            top_k=2,
        )
        assert isinstance(result, list)

    def test_with_query_dense_and_sparse(self, mock_vector_store, mock_fts_store, mock_doc_store):
        result = hybrid_search(
            "test",
            mock_vector_store,
            mock_doc_store,
            mock_fts_store,
            query_dense=np.random.rand(1024),
            query_sparse={1: 1.0},
            top_k=5,
        )
        assert len(result) > 0

    def test_empty_doc_ids_returns_all(self, mock_vector_store, mock_fts_store, mock_doc_store):
        result = hybrid_search(
            "test",
            mock_vector_store,
            mock_doc_store,
            mock_fts_store,
            query_dense=np.random.rand(1024),
            query_sparse=None,
            top_k=5,
            doc_ids=None,
        )
        # fts_store should be called without doc_ids filter
        assert mock_fts_store.query.called
        _, kwargs = mock_fts_store.query.call_args
        assert kwargs.get("doc_ids") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
