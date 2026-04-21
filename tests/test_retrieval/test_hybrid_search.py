"""Tests for retrieval/hybrid_search.py

Tests:
- hybrid_search
- _reciprocal_rank_fusion
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from rag_lab.retrieval.hybrid_search import hybrid_search, _reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_empty_lists(self):
        result = _reciprocal_rank_fusion([], [], k=60)
        assert result == []

    def test_single_item(self):
        result = _reciprocal_rank_fusion(["id1"], [{"id": "id1", "score": 0.5}], k=60)
        assert len(result) == 1
        assert result[0]["id"] == "id1"

    def test_multiple_items(self):
        dense_ids = ["id1", "id2", "id3"]
        sparse_results = [
            {"id": "id2", "score": 0.9},
            {"id": "id3", "score": 0.8},
        ]
        result = _reciprocal_rank_fusion(dense_ids, sparse_results, k=60)
        assert len(result) == 3
        # id2 and id3 should have higher scores due to RRF
        assert result[0]["id"] == "id2"

    def test_overlap_items(self):
        dense_ids = ["id1", "id2"]
        sparse_results = [
            {"id": "id2", "score": 0.9},
            {"id": "id3", "score": 0.8},
        ]
        result = _reciprocal_rank_fusion(dense_ids, sparse_results, k=60)
        assert len(result) == 3
        # id2 should have highest score (appears in both)
        assert result[0]["id"] == "id2"

    def test_no_overlap(self):
        dense_ids = ["id1", "id2"]
        sparse_results = [
            {"id": "id3", "score": 0.9},
            {"id": "id4", "score": 0.8},
        ]
        result = _reciprocal_rank_fusion(dense_ids, sparse_results, k=60)
        assert len(result) == 4


class TestHybridSearch:
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store.query.return_value = {
            "ids": ["chunk1", "chunk2", "chunk3"],
            "distances": [0.1, 0.2, 0.3],
            "documents": ["doc1", "doc2", "doc3"],
            "metadatas": [{"path": "H1"}, {"path": "H2"}, {"path": "H3"}],
        }
        return store

    @pytest.fixture
    def mock_sparse_store(self):
        store = Mock()
        store.query.return_value = [
            {"id": "chunk2", "score": 0.9},
            {"id": "chunk1", "score": 0.8},
        ]
        return store

    @pytest.fixture
    def mock_doc_store(self):
        store = Mock()
        store.get_by_ids.return_value = [
            {"chunk_id": "chunk1", "text": "Text 1"},
            {"chunk_id": "chunk2", "text": "Text 2"},
            {"chunk_id": "chunk3", "text": "Text 3"},
        ]
        return store

    def test_hybrid_search(self, mock_vector_store, mock_sparse_store, mock_doc_store):
        result = hybrid_search(
            "test question",
            mock_vector_store,
            mock_sparse_store,
            mock_doc_store,
            query_dense=np.random.rand(1024),
            query_sparse={"a": 1.0},
            top_k=2,
        )
        assert len(result) == 3

    def test_empty_dense_results(self, mock_doc_store):
        mock_vector_store = Mock()
        mock_vector_store.query.return_value = {"ids": [], "distances": [], "documents": [], "metadatas": []}
        
        mock_sparse_store = Mock()
        mock_sparse_store.query.return_value = []
        
        result = hybrid_search(
            "test question",
            mock_vector_store,
            mock_sparse_store,
            mock_doc_store,
            query_dense=None,
            query_sparse=None,
            top_k=2,
        )
        assert isinstance(result, list)

    def test_with_query_dense_and_sparse(self, mock_vector_store, mock_sparse_store, mock_doc_store):
        query_dense = np.random.rand(1024)
        query_sparse = {"term": 1.0}
        
        result = hybrid_search(
            "test",
            mock_vector_store,
            mock_sparse_store,
            mock_doc_store,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=5,
        )
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
