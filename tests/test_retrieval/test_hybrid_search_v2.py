"""Integration tests for the new two-stage hybrid_search."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from rag_lab.retrieval.hybrid_search import hybrid_search
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore


@pytest.fixture
def stores(tmp_path):
    """Populate DocStore + FTSStore with sample chunks."""
    db_path = tmp_path / "test.sqlite"

    ds = DocStore(db_path=db_path)
    ds.initialize()
    import numpy as np

    def _make_blob(tokens, weights):
        return (
            np.array(tokens, dtype=np.int32).tobytes(),
            np.array(weights, dtype=np.float32).tobytes(),
        )

    t1, w1 = _make_blob([1, 2], [0.8, 0.5])
    t2, w2 = _make_blob([2, 3], [0.6, 0.9])
    t3, w3 = _make_blob([4, 5], [0.3, 0.4])

    ds.add([
        {"chunk_id": "c1", "doc_id": "doc1", "text": "SDMX data exchange standard",
         "heading_path": "Intro", "tipo": "texto", "posicion_relativa": 0.1,
         "n_tokens": 5, "line_start": 1, "line_end": 5,
         "sparse_tokens": t1, "sparse_weights": w1,
         "embedding_model_name": "bge", "embedding_model_version": "1",
         "embedding_dim": 4, "sparse_format_version": 1},
        {"chunk_id": "c2", "doc_id": "doc1", "text": "Metadata structure definition",
         "heading_path": "Meta", "tipo": "texto", "posicion_relativa": 0.3,
         "n_tokens": 4, "line_start": 6, "line_end": 10,
         "sparse_tokens": t2, "sparse_weights": w2,
         "embedding_model_name": "bge", "embedding_model_version": "1",
         "embedding_dim": 4, "sparse_format_version": 1},
        {"chunk_id": "c3", "doc_id": "doc2", "text": "Glossary of key terms",
         "heading_path": "Glossary", "tipo": "texto", "posicion_relativa": 0.5,
         "n_tokens": 4, "line_start": 1, "line_end": 3,
         "sparse_tokens": t3, "sparse_weights": w3,
         "embedding_model_name": "bge", "embedding_model_version": "1",
         "embedding_dim": 4, "sparse_format_version": 1},
    ])

    fts = FTSStore(db_path=db_path)
    fts.initialize()

    return ds, fts


class TestHybridSearchV2:
    def test_returns_chunks_with_five_scores(self, stores):
        ds, fts = stores

        mock_vs = MagicMock()
        mock_vs.query.return_value = {"ids": ["c1", "c2"], "distances": [0.1, 0.2]}

        query_dense = np.zeros(4, dtype="float32")
        query_sparse = {1: 0.8, 2: 0.5}

        results = hybrid_search(
            "SDMX standard",
            mock_vs, ds, fts,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=3,
        )

        assert len(results) > 0
        for chunk in results:
            assert "rrf_score" in chunk
            assert "dense_score" in chunk
            assert "bm25_score" in chunk
            assert "sparse_score" in chunk
            assert "in_dense_topk" in chunk
            assert "in_bm25_topk" in chunk
            assert "in_sparse_topk" in chunk

    def test_results_sorted_by_rrf_score(self, stores):
        ds, fts = stores

        mock_vs = MagicMock()
        mock_vs.query.return_value = {"ids": ["c1", "c2", "c3"], "distances": [0.1, 0.2, 0.3]}

        results = hybrid_search(
            "SDMX",
            mock_vs, ds, fts,
            query_dense=np.zeros(4, dtype="float32"),
            query_sparse={1: 1.0},
            top_k=5,
        )

        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_doc_id_filter(self, stores):
        ds, fts = stores

        mock_vs = MagicMock()
        mock_vs.query.return_value = {"ids": ["c1"], "distances": [0.1]}

        results = hybrid_search(
            "SDMX",
            mock_vs, ds, fts,
            query_dense=np.zeros(4, dtype="float32"),
            query_sparse={},
            top_k=5,
            doc_ids=["doc1"],
        )

        for chunk in results:
            assert chunk["doc_id"] == "doc1"

    def test_no_dense_falls_back_to_bm25(self, stores):
        ds, fts = stores

        mock_vs = MagicMock()

        results = hybrid_search(
            "SDMX",
            mock_vs, ds, fts,
            query_dense=None,
            query_sparse=None,
            top_k=5,
        )

        # mock_vs.query should NOT have been called since query_dense is None
        mock_vs.query.assert_not_called()
        # BM25 may still return results
        # (might be empty if FTS5 table has no relevant matches — just don't crash)
        assert isinstance(results, list)

    def test_chunk_text_included(self, stores):
        ds, fts = stores

        mock_vs = MagicMock()
        mock_vs.query.return_value = {"ids": ["c1"], "distances": [0.1]}

        results = hybrid_search(
            "SDMX",
            mock_vs, ds, fts,
            query_dense=np.zeros(4, dtype="float32"),
            query_sparse={1: 0.5},
            top_k=3,
        )

        for chunk in results:
            assert "text" in chunk
            assert len(chunk["text"]) > 0
