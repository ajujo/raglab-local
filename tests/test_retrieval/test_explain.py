"""Tests for rank fields in fusion.py and hybrid_search.py."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from rag_lab.retrieval.fusion import weighted_rrf
from rag_lab.retrieval.hybrid_search import hybrid_search
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore


@pytest.fixture
def stores(tmp_path):
    db_path = tmp_path / "test.sqlite"
    ds = DocStore(db_path=db_path)
    ds.initialize()

    def _blob(tokens, weights):
        return (
            np.array(tokens, dtype=np.int32).tobytes(),
            np.array(weights, dtype=np.float32).tobytes(),
        )

    t1, w1 = _blob([1, 2], [0.8, 0.5])
    t2, w2 = _blob([2, 3], [0.6, 0.9])
    t3, w3 = _blob([4, 5], [0.3, 0.4])

    ds.add([
        {"chunk_id": "c1", "doc_id": "doc1", "text": "SDMX standard exchange",
         "heading_path": "Intro", "tipo": "texto", "posicion_relativa": 0.1,
         "n_tokens": 5, "line_start": 1, "line_end": 5,
         "sparse_tokens": t1, "sparse_weights": w1,
         "embedding_model_name": "bge", "embedding_model_version": "2024-09",
         "embedding_dim": 4, "sparse_format_version": 1},
        {"chunk_id": "c2", "doc_id": "doc2", "text": "Metadata structure definition",
         "heading_path": "Meta", "tipo": "texto", "posicion_relativa": 0.3,
         "n_tokens": 4, "line_start": 6, "line_end": 10,
         "sparse_tokens": t2, "sparse_weights": w2,
         "embedding_model_name": "bge", "embedding_model_version": "2024-09",
         "embedding_dim": 4, "sparse_format_version": 1},
        {"chunk_id": "c3", "doc_id": "doc3", "text": "Codelist code values",
         "heading_path": "Codes", "tipo": "texto", "posicion_relativa": 0.5,
         "n_tokens": 3, "line_start": 11, "line_end": 15,
         "sparse_tokens": t3, "sparse_weights": w3,
         "embedding_model_name": "bge", "embedding_model_version": "2024-09",
         "embedding_dim": 4, "sparse_format_version": 1},
    ])

    conn = ds._conn
    conn.execute("INSERT INTO chunks_fts(chunk_id, doc_id, text) VALUES ('c1','doc1','SDMX standard exchange')")
    conn.execute("INSERT INTO chunks_fts(chunk_id, doc_id, text) VALUES ('c2','doc2','Metadata structure definition')")
    conn.execute("INSERT INTO chunks_fts(chunk_id, doc_id, text) VALUES ('c3','doc3','Codelist code values')")
    conn.commit()

    fts = FTSStore(db_path=db_path)
    fts.initialize()

    yield ds, fts

    fts.close()
    ds.close()


class TestFusionRankFields:
    def test_dense_rank_populated(self):
        dense_ids = ["c1", "c2", "c3"]
        results = weighted_rrf(dense_ids, [], [], dense_w=1.0, bm25_w=0.0, sparse_w=0.0)
        rank_map = {r["id"]: r["dense_rank"] for r in results}
        assert rank_map["c1"] == 1
        assert rank_map["c2"] == 2
        assert rank_map["c3"] == 3

    def test_bm25_rank_populated(self):
        bm25 = [{"id": "c1", "bm25_score": 5.0}, {"id": "c2", "bm25_score": 2.0}]
        results = weighted_rrf([], bm25, [], dense_w=0.0, bm25_w=1.0, sparse_w=0.0)
        rank_map = {r["id"]: r["bm25_rank"] for r in results}
        assert rank_map["c1"] == 1
        assert rank_map["c2"] == 2

    def test_sparse_rank_populated(self):
        sparse = [{"id": "c2", "sparse_score": 0.9}, {"id": "c1", "sparse_score": 0.4}]
        results = weighted_rrf([], [], sparse, dense_w=0.0, bm25_w=0.0, sparse_w=1.0)
        rank_map = {r["id"]: r["sparse_rank"] for r in results}
        assert rank_map["c2"] == 1
        assert rank_map["c1"] == 2

    def test_rank_none_when_not_in_signal(self):
        dense_ids = ["c1"]
        bm25 = [{"id": "c2", "bm25_score": 3.0}]
        results = weighted_rrf(dense_ids, bm25, [])
        r_c1 = next(r for r in results if r["id"] == "c1")
        r_c2 = next(r for r in results if r["id"] == "c2")
        assert r_c1["bm25_rank"] is None
        assert r_c2["dense_rank"] is None

    def test_all_three_signals_ranked(self):
        dense_ids = ["c1", "c2"]
        bm25 = [{"id": "c1", "bm25_score": 4.0}, {"id": "c3", "bm25_score": 1.0}]
        sparse = [{"id": "c2", "sparse_score": 0.7}]
        results = weighted_rrf(dense_ids, bm25, sparse)
        r_c1 = next(r for r in results if r["id"] == "c1")
        assert r_c1["dense_rank"] == 1
        assert r_c1["bm25_rank"] == 1
        assert r_c1["sparse_rank"] is None

    def test_result_sorted_by_rrf_score(self):
        dense_ids = ["c1", "c2", "c3"]
        results = weighted_rrf(dense_ids, [], [])
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestHybridSearchRankFields:
    def test_rrf_rank_stamped(self, stores):
        ds, fts = stores
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"ids": ["c1", "c2"], "distances": [0.1, 0.2]}

        results = hybrid_search(
            "SDMX",
            mock_vs, ds, fts,
            query_dense=np.zeros(4, dtype="float32"),
            query_sparse={1: 0.8},
            top_k=3,
            diversity_mode="off",
        )
        for i, chunk in enumerate(results):
            assert "rrf_rank" in chunk
            assert chunk["rrf_rank"] == i + 1

    def test_was_mmr_reordered_default_false(self, stores):
        ds, fts = stores
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"ids": ["c1", "c2"], "distances": [0.1, 0.2]}

        results = hybrid_search(
            "SDMX",
            mock_vs, ds, fts,
            query_dense=np.zeros(4, dtype="float32"),
            query_sparse={1: 0.8},
            top_k=3,
            diversity_mode="off",
        )
        for chunk in results:
            assert "was_mmr_reordered" in chunk
            assert chunk["was_mmr_reordered"] is False

    def test_rank_fields_present(self, stores):
        ds, fts = stores
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"ids": ["c1", "c2", "c3"], "distances": [0.1, 0.2, 0.3]}

        results = hybrid_search(
            "SDMX",
            mock_vs, ds, fts,
            query_dense=np.zeros(4, dtype="float32"),
            query_sparse={1: 0.5},
            top_k=3,
        )
        assert len(results) > 0
        for chunk in results:
            assert "dense_rank" in chunk
            assert "bm25_rank" in chunk
            assert "sparse_rank" in chunk
            assert "rrf_rank" in chunk
            assert "was_mmr_reordered" in chunk

    def test_mmr_mode_can_reorder(self, stores):
        ds, fts = stores
        mock_vs = MagicMock()
        mock_vs.query.return_value = {
            "ids": ["c1", "c2", "c3"],
            "distances": [0.05, 0.1, 0.15],
        }

        # Three chunks from different docs — MMR shouldn't reorder much,
        # but was_mmr_reordered should be a bool regardless.
        results = hybrid_search(
            "SDMX standard",
            mock_vs, ds, fts,
            query_dense=np.zeros(4, dtype="float32"),
            query_sparse={1: 0.9, 2: 0.5},
            top_k=3,
            diversity_mode="mmr",
            mmr_lambda=0.6,
        )
        for chunk in results:
            assert isinstance(chunk["was_mmr_reordered"], bool)

    def test_diversity_mode_off_bypasses_mmr(self, stores):
        """diversity_mode='off' must not set was_mmr_reordered=True on any chunk."""
        ds, fts = stores
        mock_vs = MagicMock()
        mock_vs.query.return_value = {"ids": ["c1", "c2", "c3"], "distances": [0.1, 0.2, 0.3]}

        results = hybrid_search(
            "SDMX",
            mock_vs, ds, fts,
            query_dense=np.zeros(4, dtype="float32"),
            query_sparse={1: 0.5},
            top_k=3,
            diversity_mode="off",
        )
        for chunk in results:
            assert chunk["was_mmr_reordered"] is False
