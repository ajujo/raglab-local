"""Tests for retrieval/sparse_scorer.py."""

import sqlite3
import pytest
import numpy as np

from rag_lab.retrieval.sparse_scorer import sparse_score, load_sparse_for_chunks, rank_candidates_by_sparse


def _make_blob(tokens: list, weights: list):
    t = np.array(tokens, dtype=np.int32).tobytes()
    w = np.array(weights, dtype=np.float32).tobytes()
    return t, w


class TestSparseScore:
    def test_exact_overlap(self):
        # query has token 1 with weight 1.0; doc has token 1 with weight 2.0 → 2.0
        t, w = _make_blob([1], [2.0])
        assert sparse_score({1: 1.0}, t, w) == pytest.approx(2.0)

    def test_no_overlap(self):
        t, w = _make_blob([2], [1.0])
        assert sparse_score({1: 1.0}, t, w) == pytest.approx(0.0)

    def test_partial_overlap(self):
        t, w = _make_blob([1, 2, 3], [2.0, 3.0, 4.0])
        # query only has tokens 1 and 3
        score = sparse_score({1: 1.0, 3: 0.5}, t, w)
        assert score == pytest.approx(1.0 * 2.0 + 0.5 * 4.0)

    def test_none_blobs_returns_zero(self):
        assert sparse_score({1: 1.0}, None, None) == 0.0

    def test_empty_query_returns_zero(self):
        t, w = _make_blob([1], [1.0])
        assert sparse_score({}, t, w) == 0.0


class TestLoadSparseForChunks:
    def test_loads_blobs_from_sqlite(self, tmp_path):
        db_path = tmp_path / "test.sqlite"
        from rag_lab.storage.docstore import DocStore
        ds = DocStore(db_path=db_path)
        ds.initialize()

        t_blob, w_blob = _make_blob([10, 20], [0.5, 0.8])
        ds.add([{
            "chunk_id": "x1", "doc_id": "d1", "text": "hello",
            "heading_path": "", "tipo": "texto", "posicion_relativa": 0.0,
            "n_tokens": 1, "line_start": 0, "line_end": 1,
            "sparse_tokens": t_blob, "sparse_weights": w_blob,
            "embedding_model_name": "bge", "embedding_model_version": "1",
            "embedding_dim": 1024, "sparse_format_version": 1,
        }])

        result = load_sparse_for_chunks(ds._conn, ["x1", "nonexistent"])
        assert "x1" in result
        assert result["x1"][0] == t_blob
        assert result["x1"][1] == w_blob
        assert "nonexistent" not in result
        ds.close()

    def test_empty_chunk_ids(self, tmp_path):
        conn = sqlite3.connect(":memory:")
        result = load_sparse_for_chunks(conn, [])
        assert result == {}
        conn.close()


class TestRankCandidates:
    def test_ranks_by_sparse_score(self):
        t1, w1 = _make_blob([1], [0.9])
        t2, w2 = _make_blob([1], [0.3])
        sparse_data = {"c1": (t1, w1), "c2": (t2, w2)}
        ranking = rank_candidates_by_sparse({1: 1.0}, ["c1", "c2"], sparse_data)
        assert ranking[0]["id"] == "c1"
        assert ranking[0]["sparse_score"] > ranking[1]["sparse_score"]

    def test_missing_blob_gets_zero_score(self):
        sparse_data = {"c1": (None, None)}
        ranking = rank_candidates_by_sparse({1: 1.0}, ["c1"], sparse_data)
        assert ranking[0]["sparse_score"] == 0.0
