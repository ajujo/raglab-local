"""Tests for the canonical SQLite-BLOB sparse scorer (v1.18.2+).

Replaces test_sparse_store.py — the JSON-backed SparseStore was removed.
Sparse vectors now live exclusively as BLOB columns in docstore.sqlite.
"""

import sqlite3
import struct

import numpy as np
import pytest

from rag_lab.retrieval.sparse_scorer import (
    load_sparse_for_chunks,
    rank_candidates_by_sparse,
    sparse_score,
)


def _make_blobs(token_weights: dict) -> tuple:
    """Create (tokens_blob, weights_blob) from {token_id: weight}."""
    tokens = np.array(list(token_weights.keys()), dtype=np.int32)
    weights = np.array(list(token_weights.values()), dtype=np.float32)
    return tokens.tobytes(), weights.tobytes()


def _conn_with_chunks(chunks: list[dict]) -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the chunks schema populated."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            sparse_tokens BLOB,
            sparse_weights BLOB
        )"""
    )
    for c in chunks:
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?)",
            (c["chunk_id"], c.get("tokens"), c.get("weights")),
        )
    conn.commit()
    return conn


class TestLoadSparseForChunks:
    def test_loads_blobs_for_requested_ids(self):
        tokens_b, weights_b = _make_blobs({1: 0.9, 2: 0.5})
        conn = _conn_with_chunks([
            {"chunk_id": "c1", "tokens": tokens_b, "weights": weights_b},
            {"chunk_id": "c2", "tokens": None, "weights": None},
        ])

        result = load_sparse_for_chunks(conn, ["c1"])
        assert "c1" in result
        assert result["c1"][0] == tokens_b
        assert result["c1"][1] == weights_b

    def test_returns_none_blobs_for_missing_sparse(self):
        conn = _conn_with_chunks([{"chunk_id": "cx", "tokens": None, "weights": None}])
        result = load_sparse_for_chunks(conn, ["cx"])
        assert result["cx"] == (None, None)

    def test_empty_chunk_ids_returns_empty_dict(self):
        conn = _conn_with_chunks([])
        assert load_sparse_for_chunks(conn, []) == {}

    def test_unknown_ids_absent_from_result(self):
        conn = _conn_with_chunks([])
        result = load_sparse_for_chunks(conn, ["nonexistent"])
        assert "nonexistent" not in result

    def test_graceful_on_missing_columns(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO chunks VALUES ('c1')")
        conn.commit()
        result = load_sparse_for_chunks(conn, ["c1"])
        assert result == {}


class TestSparseScore:
    def test_exact_token_match(self):
        tokens_b, weights_b = _make_blobs({10: 1.0, 20: 0.5})
        score = sparse_score({10: 1.0, 20: 0.5}, tokens_b, weights_b)
        assert abs(score - 1.25) < 1e-4

    def test_partial_overlap(self):
        tokens_b, weights_b = _make_blobs({10: 1.0, 99: 0.8})
        score = sparse_score({10: 1.0}, tokens_b, weights_b)
        assert abs(score - 1.0) < 1e-4

    def test_no_overlap_gives_zero(self):
        tokens_b, weights_b = _make_blobs({5: 1.0})
        score = sparse_score({999: 1.0}, tokens_b, weights_b)
        assert score == 0.0

    def test_none_blobs_give_zero(self):
        assert sparse_score({1: 1.0}, None, None) == 0.0

    def test_empty_query_gives_zero(self):
        tokens_b, weights_b = _make_blobs({1: 0.9})
        assert sparse_score({}, tokens_b, weights_b) == 0.0


class TestRankCandidatesBySparse:
    def test_ranked_descending_by_score(self):
        tokens_b_high, weights_b_high = _make_blobs({1: 1.0, 2: 1.0})
        tokens_b_low, weights_b_low = _make_blobs({1: 0.1})

        sparse_data = {
            "low": (tokens_b_low, weights_b_low),
            "high": (tokens_b_high, weights_b_high),
        }
        query = {1: 1.0, 2: 1.0}
        ranked = rank_candidates_by_sparse(query, ["low", "high"], sparse_data)

        assert ranked[0]["id"] == "high"
        assert ranked[1]["id"] == "low"
        assert ranked[0]["sparse_score"] > ranked[1]["sparse_score"]

    def test_missing_candidate_scores_zero(self):
        ranked = rank_candidates_by_sparse({1: 1.0}, ["ghost"], {})
        assert ranked[0]["sparse_score"] == 0.0

    def test_empty_candidates(self):
        assert rank_candidates_by_sparse({1: 1.0}, [], {}) == []


class TestScopeGuard:
    """Invariants that must hold for the v1.18.2 architecture."""

    def test_sparse_store_module_deleted(self):
        """rag_lab.storage.sparse_store must not exist."""
        import importlib
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rag_lab.storage.sparse_store")

    def test_sparse_store_not_in_storage_init(self):
        """SparseStore must not be exported from rag_lab.storage."""
        import rag_lab.storage as storage_pkg
        assert not hasattr(storage_pkg, "SparseStore")

    def test_sparse_index_path_removed_from_config(self):
        """SPARSE_INDEX_PATH must not exist in rag_lab.config."""
        import rag_lab.config as config
        assert not hasattr(config, "SPARSE_INDEX_PATH")

    def test_hybrid_search_uses_sqlite_not_json(self):
        """hybrid_search must call load_sparse_for_chunks, not SparseStore."""
        import inspect
        import rag_lab.retrieval.hybrid_search as hs
        src = inspect.getsource(hs)
        assert "load_sparse_for_chunks" in src
        assert "SparseStore" not in src
        assert "sparse_index.json" not in src

    def test_save_sparse_index_removed_from_encoder(self):
        """save_sparse_index() must not exist in rag_lab.embedding.encoder."""
        import rag_lab.embedding.encoder as enc
        assert not hasattr(enc, "save_sparse_index")
