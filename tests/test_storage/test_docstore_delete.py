"""Tests for DocStore.delete_by_doc_id() (v1.3 addition)."""

import sqlite3
import pytest

from rag_lab.storage.docstore import DocStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path) -> DocStore:
    db_path = tmp_path / "test.sqlite"
    ds = DocStore(db_path=db_path)
    ds.initialize()
    return ds


def _chunk(chunk_id: str, doc_id: str, text: str = "some text") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "heading_path": "Section",
        "tipo": "texto",
        "posicion_relativa": 0.1,
        "n_tokens": 5,
        "line_start": 0,
        "line_end": 4,
        "embedding_model_name": "bge",
        "embedding_model_version": "2024-09",
        "embedding_dim": 1024,
        "sparse_format_version": 1,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeleteByDocId:
    def test_delete_by_doc_id_removes_chunks(self, tmp_path):
        ds = _make_store(tmp_path)
        ds.add([
            _chunk("c1", "doc1"),
            _chunk("c2", "doc1"),
            _chunk("c3", "doc2"),
        ])

        ds.delete_by_doc_id("doc1")

        assert ds.count_chunks("doc1") == 0
        assert ds.count_chunks("doc2") == 1
        # Raw check via get_by_id also confirms removal
        assert ds.get_by_id("c1") is None
        assert ds.get_by_id("c2") is None
        assert ds.get_by_id("c3") is not None
        ds.close()

    def test_delete_by_doc_id_removes_fts5(self, tmp_path):
        ds = _make_store(tmp_path)
        ds.add([
            _chunk("c1", "doc_to_delete", "alpha bravo charlie"),
            _chunk("c2", "doc_to_keep", "delta echo foxtrot"),
        ])

        ds.delete_by_doc_id("doc_to_delete")

        rows = ds._conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE doc_id = 'doc_to_delete'"
        ).fetchall()
        assert rows == [], "FTS5 should have no rows for the deleted doc_id"

        kept = ds._conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE doc_id = 'doc_to_keep'"
        ).fetchall()
        assert len(kept) == 1
        ds.close()

    def test_delete_by_doc_id_returns_count(self, tmp_path):
        ds = _make_store(tmp_path)
        ds.add([
            _chunk("c1", "doc1"),
            _chunk("c2", "doc1"),
            _chunk("c3", "doc1"),
        ])

        deleted = ds.delete_by_doc_id("doc1")
        assert deleted == 3
        ds.close()

    def test_delete_nonexistent_doc_returns_zero(self, tmp_path):
        ds = _make_store(tmp_path)
        ds.add([_chunk("c1", "doc1")])

        deleted = ds.delete_by_doc_id("doc_that_does_not_exist")
        assert deleted == 0
        # Existing data untouched
        assert ds.count() == 1
        ds.close()
