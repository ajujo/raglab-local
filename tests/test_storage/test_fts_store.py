"""Tests for storage/fts_store.py — FTS5 BM25 search."""

import pytest
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.docstore import DocStore


@pytest.fixture
def populated_db(tmp_path):
    """Create a DocStore with some chunks and return (db_path, chunk_ids)."""
    db_path = tmp_path / "test.sqlite"
    ds = DocStore(db_path=db_path)
    ds.initialize()
    ds.add([
        {
            "chunk_id": "c1", "doc_id": "doc_a", "text": "SDMX is a standard for data exchange.",
            "heading_path": "Intro", "tipo": "texto", "posicion_relativa": 0.1,
            "n_tokens": 10, "line_start": 1, "line_end": 5,
        },
        {
            "chunk_id": "c2", "doc_id": "doc_a", "text": "Metadata structure definitions in SDMX.",
            "heading_path": "Metadata", "tipo": "texto", "posicion_relativa": 0.3,
            "n_tokens": 8, "line_start": 6, "line_end": 10,
        },
        {
            "chunk_id": "c3", "doc_id": "doc_b", "text": "The glossary defines key terms.",
            "heading_path": "Glossary", "tipo": "texto", "posicion_relativa": 0.5,
            "n_tokens": 7, "line_start": 1, "line_end": 3,
        },
    ])
    ds.close()
    return db_path


class TestFTSStoreQuery:
    def test_basic_search_returns_results(self, populated_db):
        fts = FTSStore(db_path=populated_db)
        results = fts.query("SDMX standard", top_k=5)
        assert len(results) > 0
        ids = [r["id"] for r in results]
        assert "c1" in ids  # c1 has "SDMX" and "standard"

    def test_bm25_score_is_positive(self, populated_db):
        fts = FTSStore(db_path=populated_db)
        results = fts.query("SDMX", top_k=5)
        for r in results:
            assert r["bm25_score"] > 0

    def test_filter_by_doc_ids(self, populated_db):
        fts = FTSStore(db_path=populated_db)
        results = fts.query("SDMX", top_k=10, doc_ids=["doc_b"])
        # doc_b has "glossary" — SDMX is not in doc_b chunks, so expect 0 results
        for r in results:
            assert r["id"] == "c3"  # only doc_b chunks

    def test_empty_query_returns_empty(self, populated_db):
        fts = FTSStore(db_path=populated_db)
        results = fts.query("", top_k=5)
        assert results == []

    def test_no_match_returns_empty(self, populated_db):
        fts = FTSStore(db_path=populated_db)
        results = fts.query("xyzqwerty123nonexistent", top_k=5)
        assert results == []

    def test_top_k_limit(self, populated_db):
        fts = FTSStore(db_path=populated_db)
        results = fts.query("the", top_k=1)
        assert len(results) <= 1

    def test_escape_query_special_chars(self):
        """_escape_query strips FTS5 special chars and returns quoted tokens."""
        escaped = FTSStore._escape_query('SDMX "metadata" AND (structure OR format)')
        # Must be non-empty
        assert escaped
        # Every token must be wrapped in double quotes
        for token in escaped.split():
            assert token.startswith('"') and token.endswith('"'), (
                f"Token not quoted: {token}"
            )

    def test_result_has_id_and_score_keys(self, populated_db):
        fts = FTSStore(db_path=populated_db)
        results = fts.query("SDMX", top_k=3)
        for r in results:
            assert "id" in r
            assert "bm25_score" in r
