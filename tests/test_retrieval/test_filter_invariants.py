"""RAG-03 regression: filter invariants across the retrieval chain.

Invariants:
  * ``doc_ids=None``            -> search everything.
  * valid filter with matches   -> only allowed documents.
  * valid filter with **no** matches -> **zero** results (never "all").

Uses real SQLite/FTS stores and a real (temporary) ChromaDB collection.
"""

import sqlite3

import numpy as np
import pytest

from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore
from rag_lab.retrieval.filters import FilterSpec, resolve_filter
from rag_lab.retrieval.hybrid_search import hybrid_search


@pytest.fixture
def real_stores(tmp_path):
    """Two documents, one chunk each, in real SQLite/FTS + Chroma."""
    db = tmp_path / "docstore.sqlite"
    ds = DocStore(db_path=db)
    ds.initialize()

    # Metadata: two active documents, one tagged.
    ds._conn.execute("INSERT INTO documents(doc_id, title, status) VALUES ('A','Doc A','active')")
    ds._conn.execute("INSERT INTO documents(doc_id, title, status) VALUES ('B','Doc B','active')")
    ds._conn.commit()
    from rag_lab.storage.metadata_store import MetadataStore

    MetadataStore(conn=ds._conn).assign_tag("A", "keep")

    chunks = [
        {"chunk_id": "a1", "doc_id": "A", "text": "alpha sdmx content",
         "heading_path": "H", "line_start": 1, "line_end": 2, "n_tokens": 3},
        {"chunk_id": "b1", "doc_id": "B", "text": "beta sdmx content",
         "heading_path": "H", "line_start": 1, "line_end": 2, "n_tokens": 3},
    ]
    ds.add(chunks)

    fts = FTSStore(db_path=db)
    fts.initialize()

    vs = VectorStore(storage_path=tmp_path / "chroma")
    vs.initialize()
    vs.add(
        ids=["a1", "b1"],
        embeddings=np.array([[1.0] * 1024, [0.5] * 1024], dtype=np.float32),
        documents=["alpha sdmx content", "beta sdmx content"],
        metadatas=[{"doc_id": "A", "heading_path": "H"}, {"doc_id": "B", "heading_path": "H"}],
    )
    yield ds, fts, vs
    ds.close()
    fts.close()


def _run(ds, fts, vs, **kwargs):
    return hybrid_search(
        "sdmx", vs, ds, fts,
        query_dense=np.ones(1024, dtype=np.float32),
        query_sparse=None,
        top_k=10,
        diversity_mode="off",
        **kwargs,
    )


class TestStoreLevel:
    def test_vector_none_is_unfiltered(self, real_stores):
        ds, fts, vs = real_stores
        r = vs.query(np.ones(1024, dtype=np.float32), 10, doc_ids=None)
        assert set(r["ids"]) == {"a1", "b1"}

    def test_vector_empty_is_zero(self, real_stores):
        ds, fts, vs = real_stores
        r = vs.query(np.ones(1024, dtype=np.float32), 10, doc_ids=[])
        assert r["ids"] == []

    def test_fts_none_is_unfiltered(self, real_stores):
        ds, fts, vs = real_stores
        r = fts.query("sdmx", 10, doc_ids=None)
        assert {x["id"] for x in r} == {"a1", "b1"}

    def test_fts_empty_is_zero(self, real_stores):
        ds, fts, vs = real_stores
        assert fts.query("sdmx", 10, doc_ids=[]) == []


class TestHybridInvariants:
    def test_no_filter_returns_all(self, real_stores):
        ds, fts, vs = real_stores
        got = _run(ds, fts, vs, doc_ids=None)
        assert {c["doc_id"] for c in got} == {"A", "B"}

    def test_explicit_doc_ids_restricts(self, real_stores):
        ds, fts, vs = real_stores
        got = _run(ds, fts, vs, doc_ids=["A"])
        assert {c["doc_id"] for c in got} == {"A"}

    def test_empty_doc_ids_returns_zero(self, real_stores):
        ds, fts, vs = real_stores
        assert _run(ds, fts, vs, doc_ids=[]) == []

    def test_nonexistent_doc_id_returns_zero(self, real_stores):
        ds, fts, vs = real_stores
        assert _run(ds, fts, vs, doc_ids=["does-not-exist"]) == []

    def test_filter_spec_no_match_returns_zero(self, real_stores):
        ds, fts, vs = real_stores
        spec = FilterSpec(source_id="nonexistent-source", status=None)
        assert resolve_filter(ds._conn, spec) == []
        assert _run(ds, fts, vs, filter_spec=spec) == []

    def test_filter_spec_empty_doc_ids_returns_zero(self, real_stores):
        ds, fts, vs = real_stores
        spec = FilterSpec(doc_ids=[], status=None)
        assert spec.is_empty() is False
        assert resolve_filter(ds._conn, spec) == []
        assert _run(ds, fts, vs, filter_spec=spec) == []

    def test_filter_spec_tag_positive(self, real_stores):
        ds, fts, vs = real_stores
        spec = FilterSpec(tags_include=["keep"], status=None)
        assert resolve_filter(ds._conn, spec) == ["A"]
        got = _run(ds, fts, vs, filter_spec=spec)
        assert {c["doc_id"] for c in got} == {"A"}

    def test_filter_spec_tag_no_match_returns_zero(self, real_stores):
        ds, fts, vs = real_stores
        spec = FilterSpec(tags_include=["missing-tag"], status=None)
        assert _run(ds, fts, vs, filter_spec=spec) == []
