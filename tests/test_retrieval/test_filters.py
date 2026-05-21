"""Tests for retrieval/filters.py — FilterSpec, resolve_filter, filter_stats."""

import sqlite3
import pytest

from rag_lab.retrieval.filters import FilterSpec, filter_stats, resolve_filter
from rag_lab.storage.metadata_store import MetadataStore


# ---------------------------------------------------------------------------
# Fixture: populated metadata DB
# ---------------------------------------------------------------------------

@pytest.fixture
def metadata_conn(tmp_path):
    """
    Open an in-memory-style temp SQLite connection with v3 tables.
    Inserts 4 test documents with doc_ids, tags, source_id.

    doc_glossary  — tag:glossary, source:sdmx, status:active
    doc_guide     — tag:guide,    source:sdmx, status:active
    doc_notas     — tag:guide,                 status:active
    doc_test      — tag:test,                  status:archived
    """
    db_path = tmp_path / "meta.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")

    meta = MetadataStore(conn=conn)
    meta.initialize()

    meta.upsert_source("sdmx", "SDMX Registry")
    conn.commit()

    meta.upsert_document("doc_glossary", source_id="sdmx", status="active")
    meta.upsert_document("doc_guide",    source_id="sdmx", status="active")
    meta.upsert_document("doc_notas",                      status="active")
    meta.upsert_document("doc_test",                       status="archived")
    conn.commit()

    meta.assign_tag("doc_glossary", "glossary")
    meta.assign_tag("doc_guide",    "guide")
    meta.assign_tag("doc_notas",    "guide")
    meta.assign_tag("doc_test",     "test")
    conn.commit()

    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# FilterSpec.is_empty
# ---------------------------------------------------------------------------

class TestFilterSpecIsEmpty:
    def test_empty_filter_returns_none(self, metadata_conn):
        # status must be None too — the default 'active' is itself a filter criterion
        spec = FilterSpec(status=None)
        assert spec.is_empty() is True
        result = resolve_filter(metadata_conn, spec)
        assert result is None

    def test_non_empty_when_doc_ids_set(self):
        spec = FilterSpec(doc_ids=["doc_glossary"])
        assert spec.is_empty() is False

    def test_non_empty_when_status_set(self):
        # status="active" (the default) still makes the spec non-empty
        spec = FilterSpec(status="active")
        assert spec.is_empty() is False

    def test_is_empty_when_status_none_and_no_other_criteria(self):
        spec = FilterSpec(status=None)
        assert spec.is_empty() is True


# ---------------------------------------------------------------------------
# resolve_filter
# ---------------------------------------------------------------------------

class TestResolveFilter:
    def test_filter_by_explicit_doc_ids(self, metadata_conn):
        spec = FilterSpec(doc_ids=["doc_glossary"], status=None)
        result = resolve_filter(metadata_conn, spec)
        assert result == ["doc_glossary"]

    def test_filter_by_tag_include_single(self, metadata_conn):
        spec = FilterSpec(tags_include=["glossary"], status=None)
        result = resolve_filter(metadata_conn, spec)
        assert result == ["doc_glossary"]

    def test_filter_by_tag_include_and_logic(self, metadata_conn):
        spec = FilterSpec(tags_include=["guide"], status=None)
        result = resolve_filter(metadata_conn, spec)
        assert sorted(result) == ["doc_guide", "doc_notas"]

    def test_filter_by_tag_include_multiple_and(self, metadata_conn):
        # No single document has BOTH "guide" and "test"
        spec = FilterSpec(tags_include=["guide", "test"], status=None)
        result = resolve_filter(metadata_conn, spec)
        assert result == []

    def test_filter_by_tag_exclude(self, metadata_conn):
        # Exclude "test" tag; default status="active" also filters out doc_test
        spec = FilterSpec(tags_exclude=["test"], status="active")
        result = resolve_filter(metadata_conn, spec)
        assert sorted(result) == ["doc_glossary", "doc_guide", "doc_notas"]

    def test_filter_by_source_id(self, metadata_conn):
        spec = FilterSpec(source_id="sdmx", status=None)
        result = resolve_filter(metadata_conn, spec)
        assert sorted(result) == ["doc_glossary", "doc_guide"]

    def test_combined_tag_and_source(self, metadata_conn):
        spec = FilterSpec(tags_include=["guide"], source_id="sdmx", status=None)
        result = resolve_filter(metadata_conn, spec)
        assert result == ["doc_guide"]

    def test_status_filter_excludes_archived(self, metadata_conn):
        spec = FilterSpec(status="active")
        result = resolve_filter(metadata_conn, spec)
        assert "doc_test" not in result
        assert sorted(result) == ["doc_glossary", "doc_guide", "doc_notas"]

    def test_status_filter_none_includes_all(self, metadata_conn):
        spec = FilterSpec(doc_ids=["doc_test"], status=None)
        result = resolve_filter(metadata_conn, spec)
        assert result == ["doc_test"]

    def test_filter_empty_result(self, metadata_conn):
        spec = FilterSpec(tags_include=["nonexistent_tag"], status=None)
        result = resolve_filter(metadata_conn, spec)
        assert result == []

    def test_doc_ids_intersects_with_tags(self, metadata_conn):
        # doc_guide exists but doesn't have the "glossary" tag
        spec = FilterSpec(doc_ids=["doc_guide"], tags_include=["glossary"], status=None)
        result = resolve_filter(metadata_conn, spec)
        assert result == []


# ---------------------------------------------------------------------------
# filter_stats
# ---------------------------------------------------------------------------

class TestFilterStats:
    def test_filter_stats_total_and_matched(self, metadata_conn):
        spec = FilterSpec(source_id="sdmx", status=None)
        stats = filter_stats(metadata_conn, spec)
        assert stats["total_documents"] == 4
        assert stats["matched_documents"] == 2
        assert sorted(stats["resolved_doc_ids"]) == ["doc_glossary", "doc_guide"]

    def test_filter_stats_empty_spec(self, metadata_conn):
        # status=None makes the spec truly empty; resolve_filter returns None
        spec = FilterSpec(status=None)
        stats = filter_stats(metadata_conn, spec)
        assert stats["total_documents"] == 4
        # empty spec → resolve_filter returns None → matched = total
        assert stats["matched_documents"] == stats["total_documents"]
        assert stats["resolved_doc_ids"] is None


# ---------------------------------------------------------------------------
# Graceful fallback when documents table is missing
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    def test_graceful_fallback_missing_documents_table(self, tmp_path):
        # A bare SQLite connection with no v3 tables at all
        db_path = tmp_path / "bare.sqlite"
        conn = sqlite3.connect(str(db_path))

        spec = FilterSpec(doc_ids=["doc_a", "doc_b"], status=None)
        result = resolve_filter(conn, spec)
        # Should return sorted explicit doc_ids without raising
        assert result == ["doc_a", "doc_b"]
        conn.close()

    def test_graceful_fallback_no_doc_ids_returns_none(self, tmp_path):
        db_path = tmp_path / "bare.sqlite"
        conn = sqlite3.connect(str(db_path))

        spec = FilterSpec(tags_include=["something"], status=None)
        result = resolve_filter(conn, spec)
        assert result is None
        conn.close()
