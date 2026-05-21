"""Tests for storage/metadata_store.py — MetadataStore class."""

import sqlite3
import pytest

from rag_lab.storage.metadata_store import MetadataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_store(tmp_path) -> MetadataStore:
    db_path = tmp_path / "meta.sqlite"
    store = MetadataStore(db_path=db_path)
    store.initialize()
    return store


def _table_names(conn: sqlite3.Connection) -> set:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# Schema / initialization
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_initialize_creates_tables(self, tmp_path):
        store = _fresh_store(tmp_path)
        tables = _table_names(store._conn)
        for expected in ("sources", "documents", "tags", "document_tags"):
            assert expected in tables, f"Missing table: {expected}"
        store.close()


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------

class TestUpsertDocument:
    def test_upsert_and_get_document(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_document(
            "doc1",
            title="My Doc",
            path="/data/my_doc.md",
            content_hash="abc123",
            source_id=None,
            status="active",
            embedding_model_version="2024-09",
            embedding_dim=1024,
            sparse_format_version=1,
        )
        doc = store.get_document("doc1")
        assert doc is not None
        assert doc["doc_id"] == "doc1"
        assert doc["title"] == "My Doc"
        assert doc["path"] == "/data/my_doc.md"
        assert doc["content_hash"] == "abc123"
        assert doc["status"] == "active"
        assert doc["embedding_model_version"] == "2024-09"
        assert doc["embedding_dim"] == 1024
        assert doc["sparse_format_version"] == 1
        assert doc["tags"] == []
        store.close()

    def test_upsert_is_idempotent(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_document("doc1", title="First")
        store.upsert_document("doc1", title="Updated")

        count = store._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_id = 'doc1'"
        ).fetchone()[0]
        assert count == 1

        doc = store.get_document("doc1")
        assert doc["title"] == "Updated"
        store.close()

    def test_get_document_not_found(self, tmp_path):
        store = _fresh_store(tmp_path)
        assert store.get_document("nonexistent") is None
        store.close()


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------

class TestListDocuments:
    @pytest.fixture
    def populated(self, tmp_path):
        store = _fresh_store(tmp_path)

        # Must create source row before referencing it in documents (FK constraint)
        store.upsert_source("sdmx", "SDMX Registry")

        store.upsert_document("doc_a", title="A", status="active", source_id="sdmx")
        store.upsert_document("doc_b", title="B", status="active", source_id="sdmx")
        store.upsert_document("doc_c", title="C", status="archived")
        store.upsert_document("doc_d", title="D", status="active")
        store.assign_tag("doc_a", "glossary")
        store.assign_tag("doc_b", "guide")
        store.assign_tag("doc_d", "guide")

        yield store
        store.close()

    def test_list_documents_all(self, populated):
        docs = populated.list_documents()
        ids = {d["doc_id"] for d in docs}
        assert ids == {"doc_a", "doc_b", "doc_d"}  # doc_c is archived, status=active default

    def test_list_documents_by_tag(self, populated):
        docs = populated.list_documents(tag="guide")
        ids = {d["doc_id"] for d in docs}
        assert ids == {"doc_b", "doc_d"}

    def test_list_documents_by_source(self, populated):
        docs = populated.list_documents(source_id="sdmx")
        ids = {d["doc_id"] for d in docs}
        assert ids == {"doc_a", "doc_b"}

    def test_list_documents_all_statuses(self, populated):
        docs = populated.list_documents(status=None)
        ids = {d["doc_id"] for d in docs}
        assert ids == {"doc_a", "doc_b", "doc_c", "doc_d"}


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

class TestTagOperations:
    def test_assign_and_unassign_tag(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_document("doc1")
        store.assign_tag("doc1", "important")
        assert "important" in store.get_tags_for_doc("doc1")
        store.unassign_tag("doc1", "important")
        assert "important" not in store.get_tags_for_doc("doc1")
        store.close()

    def test_assign_tag_idempotent(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_document("doc1")
        store.assign_tag("doc1", "dup")
        store.assign_tag("doc1", "dup")  # should not raise or duplicate
        tags = store.get_tags_for_doc("doc1")
        assert tags.count("dup") == 1
        store.close()

    def test_get_tags_for_doc(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_document("doc1")
        store.assign_tag("doc1", "beta")
        store.assign_tag("doc1", "alpha")
        tags = store.get_tags_for_doc("doc1")
        assert set(tags) == {"alpha", "beta"}
        store.close()

    def test_list_tags_with_count(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_document("doc1")
        store.upsert_document("doc2")
        store.assign_tag("doc1", "shared")
        store.assign_tag("doc2", "shared")
        store.assign_tag("doc1", "solo")
        tags = store.list_tags()
        by_name = {t["name"]: t for t in tags}
        assert by_name["shared"]["doc_count"] == 2
        assert by_name["solo"]["doc_count"] == 1
        store.close()

    def test_rename_tag(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_document("doc1")
        store.assign_tag("doc1", "old_name")
        result = store.rename_tag("old_name", "new_name")
        assert result is True
        tags = store.get_tags_for_doc("doc1")
        assert "new_name" in tags
        assert "old_name" not in tags
        store.close()

    def test_rename_tag_returns_false_if_missing(self, tmp_path):
        store = _fresh_store(tmp_path)
        result = store.rename_tag("nonexistent", "whatever")
        assert result is False
        store.close()

    def test_delete_tag_cascades(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_document("doc1")
        store.assign_tag("doc1", "to_delete")

        # Confirm the document_tags row exists before deletion
        row = store._conn.execute(
            "SELECT COUNT(*) FROM document_tags dt "
            "JOIN tags t ON dt.tag_id = t.tag_id WHERE t.name = 'to_delete'"
        ).fetchone()[0]
        assert row == 1

        store._conn.execute("PRAGMA foreign_keys = ON")
        store.delete_tag("to_delete")

        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM document_tags dt "
            "LEFT JOIN tags t ON dt.tag_id = t.tag_id WHERE t.name = 'to_delete'"
        ).fetchone()[0]
        assert remaining == 0
        store.close()

    def test_delete_document_cascades_tags(self, tmp_path):
        db_path = tmp_path / "meta.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        store = MetadataStore(conn=conn)
        store.initialize()
        store.upsert_document("doc1")
        store.assign_tag("doc1", "mytag")
        conn.commit()

        # Verify the assignment exists
        count_before = conn.execute(
            "SELECT COUNT(*) FROM document_tags WHERE doc_id = 'doc1'"
        ).fetchone()[0]
        assert count_before == 1

        store.delete_document("doc1")
        conn.commit()

        count_after = conn.execute(
            "SELECT COUNT(*) FROM document_tags WHERE doc_id = 'doc1'"
        ).fetchone()[0]
        assert count_after == 0
        conn.close()


# ---------------------------------------------------------------------------
# count_documents
# ---------------------------------------------------------------------------

class TestCountDocuments:
    def test_count_documents(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_document("d1", status="active")
        store.upsert_document("d2", status="active")
        store.upsert_document("d3", status="archived")

        assert store.count_documents(status="active") == 2
        assert store.count_documents(status="archived") == 1
        assert store.count_documents(status=None) == 3
        store.close()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class TestSources:
    def test_upsert_source_and_list(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.upsert_source(
            "sdmx",
            "SDMX Registry",
            description="Official SDMX docs",
            url="https://sdmx.org",
        )
        sources = store.list_sources()
        assert len(sources) == 1
        s = sources[0]
        assert s["source_id"] == "sdmx"
        assert s["name"] == "SDMX Registry"
        assert s["description"] == "Official SDMX docs"
        assert s["url"] == "https://sdmx.org"
        assert s["doc_count"] == 0
        store.close()


# ---------------------------------------------------------------------------
# Shared connection
# ---------------------------------------------------------------------------

class TestSharedConnection:
    def test_shared_connection(self, tmp_path):
        db_path = tmp_path / "shared.sqlite"
        conn = sqlite3.connect(str(db_path))

        store = MetadataStore(conn=conn)
        store.initialize()
        store.upsert_document("doc_shared", title="Shared")
        conn.commit()

        # store should NOT close the connection when close() is called
        store.close()
        # The connection should still be usable
        row = conn.execute(
            "SELECT title FROM documents WHERE doc_id = 'doc_shared'"
        ).fetchone()
        assert row is not None
        assert row[0] == "Shared"
        conn.close()
