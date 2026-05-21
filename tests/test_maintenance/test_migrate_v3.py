"""Tests for maintenance/migrate_to_v3.py — migrate() function."""

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch

from rag_lab.storage.docstore import DocStore
from rag_lab.storage.metadata_store import MetadataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docstore(tmp_path: Path, chunks: list) -> tuple[Path, DocStore]:
    """Create a DocStore at tmp_path and populate it with chunks."""
    db_path = tmp_path / "docstore.sqlite"
    ds = DocStore(db_path=db_path)
    ds.initialize()
    if chunks:
        ds.add(chunks)
    return db_path, ds


def _minimal_chunks(doc_ids: list) -> list:
    chunks = []
    for i, doc_id in enumerate(doc_ids):
        chunks.append({
            "chunk_id": f"c{i}_{doc_id}",
            "doc_id": doc_id,
            "text": f"Text for {doc_id}",
            "heading_path": "Section",
            "tipo": "texto",
            "posicion_relativa": 0.1 * (i + 1),
            "n_tokens": 5,
            "line_start": i * 5,
            "line_end": i * 5 + 4,
            "embedding_model_name": "bge",
            "embedding_model_version": "2024-09",
            "embedding_dim": 1024,
            "sparse_format_version": 1,
        })
    return chunks


def _make_doc_manager_db(path: Path, doc_rows: list, tag_rows: list) -> None:
    """Create a minimal doc_manager.db with documents and tags tables."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, path TEXT, hash TEXT)"
    )
    conn.execute(
        "CREATE TABLE tags (doc_id TEXT, tag_name TEXT)"
    )
    conn.executemany("INSERT INTO documents VALUES (?, ?, ?)", doc_rows)
    conn.executemany("INSERT INTO tags VALUES (?, ?)", tag_rows)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMigrationCreatesTablesV3:
    def test_migration_creates_tables(self, tmp_path):
        db_path, ds = _make_docstore(tmp_path, _minimal_chunks(["doc_a"]))
        ds.close()

        # migrate() calls DocStore() with no arg, so patch the default path
        with patch("rag_lab.maintenance.migrate_to_v3.DocStore") as MockDS, \
             patch("rag_lab.maintenance.migrate_to_v3.DOC_MANAGER_DB_PATH", tmp_path / "nonexistent.db"):

            real_ds = DocStore(db_path=db_path)
            real_ds.initialize()
            MockDS.return_value = real_ds

            from rag_lab.maintenance.migrate_to_v3 import migrate
            migrate()

        conn = sqlite3.connect(str(db_path))
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()

        for expected in ("documents", "tags", "document_tags"):
            assert expected in tables, f"Missing table after migration: {expected}"


class TestMigrationPopulatesDocuments:
    def test_migration_populates_documents_from_chunks(self, tmp_path):
        doc_ids = ["doc_alpha", "doc_beta", "doc_gamma"]
        # doc_alpha has 2 chunks; others have 1 each
        chunks = _minimal_chunks(doc_ids)
        chunks.append({
            "chunk_id": "c_extra_alpha",
            "doc_id": "doc_alpha",
            "text": "Second chunk for alpha",
            "heading_path": "Section",
            "tipo": "texto",
            "posicion_relativa": 0.9,
            "n_tokens": 4,
            "line_start": 100,
            "line_end": 104,
            "embedding_model_name": "bge",
            "embedding_model_version": "2024-09",
            "embedding_dim": 1024,
            "sparse_format_version": 1,
        })
        db_path, ds = _make_docstore(tmp_path, chunks)
        ds.close()

        with patch("rag_lab.maintenance.migrate_to_v3.DocStore") as MockDS, \
             patch("rag_lab.maintenance.migrate_to_v3.DOC_MANAGER_DB_PATH", tmp_path / "nonexistent.db"):

            real_ds = DocStore(db_path=db_path)
            real_ds.initialize()
            MockDS.return_value = real_ds

            from rag_lab.maintenance.migrate_to_v3 import migrate
            result = migrate()

        assert result["documents_inserted"] == 3
        assert result["final_documents"] == 3

        conn = sqlite3.connect(str(db_path))
        doc_id_rows = conn.execute("SELECT doc_id FROM documents").fetchall()
        conn.close()
        found_ids = {r[0] for r in doc_id_rows}
        assert found_ids == set(doc_ids)


class TestMigrationIdempotent:
    def test_migration_is_idempotent(self, tmp_path):
        db_path, ds = _make_docstore(tmp_path, _minimal_chunks(["doc_x", "doc_y"]))
        ds.close()

        def make_real_ds():
            real_ds = DocStore(db_path=db_path)
            real_ds.initialize()
            return real_ds

        # Run twice
        for _ in range(2):
            with patch("rag_lab.maintenance.migrate_to_v3.DocStore") as MockDS, \
                 patch("rag_lab.maintenance.migrate_to_v3.DOC_MANAGER_DB_PATH", tmp_path / "nonexistent.db"):
                MockDS.return_value = make_real_ds()
                from rag_lab.maintenance.migrate_to_v3 import migrate
                migrate()

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        assert count == 2


class TestMigrationDocManagerTags:
    def test_migration_migrates_doc_manager_tags(self, tmp_path):
        db_path, ds = _make_docstore(tmp_path, _minimal_chunks(["doc_p", "doc_q"]))
        ds.close()

        mgr_db = tmp_path / "doc_manager.db"
        _make_doc_manager_db(
            mgr_db,
            doc_rows=[
                ("doc_p", "/path/to/p.md", "hashp"),
                ("doc_q", "/path/to/q.md", "hashq"),
            ],
            tag_rows=[
                ("doc_p", "important"),
                ("doc_q", "important"),
                ("doc_p", "featured"),
            ],
        )

        with patch("rag_lab.maintenance.migrate_to_v3.DocStore") as MockDS, \
             patch("rag_lab.maintenance.migrate_to_v3.DOC_MANAGER_DB_PATH", mgr_db):
            real_ds = DocStore(db_path=db_path)
            real_ds.initialize()
            MockDS.return_value = real_ds

            from rag_lab.maintenance.migrate_to_v3 import migrate
            result = migrate()

        assert result["docs_migrated_from_manager"] == 2
        assert result["tags_migrated"] == 3

        conn = sqlite3.connect(str(db_path))
        tag_count = conn.execute("SELECT COUNT(*) FROM document_tags").fetchone()[0]
        conn.close()
        assert tag_count == 3

    def test_migration_skips_missing_doc_manager(self, tmp_path):
        db_path, ds = _make_docstore(tmp_path, _minimal_chunks(["doc_only"]))
        ds.close()

        missing_path = tmp_path / "does_not_exist.db"

        with patch("rag_lab.maintenance.migrate_to_v3.DocStore") as MockDS, \
             patch("rag_lab.maintenance.migrate_to_v3.DOC_MANAGER_DB_PATH", missing_path):
            real_ds = DocStore(db_path=db_path)
            real_ds.initialize()
            MockDS.return_value = real_ds

            from rag_lab.maintenance.migrate_to_v3 import migrate
            result = migrate()

        assert result["docs_migrated_from_manager"] == 0
        assert result["tags_migrated"] == 0
        assert result["final_documents"] == 1
