"""Unit tests for IngestTransaction and IngestRunStore."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_lab.storage.docstore import DocStore
from rag_lab.ingest.transaction import IngestTransaction, IngestRunStore, _now
from rag_lab.exceptions import VersioningError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds(tmp_path):
    db = DocStore(db_path=tmp_path / "test.sqlite")
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def run_store(ds):
    return IngestRunStore(ds._conn)


def _chunk(chunk_id: str, doc_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": f"Text for {chunk_id}",
        "heading_path": "H1",
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
# IngestRunStore CRUD
# ---------------------------------------------------------------------------

class TestIngestRunStore:
    def test_create_and_get(self, run_store):
        run_store.create("run001", "doc_a", "/data/doc_a.md")
        r = run_store.get("run001")
        assert r["run_id"] == "run001"
        assert r["doc_id"] == "doc_a"
        assert r["status"] == "IN_PROGRESS"
        assert r["source_path"] == "/data/doc_a.md"
        assert r["chunks_expected"] == 0

    def test_get_nonexistent_returns_none(self, run_store):
        assert run_store.get("no_such_run") is None

    def test_update_fields(self, run_store):
        run_store.create("run002", "doc_b", None)
        run_store.update("run002", chunks_expected=10, chunks_written_chroma=10)
        r = run_store.get("run002")
        assert r["chunks_expected"] == 10
        assert r["chunks_written_chroma"] == 10

    def test_update_status_to_committed(self, run_store):
        run_store.create("run003", "doc_c", None)
        run_store.update("run003", status="COMMITTED", finished_at=_now())
        r = run_store.get("run003")
        assert r["status"] == "COMMITTED"
        assert r["finished_at"] is not None

    def test_list_runs_all(self, run_store):
        run_store.create("r1", "doc1", None)
        run_store.create("r2", "doc2", None)
        runs = run_store.list_runs()
        ids = [r["run_id"] for r in runs]
        assert "r1" in ids
        assert "r2" in ids

    def test_list_runs_by_status(self, run_store):
        run_store.create("r_ok", "doc_ok", None)
        run_store.create("r_fail", "doc_fail", None)
        run_store.update("r_fail", status="FAILED")
        failed = run_store.list_runs(status="FAILED")
        assert len(failed) == 1
        assert failed[0]["run_id"] == "r_fail"

    def test_list_runs_by_doc_id(self, run_store):
        run_store.create("rA", "docX", None)
        run_store.create("rB", "docY", None)
        runs = run_store.list_runs(doc_id="docX")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "rA"

    def test_get_failed(self, run_store):
        run_store.create("rf1", "d1", None)
        run_store.create("rf2", "d2", None)
        run_store.update("rf1", status="FAILED")
        failed = run_store.get_failed()
        assert any(r["run_id"] == "rf1" for r in failed)
        assert not any(r["run_id"] == "rf2" for r in failed)

    def test_get_stale_in_progress_empty_when_recent(self, run_store):
        run_store.create("recent_run", "doc_r", None)
        stale = run_store.get_stale_in_progress(minutes=30)
        # Run was just created — should not be stale
        assert not any(r["run_id"] == "recent_run" for r in stale)

    def test_get_stale_in_progress_detects_old(self, ds):
        """Insert a run with an old started_at directly to simulate a stale run."""
        ds._conn.execute(
            "INSERT INTO ingest_runs (run_id, doc_id, source_path, started_at, status) "
            "VALUES ('stale_run', 'doc_stale', NULL, datetime('now', '-2 hours'), 'IN_PROGRESS')"
        )
        ds._conn.commit()
        store = IngestRunStore(ds._conn)
        stale = store.get_stale_in_progress(minutes=30)
        assert any(r["run_id"] == "stale_run" for r in stale)


# ---------------------------------------------------------------------------
# IngestTransaction context manager
# ---------------------------------------------------------------------------

class TestIngestTransaction:
    def test_enter_creates_in_progress_run(self, ds):
        with IngestTransaction("doc_t", "/data/doc_t.md", ds) as txn:
            run_id = txn.run_id
            r = IngestRunStore(ds._conn).get(run_id)
            assert r["status"] == "IN_PROGRESS"

    def test_exit_success_marks_committed(self, ds):
        with IngestTransaction("doc_ok", "/data/doc_ok.md", ds) as txn:
            run_id = txn.run_id

        r = IngestRunStore(ds._conn).get(run_id)
        assert r["status"] == "COMMITTED"
        assert r["finished_at"] is not None

    def test_update_progress(self, ds):
        with IngestTransaction("doc_upd", None, ds) as txn:
            run_id = txn.run_id
            txn.update(chunks_expected=5, chunks_written_chroma=5)

        r = IngestRunStore(ds._conn).get(run_id)
        assert r["chunks_expected"] == 5
        assert r["chunks_written_chroma"] == 5

    def test_exit_exception_marks_failed_and_rolls_back(self, ds):
        """RAG-01: a failed replacement restores the previous committed version."""
        doc_id = "doc_fail_test"
        ds.add([_chunk("cx1", doc_id), _chunk("cx2", doc_id)])
        assert ds.count_chunks(doc_id) == 2

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = mock_vs

            run_id = None
            try:
                with IngestTransaction(
                    doc_id, None, ds,
                    vector_store=mock_vs, chunk_ids=["new1", "new2"],
                ) as txn:
                    run_id = txn.run_id
                    ds.add([_chunk("new1", doc_id), _chunk("new2", doc_id)])
                    raise RuntimeError("Injected failure")
            except RuntimeError:
                pass

        r = IngestRunStore(ds._conn).get(run_id)
        assert r["status"] == "ROLLED_BACK"
        assert "Injected failure" in (r["error_message"] or "")
        # RAG-01: previous version preserved, failed new version removed.
        ids = {
            row[0]
            for row in ds._conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
            )
        }
        assert ids == {"cx1", "cx2"}

    def test_exception_is_reraised(self, ds):
        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = mock_vs

            with pytest.raises(ValueError, match="sentinel"):
                with IngestTransaction("doc_reraise", None, ds):
                    raise ValueError("sentinel")

    def test_run_id_is_12_hex_chars(self, ds):
        with IngestTransaction("doc_id_check", None, ds) as txn:
            assert len(txn.run_id) == 12
            assert all(c in "0123456789abcdef" for c in txn.run_id)


# ---------------------------------------------------------------------------
# Rollback compensation
# ---------------------------------------------------------------------------

class TestRollbackCompensation:
    def test_rollback_without_handle_fails_closed(self, ds):
        """RAG-01: without a snapshot handle, rollback must NOT delete data."""
        doc_id = "doc_rb"
        ds.add([_chunk("rb1", doc_id), _chunk("rb2", doc_id)])
        assert ds.count_chunks(doc_id) == 2

        with pytest.raises(VersioningError):
            IngestTransaction(doc_id, None, ds).rollback()

        assert ds.count_chunks(doc_id) == 2

    def test_versioned_rollback_restores_previous(self, ds):
        """A failed replacement restores the previous committed version."""
        doc_id = "doc_rb_restore"
        ds.add([_chunk("old1", doc_id), _chunk("old2", doc_id)])

        txn = IngestTransaction(doc_id, None, ds, chunk_ids=["new1"])
        txn.__enter__()
        ds.add([_chunk("new1", doc_id)])
        txn.rollback()

        r = IngestRunStore(ds._conn).get(txn.run_id)
        assert r["status"] == "ROLLED_BACK"
        ids = {
            row[0]
            for row in ds._conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
            )
        }
        assert ids == {"old1", "old2"}
        fts_ids = {
            row[0]
            for row in ds._conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE doc_id = ?", (doc_id,)
            )
        }
        assert fts_ids == {"old1", "old2"}

    def test_versioned_rollback_is_idempotent(self, ds):
        doc_id = "doc_idem"
        ds.add([_chunk("id_old", doc_id)])
        txn = IngestTransaction(doc_id, None, ds, chunk_ids=["id_new"])
        txn.__enter__()
        ds.add([_chunk("id_new", doc_id)])
        txn.rollback()
        txn.rollback()  # second call must be safe
        ids = {
            row[0]
            for row in ds._conn.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
            )
        }
        assert ids == {"id_old"}

    def test_fresh_rollback_removes_new_data(self, ds):
        """First ingest that fails leaves no chunks/FTS/documents."""
        doc_id = "doc_fresh"
        txn = IngestTransaction(doc_id, None, ds, chunk_ids=["f1", "f2"])
        txn.__enter__()
        ds.add([_chunk("f1", doc_id), _chunk("f2", doc_id)])
        txn.rollback()
        assert ds.count_chunks(doc_id) == 0
        assert ds._conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0] == 0
        assert ds._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0] == 0
