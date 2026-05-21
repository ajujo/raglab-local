"""Unit tests for IngestTransaction and IngestRunStore."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_lab.storage.docstore import DocStore
from rag_lab.ingest.transaction import IngestTransaction, IngestRunStore, _now


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
        """On exception, run is marked FAILED then ROLLED_BACK after compensation."""
        doc_id = "doc_fail_test"
        ds.add([_chunk("cx1", doc_id), _chunk("cx2", doc_id)])
        assert ds.count_chunks(doc_id) == 2

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = mock_vs

            run_id = None
            try:
                with IngestTransaction(doc_id, None, ds) as txn:
                    run_id = txn.run_id
                    raise RuntimeError("Injected failure")
            except RuntimeError:
                pass

        r = IngestRunStore(ds._conn).get(run_id)
        assert r["status"] == "ROLLED_BACK"
        assert "Injected failure" in (r["error_message"] or "")
        # Chunks should be gone
        assert ds.count_chunks(doc_id) == 0

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
    def test_rollback_deletes_chunks(self, ds):
        doc_id = "doc_rb"
        ds.add([_chunk("rb1", doc_id), _chunk("rb2", doc_id)])
        assert ds.count_chunks(doc_id) == 2

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.delete_by_doc_id.return_value = 2
            mock_vs_cls.return_value = mock_vs

            txn = IngestTransaction(doc_id, None, ds)
            txn.run_id = "manual_test_run"
            IngestRunStore(ds._conn).create("manual_test_run", doc_id, None)
            txn.rollback()

        assert ds.count_chunks(doc_id) == 0
        mock_vs.delete_by_doc_id.assert_called_once_with(doc_id)

    def test_rollback_marks_run_rolled_back(self, ds):
        IngestRunStore(ds._conn).create("rb_run_01", "doc_rb_status", None)

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = mock_vs

            txn = IngestTransaction("doc_rb_status", None, ds)
            txn.run_id = "rb_run_01"
            txn.rollback()

        r = IngestRunStore(ds._conn).get("rb_run_01")
        assert r["status"] == "ROLLED_BACK"

    def test_rollback_is_idempotent(self, ds):
        """Second rollback call is safe even when data is already gone."""
        doc_id = "doc_idem"
        ds.add([_chunk("id1", doc_id)])

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.delete_by_doc_id.return_value = 1
            mock_vs_cls.return_value = mock_vs

            IngestRunStore(ds._conn).create("rb_idem_run", doc_id, None)
            txn = IngestTransaction(doc_id, None, ds)
            txn.run_id = "rb_idem_run"

            txn.rollback()  # first call
            txn.rollback()  # second call — should not raise

        assert ds.count_chunks(doc_id) == 0

    def test_rollback_cleans_fts5(self, ds):
        doc_id = "doc_fts_rb"
        ds.add([_chunk("fts_rb_1", doc_id)])

        fts_before = ds._conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
        assert fts_before == 1

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.delete_by_doc_id.return_value = 1
            mock_vs_cls.return_value = mock_vs

            IngestRunStore(ds._conn).create("fts_rb_run", doc_id, None)
            txn = IngestTransaction(doc_id, None, ds)
            txn.run_id = "fts_rb_run"
            txn.rollback()

        fts_after = ds._conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
        assert fts_after == 0

    def test_rollback_also_cleans_documents_table(self, ds):
        doc_id = "doc_meta_rb"
        from rag_lab.storage.metadata_store import MetadataStore
        MetadataStore(conn=ds._conn).upsert_document(doc_id)

        exists = ds._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
        assert exists == 1

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = mock_vs

            IngestRunStore(ds._conn).create("meta_rb_run", doc_id, None)
            txn = IngestTransaction(doc_id, None, ds)
            txn.run_id = "meta_rb_run"
            txn.rollback()

        gone = ds._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
        assert gone == 0
