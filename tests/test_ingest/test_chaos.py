"""Chaos tests: inject failures at each ingest stage and verify rollback.

Each test injects a failure at a specific point in _ingest_one() and asserts
that the system is left in a clean, consistent state (no partial data).
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_lab.storage.docstore import DocStore
from rag_lab.ingest.transaction import IngestRunStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds(tmp_path):
    db = DocStore(db_path=tmp_path / "chaos.sqlite")
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def mock_vector_store():
    """A VectorStore mock that captures add/delete calls."""
    vs = MagicMock()
    vs.delete_by_doc_id.return_value = 0
    return vs


def _make_chunk_dicts(doc_id: str, n: int = 3) -> list:
    return [
        {
            "chunk_id": f"{doc_id}_chunk_{i}",
            "doc_id": doc_id,
            "text": f"Chunk {i} text",
            "heading_path": "H",
            "tipo": "texto",
            "posicion_relativa": i / n,
            "n_tokens": 5,
            "line_start": i * 5,
            "line_end": i * 5 + 4,
            "embedding_model_name": "bge",
            "embedding_model_version": "2024-09",
            "embedding_dim": 1024,
            "sparse_format_version": 1,
            "sparse_tokens": None,
            "sparse_weights": None,
        }
        for i in range(n)
    ]


def _count_chunks(ds: DocStore, doc_id: str) -> int:
    return ds._conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0]


def _count_fts(ds: DocStore, doc_id: str) -> int:
    return ds._conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0]


def _latest_run(ds: DocStore, doc_id: str) -> dict:
    store = IngestRunStore(ds._conn)
    runs = store.list_runs(doc_id=doc_id, limit=1)
    return runs[0] if runs else {}


# ---------------------------------------------------------------------------
# Stage 1: Failure BEFORE ChromaDB write
# ---------------------------------------------------------------------------

class TestFailBeforeChroma:
    def test_no_data_written_no_run_committed(self, ds, mock_vector_store):
        doc_id = "chaos_pre_chroma"
        chunk_dicts = _make_chunk_dicts(doc_id)

        from rag_lab.ingest.transaction import IngestTransaction

        mock_vector_store.add.side_effect = RuntimeError("ChromaDB unavailable")

        with patch("rag_lab.ingest.transaction.VectorStore") as vs_cls:
            vs_cls.return_value = mock_vector_store

            try:
                with IngestTransaction(doc_id, "/fake/path.md", ds) as txn:
                    txn.update(chunks_expected=len(chunk_dicts))
                    mock_vector_store.add.side_effect = RuntimeError("ChromaDB unavailable")
                    raise RuntimeError("ChromaDB unavailable")
            except RuntimeError:
                pass

        assert _count_chunks(ds, doc_id) == 0
        assert _count_fts(ds, doc_id) == 0
        run = _latest_run(ds, doc_id)
        assert run["status"] == "ROLLED_BACK"


# ---------------------------------------------------------------------------
# Stage 2: Failure AFTER ChromaDB, BEFORE DocStore
# ---------------------------------------------------------------------------

class TestFailAfterChromaBeforeDocStore:
    def test_chroma_data_removed_by_rollback(self, ds, mock_vector_store):
        """ChromaDB was written but DocStore write fails — rollback must delete ChromaDB vectors."""
        doc_id = "chaos_after_chroma"
        chunk_dicts = _make_chunk_dicts(doc_id)

        chroma_written = []

        def fake_chroma_add(**kwargs):
            chroma_written.extend(kwargs.get("ids", []))

        mock_vector_store.add.side_effect = fake_chroma_add

        from rag_lab.ingest.transaction import IngestTransaction

        with patch("rag_lab.ingest.transaction.VectorStore") as vs_cls:
            vs_cls.return_value = mock_vector_store

            try:
                with IngestTransaction(doc_id, "/fake/path.md", ds) as txn:
                    txn.update(chunks_expected=len(chunk_dicts))
                    # ChromaDB "succeeds"
                    mock_vector_store.add(ids=[c["chunk_id"] for c in chunk_dicts],
                                         embeddings=[], documents=[], metadatas=[])
                    txn.update(chunks_written_chroma=len(chunk_dicts))
                    # DocStore write fails
                    raise RuntimeError("DocStore write failed")
            except RuntimeError:
                pass

        # Verify ChromaDB rollback was called
        mock_vector_store.delete_by_doc_id.assert_called_with(doc_id)

        # Verify DocStore is clean (nothing was written before the exception)
        assert _count_chunks(ds, doc_id) == 0

        run = _latest_run(ds, doc_id)
        assert run["status"] == "ROLLED_BACK"
        assert run["chunks_written_chroma"] == len(chunk_dicts)


# ---------------------------------------------------------------------------
# Stage 3: Failure AFTER DocStore, BEFORE metadata
# ---------------------------------------------------------------------------

class TestFailAfterDocStoreBeforeMetadata:
    def test_chunks_removed_by_rollback(self, ds, mock_vector_store):
        doc_id = "chaos_after_docstore"
        chunk_dicts = _make_chunk_dicts(doc_id)

        from rag_lab.ingest.transaction import IngestTransaction

        with patch("rag_lab.ingest.transaction.VectorStore") as vs_cls:
            vs_cls.return_value = mock_vector_store

            try:
                with IngestTransaction(doc_id, "/fake/path.md", ds) as txn:
                    txn.update(chunks_expected=len(chunk_dicts))
                    # ChromaDB succeeds (no-op mock)
                    mock_vector_store.add()
                    txn.update(chunks_written_chroma=len(chunk_dicts))
                    # DocStore succeeds
                    ds.add(chunk_dicts)
                    txn.update(
                        chunks_written_docstore=len(chunk_dicts),
                        chunks_written_fts5=len(chunk_dicts),
                    )
                    # Metadata fails
                    raise RuntimeError("Metadata write failed")
            except RuntimeError:
                pass

        # Both chunks AND FTS should be gone after rollback
        assert _count_chunks(ds, doc_id) == 0
        assert _count_fts(ds, doc_id) == 0
        mock_vector_store.delete_by_doc_id.assert_called_with(doc_id)

        run = _latest_run(ds, doc_id)
        assert run["status"] == "ROLLED_BACK"
        assert run["chunks_written_docstore"] == len(chunk_dicts)


# ---------------------------------------------------------------------------
# Stage 4: Success path — run should be COMMITTED
# ---------------------------------------------------------------------------

class TestSuccessPath:
    def test_committed_run_with_correct_counts(self, ds, mock_vector_store):
        doc_id = "chaos_success"
        chunk_dicts = _make_chunk_dicts(doc_id, n=4)

        from rag_lab.ingest.transaction import IngestTransaction
        from rag_lab.storage.metadata_store import MetadataStore

        with patch("rag_lab.ingest.transaction.VectorStore") as vs_cls:
            vs_cls.return_value = mock_vector_store

            with IngestTransaction(doc_id, "/fake/success.md", ds) as txn:
                txn.update(chunks_expected=len(chunk_dicts))
                mock_vector_store.add()
                txn.update(chunks_written_chroma=len(chunk_dicts))
                ds.add(chunk_dicts)
                txn.update(
                    chunks_written_docstore=len(chunk_dicts),
                    chunks_written_fts5=len(chunk_dicts),
                )
                MetadataStore(conn=ds._conn).upsert_document(doc_id)
                txn.update(metadata_written=1)

        assert _count_chunks(ds, doc_id) == 4
        assert _count_fts(ds, doc_id) == 4

        run = _latest_run(ds, doc_id)
        assert run["status"] == "COMMITTED"
        assert run["chunks_expected"] == 4
        assert run["chunks_written_docstore"] == 4
        assert run["chunks_written_chroma"] == 4
        assert run["metadata_written"] == 1

    def test_no_rollback_on_success(self, ds, mock_vector_store):
        doc_id = "chaos_no_rollback"

        from rag_lab.ingest.transaction import IngestTransaction

        with patch("rag_lab.ingest.transaction.VectorStore") as vs_cls:
            vs_cls.return_value = mock_vector_store

            with IngestTransaction(doc_id, None, ds) as txn:
                txn.update(chunks_expected=0)

        mock_vector_store.delete_by_doc_id.assert_not_called()


# ---------------------------------------------------------------------------
# Multiple failures: re-ingest after rollback
# ---------------------------------------------------------------------------

class TestReIngestAfterRollback:
    def test_clean_re_ingest_after_failed_run(self, ds, mock_vector_store):
        """After a failed+rolled-back run, a second run can ingest cleanly."""
        doc_id = "chaos_retry_doc"
        chunk_dicts = _make_chunk_dicts(doc_id, n=2)

        from rag_lab.ingest.transaction import IngestTransaction
        from rag_lab.storage.metadata_store import MetadataStore

        with patch("rag_lab.ingest.transaction.VectorStore") as vs_cls:
            vs_cls.return_value = mock_vector_store

            # First run — fails after writing to DocStore
            try:
                with IngestTransaction(doc_id, "/p.md", ds) as txn:
                    ds.add(chunk_dicts)
                    txn.update(chunks_written_docstore=len(chunk_dicts))
                    raise RuntimeError("Transient failure")
            except RuntimeError:
                pass

            assert _count_chunks(ds, doc_id) == 0

            # Second run — succeeds
            with IngestTransaction(doc_id, "/p.md", ds) as txn:
                txn.update(chunks_expected=len(chunk_dicts))
                ds.add(chunk_dicts)
                txn.update(chunks_written_docstore=len(chunk_dicts))
                MetadataStore(conn=ds._conn).upsert_document(doc_id)
                txn.update(metadata_written=1)

        assert _count_chunks(ds, doc_id) == len(chunk_dicts)

        from rag_lab.ingest.transaction import IngestRunStore
        store = IngestRunStore(ds._conn)
        runs = store.list_runs(doc_id=doc_id)
        statuses = {r["status"] for r in runs}
        assert "COMMITTED" in statuses
        assert "ROLLED_BACK" in statuses


# ---------------------------------------------------------------------------
# Reconcile integration: stale runs appear in reconcile report
# ---------------------------------------------------------------------------

class TestReconcileIngestHealth:
    def test_stale_run_appears_in_reconcile(self, ds):
        """A stale IN_PROGRESS run is detected by reconcile."""
        ds._conn.execute(
            "INSERT INTO ingest_runs (run_id, doc_id, source_path, started_at, status) "
            "VALUES ('stale_recon', 'doc_stale', NULL, datetime('now', '-2 hours'), 'IN_PROGRESS')"
        )
        ds._conn.commit()

        from rag_lab.maintenance.reconcile import reconcile, _has_issues

        with (
            patch("rag_lab.maintenance.reconcile.DocStore", return_value=ds),
            patch("rag_lab.maintenance.reconcile.VectorStore") as vs_cls,
        ):
            # ds.initialize() is a no-op inside reconcile since ds is already open,
            # but reconcile calls DocStore() — so we patch it to return our ds
            vs = MagicMock()
            vs._collection.get.return_value = {"ids": []}
            vs._collection.count.return_value = 0
            vs_cls.return_value = vs

            # Patch ds.initialize to avoid re-init side effects
            original_init = ds.initialize
            ds.initialize = lambda: None

            result = reconcile(quiet=True)
            ds.initialize = original_init

        assert len(result["stale_ingest_runs"]) >= 1
        run_ids = [r["run_id"] for r in result["stale_ingest_runs"]]
        assert "stale_recon" in run_ids
        assert _has_issues(result)

    def test_failed_run_appears_in_reconcile(self, ds):
        """A FAILED run is detected by reconcile."""
        ds._conn.execute(
            "INSERT INTO ingest_runs (run_id, doc_id, source_path, started_at, status, error_message) "
            "VALUES ('fail_recon', 'doc_fail_r', NULL, datetime('now', '-10 minutes'), 'FAILED', 'oops')"
        )
        ds._conn.commit()

        from rag_lab.maintenance.reconcile import reconcile, _has_issues

        with (
            patch("rag_lab.maintenance.reconcile.DocStore", return_value=ds),
            patch("rag_lab.maintenance.reconcile.VectorStore") as vs_cls,
        ):
            vs = MagicMock()
            vs._collection.get.return_value = {"ids": []}
            vs._collection.count.return_value = 0
            vs_cls.return_value = vs

            ds.initialize = lambda: None
            result = reconcile(quiet=True)
            del ds.initialize

        assert len(result["failed_ingest_runs"]) >= 1
        run_ids = [r["run_id"] for r in result["failed_ingest_runs"]]
        assert "fail_recon" in run_ids
        assert _has_issues(result)
