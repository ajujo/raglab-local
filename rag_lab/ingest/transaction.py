"""Ingest transaction tracking and rollback compensation.

IngestTransaction wraps a single document ingest in a context manager that:
  - Records each stage's progress in `ingest_runs` (docstore.sqlite)
  - Automatically rolls back across all stores on failure
  - Provides idempotent compensation: ChromaDB delete + SQLite DELETE WHERE

Rollback is a best-effort compensation, not a true ACID rollback.  Each
store write still commits on its own; on failure the compensation undoes
those writes.  The window between stage commits is small and tolerable;
any residual inconsistency is detectable by reconcile.

Usage::

    with IngestTransaction(doc_id, source_path, doc_store) as txn:
        txn.update(chunks_expected=len(chunks))
        vector_store.add(...)
        txn.update(chunks_written_chroma=len(chunks))
        doc_store.add(chunk_dicts)
        txn.update(chunks_written_docstore=len(chunks), ...)
        metadata_store.upsert_document(doc_id, ...)
        txn.update(metadata_written=1)
    # __exit__ marks COMMITTED on success, FAILED + rollback on exception
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")

_INGEST_RUNS_COLS = [
    "run_id",
    "doc_id",
    "source_path",
    "started_at",
    "finished_at",
    "status",
    "error_message",
    "chunks_expected",
    "chunks_written_docstore",
    "chunks_written_fts5",
    "chunks_written_chroma",
    "chunks_written_sparse",
    "metadata_written",
]

_INGEST_BATCHES_COLS = [
    "batch_id", "started_at", "finished_at", "status", "source_path",
    "total_docs", "committed_docs", "skipped_docs", "failed_docs",
    "rolled_back_docs", "total_chunks",
]

_INGEST_DOCS_COLS = [
    "id", "batch_id", "run_id", "doc_id", "path", "content_hash",
    "status", "error_message", "started_at", "finished_at",
    "chunks_count", "retry_count", "validation_summary",
]

# Valid status values for ingest_documents
INGEST_DOC_STATUSES = frozenset([
    "PENDING", "VALIDATING", "VALIDATED", "CHUNKING", "EMBEDDING",
    "WRITING", "COMMITTED", "FAILED", "ROLLED_BACK", "SKIPPED",
])


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _row_to_dict(row) -> dict:
    return dict(zip(_INGEST_RUNS_COLS, row))


class IngestRunStore:
    """Low-level CRUD on the `ingest_runs` table."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def create(self, run_id: str, doc_id: str, source_path: Optional[str]) -> None:
        self._conn.execute(
            """
            INSERT INTO ingest_runs
                (run_id, doc_id, source_path, started_at, status)
            VALUES (?, ?, ?, ?, 'IN_PROGRESS')
            """,
            (run_id, doc_id, source_path, _now()),
        )
        self._conn.commit()

    def update(self, run_id: str, **fields) -> None:
        if not fields:
            return
        set_parts = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [run_id]
        self._conn.execute(
            f"UPDATE ingest_runs SET {set_parts} WHERE run_id = ?",
            values,
        )
        self._conn.commit()

    def get(self, run_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM ingest_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def list_runs(
        self,
        doc_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        where, params = [], []
        if doc_id:
            where.append("doc_id = ?")
            params.append(doc_id)
        if status:
            where.append("status = ?")
            params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM ingest_runs {clause} ORDER BY started_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_failed(self) -> List[dict]:
        return self.list_runs(status="FAILED")

    def get_stale_in_progress(self, minutes: int = 30) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM ingest_runs "
            "WHERE status = 'IN_PROGRESS' "
            "AND started_at < datetime('now', ?)",
            (f"-{minutes} minutes",),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


class IngestBatchStore:
    """CRUD for the ingest_batches table (one row per batch/CLI invocation)."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def create(self, source_path=None) -> str:
        batch_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO ingest_batches (batch_id, started_at, source_path) "
            "VALUES (?, ?, ?)",
            (batch_id, _now(), str(source_path) if source_path else None),
        )
        self._conn.commit()
        return batch_id

    def update(self, batch_id: str, **fields) -> None:
        if not fields:
            return
        set_parts = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [batch_id]
        self._conn.execute(
            f"UPDATE ingest_batches SET {set_parts} WHERE batch_id = ?", values
        )
        self._conn.commit()

    def finalize(self, batch_id: str) -> None:
        """Compute final doc counts from ingest_documents and set terminal status."""
        row = self._conn.execute(
            """
            SELECT
              SUM(CASE WHEN status='COMMITTED' THEN 1 ELSE 0 END),
              SUM(CASE WHEN status='SKIPPED' THEN 1 ELSE 0 END),
              SUM(CASE WHEN status IN ('FAILED') THEN 1 ELSE 0 END),
              SUM(CASE WHEN status='ROLLED_BACK' THEN 1 ELSE 0 END),
              SUM(chunks_count)
            FROM ingest_documents WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()
        committed = row[0] or 0
        skipped = row[1] or 0
        failed = row[2] or 0
        rolled_back = row[3] or 0
        total_chunks = row[4] or 0

        if failed + rolled_back == 0:
            status = "COMPLETED"
        elif committed + skipped > 0:
            status = "PARTIAL"
        else:
            status = "FAILED"

        self._conn.execute(
            """
            UPDATE ingest_batches
            SET status=?, finished_at=?, committed_docs=?, skipped_docs=?,
                failed_docs=?, rolled_back_docs=?, total_chunks=?
            WHERE batch_id=?
            """,
            (status, _now(), committed, skipped, failed, rolled_back, total_chunks, batch_id),
        )
        self._conn.commit()

    def get(self, batch_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM ingest_batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(zip(_INGEST_BATCHES_COLS, row))

    def get_latest_incomplete(self) -> Optional[str]:
        """Return batch_id of most recent IN_PROGRESS batch, or None."""
        row = self._conn.execute(
            "SELECT batch_id FROM ingest_batches WHERE status='IN_PROGRESS' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def list_batches(self, limit: int = 20) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM ingest_batches ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(zip(_INGEST_BATCHES_COLS, r)) for r in rows]


class IngestDocumentStore:
    """CRUD for the ingest_documents table (one row per doc per batch)."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def create(self, batch_id: str, doc_id: str, path: str,
               content_hash: Optional[str] = None) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO ingest_documents
                (batch_id, doc_id, path, content_hash, status, started_at)
            VALUES (?, ?, ?, ?, 'PENDING', ?)
            """,
            (batch_id, doc_id, path, content_hash, _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def update(self, id: int, **fields) -> None:
        if not fields:
            return
        set_parts = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [id]
        self._conn.execute(
            f"UPDATE ingest_documents SET {set_parts} WHERE id = ?", values
        )
        self._conn.commit()

    def set_status(self, id: int, status: str, **kwargs) -> None:
        """Convenience: update status plus any extra fields in one call."""
        self.update(id, status=status, **kwargs)

    def find_committed(self, doc_id: str, content_hash: str) -> bool:
        """Return True if a COMMITTED row exists for this (doc_id, content_hash)."""
        if not content_hash:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM ingest_documents "
            "WHERE doc_id=? AND content_hash=? AND status='COMMITTED' LIMIT 1",
            (doc_id, content_hash),
        ).fetchone()
        return row is not None

    def list_by_batch(self, batch_id: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM ingest_documents WHERE batch_id=? ORDER BY id",
            (batch_id,),
        ).fetchall()
        return [dict(zip(_INGEST_DOCS_COLS, r)) for r in rows]

    def list_resumable(self, batch_id: str) -> List[dict]:
        """Return PENDING, FAILED, ROLLED_BACK docs in a batch."""
        rows = self._conn.execute(
            "SELECT * FROM ingest_documents "
            "WHERE batch_id=? AND status IN ('PENDING','FAILED','ROLLED_BACK') ORDER BY id",
            (batch_id,),
        ).fetchall()
        return [dict(zip(_INGEST_DOCS_COLS, r)) for r in rows]

    def list_failed(self, batch_id: Optional[str] = None) -> List[dict]:
        """Return FAILED/ROLLED_BACK docs, optionally filtered by batch."""
        if batch_id:
            rows = self._conn.execute(
                "SELECT * FROM ingest_documents "
                "WHERE batch_id=? AND status IN ('FAILED','ROLLED_BACK') ORDER BY id",
                (batch_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM ingest_documents "
                "WHERE status IN ('FAILED','ROLLED_BACK') ORDER BY id",
            ).fetchall()
        return [dict(zip(_INGEST_DOCS_COLS, r)) for r in rows]

    def get_by_id(self, id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM ingest_documents WHERE id=?", (id,)
        ).fetchone()
        return dict(zip(_INGEST_DOCS_COLS, row)) if row else None


class IngestTransaction:
    """Context manager: tracks ingest progress and rolls back on failure.

    Args:
        doc_id:      Document identifier (same as stored in chunks.doc_id).
        source_path: Original source file path (stored for retry purposes).
        doc_store:   Initialized DocStore instance whose connection holds
                     the ingest_runs table.
    """

    def __init__(self, doc_id: str, source_path, doc_store) -> None:
        self.doc_id = doc_id
        self.source_path = str(source_path) if source_path else None
        self._ds = doc_store
        self._run_store = IngestRunStore(doc_store._conn)
        self.run_id: Optional[str] = None

    def __enter__(self) -> "IngestTransaction":
        self.run_id = uuid.uuid4().hex[:12]
        self._run_store.create(self.run_id, self.doc_id, self.source_path)
        logger.info(
            f"IngestTransaction started: run_id={self.run_id} doc_id={self.doc_id!r}"
        )
        return self

    def update(self, **fields) -> None:
        """Update progress counters or status fields on the active run record."""
        if self.run_id:
            self._run_store.update(self.run_id, **fields)

    def rollback(self) -> None:
        """Compensation rollback: delete all data written for this doc_id.

        Safe to call multiple times (idempotent DELETE WHERE).
        Updates run status to ROLLED_BACK.
        """
        logger.warning(
            f"Rolling back ingest: run_id={self.run_id} doc_id={self.doc_id!r}"
        )

        # 1. ChromaDB compensation
        try:
            vs = VectorStore()
            vs.initialize()
            n = vs.delete_by_doc_id(self.doc_id)
            logger.info(
                f"Rollback: removed {n} ChromaDB vectors for doc_id={self.doc_id!r}"
            )
        except Exception as e:
            logger.error(f"Rollback: ChromaDB deletion error: {e}")

        # 2. SQLite chunks + FTS5 + metadata
        conn = self._ds._conn
        try:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (self.doc_id,))
            conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (self.doc_id,))
        except Exception as e:
            logger.error(f"Rollback: chunks deletion error: {e}")
        try:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (self.doc_id,))
        except Exception:
            pass  # documents table may not exist

        # 3. Mark run as ROLLED_BACK
        if self.run_id:
            self._run_store.update(
                self.run_id,
                status="ROLLED_BACK",
                finished_at=_now(),
            )

        conn.commit()
        logger.info(f"Rollback complete: run_id={self.run_id}")

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self._run_store.update(
                self.run_id,
                status="COMMITTED",
                finished_at=_now(),
            )
            logger.info(f"IngestTransaction committed: run_id={self.run_id}")
        else:
            self._run_store.update(
                self.run_id,
                status="FAILED",
                finished_at=_now(),
                error_message=str(exc_val)[:500] if exc_val else "unknown error",
            )
            logger.error(
                f"IngestTransaction failed: run_id={self.run_id} "
                f"doc={self.doc_id!r} error={exc_val}"
            )
            self.rollback()
        return False  # always re-raise
