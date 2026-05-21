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
