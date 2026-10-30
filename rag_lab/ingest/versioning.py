"""Document-version tracking, pre-image snapshot and safe rollback (P0).

Implements the RAG-01/RAG-02/RAG-04 guarantees:

* **RAG-01** a failed re-ingest restores the previously committed version
  (chunks, FTS5, Chroma vectors, ``documents`` row and tags).
* **RAG-02** a rollback of a run that is no longer the active version never
  destroys a newer committed version.
* **RAG-04** a successful replacement removes the chunks that disappeared.

Design (hardened after adversarial review)
------------------------------------------
One active version per ``doc_id`` lives in ``document_versions``; a single-slot
pre-image lives in the ``version_snapshot_*`` tables.  Every replacement attempt
is journaled in ``version_ops`` **before the first write** (state
``WRITING``).  The attempt then transitions to ``COMMITTING`` (on success) or
``RESTORING`` (on failure) and finally ``DONE``.  A crash leaves the attempt in
a non-``DONE`` state that :func:`recover_pending_ops` can complete idempotently.

Fail-closed rules:
* For a real vector store (``VectorStore.supports_versioning is True``) a failed
  snapshot/delete/restore raises :class:`VersioningError`; the operation is left
  pending for recovery and is never reported as done.
* Test doubles (which do not declare ``supports_versioning``) keep best-effort
  behaviour so unit tests without real vectors still exercise the SQL logic.
* A completed rollback **supersedes** any pending commit for the same document
  so recovery cannot replay it over the restored version.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from rag_lab.exceptions import VersioningError

logger = logging.getLogger("rag_lab")

_CHUNK_COLS = [
    "chunk_id", "doc_id", "text", "heading_path", "tipo", "posicion_relativa",
    "n_tokens", "line_start", "line_end", "sparse_tokens", "sparse_weights",
    "embedding_model_name", "embedding_model_version", "embedding_dim",
    "sparse_format_version",
]

_PENDING_STATES = ("WRITING", "COMMITTING", "RESTORING")


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


@dataclass
class VersionHandle:
    """Identifies the attempt/previous version so it can be committed/restored."""

    doc_id: str
    old_run_id: Optional[str]
    old_chunk_ids: List[str] = field(default_factory=list)
    had_previous: bool = False
    op_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _strict(vector_store) -> bool:
    """True only for a real store that declares versioning support."""
    return getattr(vector_store, "supports_versioning", False) is True


def _get_active(conn, doc_id: str) -> Optional[dict]:
    if not _table_exists(conn, "document_versions"):
        return None
    row = conn.execute(
        "SELECT active_run_id, generation, chunk_ids FROM document_versions WHERE doc_id=?",
        (doc_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        ids = json.loads(row[2]) if row[2] else []
    except (TypeError, ValueError):
        ids = []
    return {"active_run_id": row[0], "generation": row[1] or 0, "chunk_ids": ids}


def _current_chunk_ids(conn, doc_id: str) -> List[str]:
    active = _get_active(conn, doc_id)
    if active is not None:
        return list(active["chunk_ids"])
    return [r[0] for r in conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)
    )]


def _latest_committed_run(conn, doc_id: str) -> Optional[str]:
    """Latest COMMITTED run, deterministic by insertion order (rowid)."""
    row = conn.execute(
        "SELECT run_id FROM ingest_runs WHERE doc_id=? AND status='COMMITTED' "
        "ORDER BY rowid DESC LIMIT 1",
        (doc_id,),
    ).fetchone()
    return row[0] if row else None


def _latest_run_any_status(conn, doc_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT run_id FROM ingest_runs WHERE doc_id=? ORDER BY rowid DESC LIMIT 1",
        (doc_id,),
    ).fetchone()
    return row[0] if row else None


def _pending_ops(conn, doc_id: str):
    if not _table_exists(conn, "version_ops"):
        return []
    return conn.execute(
        "SELECT id, state, payload_json FROM version_ops WHERE doc_id=? AND state IN "
        f"({','.join('?' * len(_PENDING_STATES))})",
        (doc_id, *_PENDING_STATES),
    ).fetchall()


def _mark_run(conn, run_id: Optional[str], status: str) -> None:
    """Reconcile ingest_runs / ingest_documents control rows during recovery."""
    if not run_id:
        return
    if _table_exists(conn, "ingest_runs"):
        conn.execute(
            "UPDATE ingest_runs SET status=?, finished_at=? WHERE run_id=?",
            (status, _now(), run_id),
        )
    if _table_exists(conn, "ingest_documents"):
        conn.execute(
            "UPDATE ingest_documents SET status=?, finished_at=? WHERE run_id=?",
            (status, _now(), run_id),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _vector_snapshot(doc_id, vector_store):
    """Return {chunk_id: (embedding_bytes, document, metadata_json)}.

    For a strict (real) store, an inability to read the vectors is fatal: we
    must not overwrite a version we cannot restore.
    """
    getter = getattr(vector_store, "get_by_doc_id", None)
    if getter is None:
        if _strict(vector_store):
            raise VersioningError(
                f"vector store lacks get_by_doc_id; cannot snapshot {doc_id!r}"
            )
        return {}
    try:
        data = getter(doc_id)
    except Exception as exc:
        if _strict(vector_store):
            raise VersioningError(
                f"failed to read existing vectors for {doc_id!r}: {exc}"
            ) from exc
        logger.warning("versioning: vector snapshot read failed for %r: %s", doc_id, exc)
        return {}
    if not isinstance(data, dict):
        if _strict(vector_store):
            raise VersioningError(
                f"vector store returned non-dict snapshot for {doc_id!r}: {type(data)!r}"
            )
        return {}
    ids = data.get("ids") or []
    embeddings = data.get("embeddings")
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    out = {}
    for i, cid in enumerate(ids):
        emb = embeddings[i] if embeddings is not None else None
        if emb is None:
            continue
        arr = np.asarray(emb, dtype=np.float32)
        doc = documents[i] if i < len(documents) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        out[cid] = (arr.tobytes(), doc, json.dumps(meta, ensure_ascii=False))
    return out


def _vector_delete(ids, vector_store) -> None:
    if not ids:
        return
    fn = getattr(vector_store, "delete_ids", None)
    if fn is None:
        if _strict(vector_store):
            raise VersioningError("vector store lacks delete_ids")
        return
    try:
        fn(list(ids))
    except Exception as exc:
        if _strict(vector_store):
            raise VersioningError(f"failed to delete vectors {ids[:3]}...: {exc}") from exc
        logger.warning("versioning: vector delete failed: %s", exc)


def _vector_delete_doc(doc_id, vector_store) -> None:
    fn = getattr(vector_store, "delete_by_doc_id", None)
    if fn is None:
        if _strict(vector_store):
            raise VersioningError("vector store lacks delete_by_doc_id")
        return
    try:
        fn(doc_id)
    except Exception as exc:
        if _strict(vector_store):
            raise VersioningError(f"failed to delete vectors for {doc_id!r}: {exc}") from exc
        logger.warning("versioning: vector delete_by_doc_id failed: %s", exc)


def _vector_restore(rows, vector_store) -> None:
    if not rows:
        return
    fn = getattr(vector_store, "restore_vectors", None)
    if fn is None:
        if _strict(vector_store):
            raise VersioningError("vector store lacks restore_vectors")
        return
    ids, embs, docs, metas = [], [], [], []
    for cid, (emb_bytes, doc, meta_json) in rows:
        ids.append(cid)
        embs.append(np.frombuffer(emb_bytes, dtype=np.float32).tolist())
        docs.append(doc)
        try:
            metas.append(json.loads(meta_json))
        except (TypeError, ValueError):
            metas.append({})
    try:
        fn(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    except Exception as exc:
        if _strict(vector_store):
            raise VersioningError(f"failed to restore {len(ids)} vectors: {exc}") from exc
        logger.warning("versioning: vector restore failed: %s", exc)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def _clear_snapshot(conn, doc_id: str) -> None:
    conn.execute("DELETE FROM version_snapshot_chunks WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM version_snapshot_vectors WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM version_snapshot_docs WHERE doc_id=?", (doc_id,))


def _snapshot_document(conn, doc_id: str, old_run_id: Optional[str], vector_store) -> None:
    # Read vectors FIRST so a read failure aborts before we discard anything.
    vectors = _vector_snapshot(doc_id, vector_store)

    _clear_snapshot(conn, doc_id)
    rows = conn.execute("SELECT * FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
    cols = ", ".join(_CHUNK_COLS)
    placeholders = ", ".join("?" * len(_CHUNK_COLS))
    for row in rows:
        conn.execute(
            f"INSERT OR REPLACE INTO version_snapshot_chunks ({cols}) VALUES ({placeholders})",
            row,
        )
    for cid, (emb_bytes, doc, meta_json) in vectors.items():
        conn.execute(
            "INSERT OR REPLACE INTO version_snapshot_vectors "
            "(doc_id, chunk_id, embedding, document, metadata_json) VALUES (?,?,?,?,?)",
            (doc_id, cid, emb_bytes, doc, meta_json),
        )

    doc_row = None
    if _table_exists(conn, "documents"):
        cur = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
        dcols = [d[0] for d in cur.description]
        fetched = cur.fetchone()
        if fetched is not None:
            doc_row = json.dumps(dict(zip(dcols, fetched)), ensure_ascii=False)
    tags = []
    if _table_exists(conn, "document_tags"):
        tags = [r[0] for r in conn.execute(
            "SELECT t.name FROM document_tags dt JOIN tags t ON dt.tag_id=t.tag_id "
            "WHERE dt.doc_id=?",
            (doc_id,),
        )]
    conn.execute(
        "INSERT OR REPLACE INTO version_snapshot_docs (doc_id, run_id, doc_json, tags_json, created_at) "
        "VALUES (?,?,?,?,datetime('now'))",
        (doc_id, old_run_id, doc_row, json.dumps(tags, ensure_ascii=False)),
    )
    conn.commit()


def _restore_document_meta(conn, doc_id: str) -> None:
    snap = conn.execute(
        "SELECT doc_json, tags_json FROM version_snapshot_docs WHERE doc_id=?", (doc_id,)
    ).fetchone()
    if not _table_exists(conn, "documents"):
        return
    conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
    if snap and snap[0]:
        doc = json.loads(snap[0])
        cols = list(doc.keys())
        placeholders = ", ".join("?" * len(cols))
        conn.execute(
            f"INSERT OR REPLACE INTO documents ({', '.join(cols)}) VALUES ({placeholders})",
            [doc[c] for c in cols],
        )
        tags = json.loads(snap[1]) if snap[1] else []
        for tag in tags:
            conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
            tag_id = conn.execute("SELECT tag_id FROM tags WHERE name=?", (tag,)).fetchone()[0]
            conn.execute(
                "INSERT OR IGNORE INTO document_tags (doc_id, tag_id) VALUES (?,?)",
                (doc_id, tag_id),
            )
    conn.commit()


def _restore_chunks(conn, doc_id: str) -> None:
    conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM chunks_fts WHERE doc_id=?", (doc_id,))
    rows = conn.execute("SELECT * FROM version_snapshot_chunks WHERE doc_id=?", (doc_id,)).fetchall()
    cols = ", ".join(_CHUNK_COLS)
    placeholders = ", ".join("?" * len(_CHUNK_COLS))
    for row in rows:
        conn.execute(f"INSERT OR REPLACE INTO chunks ({cols}) VALUES ({placeholders})", row)
        conn.execute(
            "INSERT INTO chunks_fts (chunk_id, doc_id, text) VALUES (?,?,?)",
            (row[0], row[1], row[2]),
        )
    conn.commit()


def _restore_vectors(conn, doc_id: str, vector_store) -> None:
    """Delete the current vectors for the doc, then restore the snapshot."""
    _vector_delete_doc(doc_id, vector_store)
    rows = conn.execute(
        "SELECT chunk_id, embedding, document, metadata_json "
        "FROM version_snapshot_vectors WHERE doc_id=?",
        (doc_id,),
    ).fetchall()
    _vector_restore([(r[0], (r[1], r[2], r[3])) for r in rows], vector_store)


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def _supersede_other_ops(conn, doc_id: str, keep_op_id: Optional[int]) -> None:
    if keep_op_id is None:
        conn.execute(
            f"UPDATE version_ops SET state='SUPERSEDED' WHERE doc_id=? AND state IN "
            f"({','.join('?' * len(_PENDING_STATES))})",
            (doc_id, *_PENDING_STATES),
        )
    else:
        conn.execute(
            f"UPDATE version_ops SET state='SUPERSEDED' WHERE doc_id=? AND id!=? AND state IN "
            f"({','.join('?' * len(_PENDING_STATES))})",
            (doc_id, keep_op_id, *_PENDING_STATES),
        )


def _set_op(conn, op_id: int, state: str, payload: dict) -> None:
    conn.execute(
        "UPDATE version_ops SET state=?, payload_json=?, run_id=? WHERE id=?",
        (state, json.dumps(payload), payload.get("new_run") or payload.get("old_run"), op_id),
    )
    conn.commit()


def _new_op(conn, doc_id: str, state: str, payload: dict) -> int:
    return conn.execute(
        "INSERT INTO version_ops (doc_id, op, run_id, state, created_at, payload_json) "
        "VALUES (?, ?, ?, ?, datetime('now'), ?)",
        (doc_id, payload.get("phase", "REPLACE"), payload.get("old_run") or payload.get("new_run"),
         state, json.dumps(payload)),
    ).lastrowid


# ---------------------------------------------------------------------------
# Apply helpers (shared by normal and recovery paths)
# ---------------------------------------------------------------------------

def _apply_commit(conn, vector_store, doc_id: str, new_run_id: Optional[str],
                  new_ids: List[str], obsolete: List[str],
                  generation: Optional[int] = None) -> None:
    new_ids = list(dict.fromkeys(new_ids or []))
    if obsolete:
        ph = ", ".join("?" * len(obsolete))
        conn.execute(
            f"DELETE FROM chunks WHERE doc_id=? AND chunk_id IN ({ph})", [doc_id, *obsolete]
        )
        conn.execute(
            f"DELETE FROM chunks_fts WHERE doc_id=? AND chunk_id IN ({ph})", [doc_id, *obsolete]
        )
        conn.commit()
        _vector_delete(obsolete, vector_store)

    if new_run_id:
        if generation is None:
            generation = (_get_active(conn, doc_id) or {}).get("generation", 0) + 1
        conn.execute(
            "INSERT OR REPLACE INTO document_versions "
            "(doc_id, active_run_id, generation, chunk_ids, updated_at) "
            "VALUES (?,?,?,?,datetime('now'))",
            (doc_id, new_run_id, generation, json.dumps(new_ids)),
        )
    else:
        conn.execute("DELETE FROM document_versions WHERE doc_id=?", (doc_id,))
    conn.commit()

    if new_run_id and _table_exists(conn, "ingest_runs"):
        conn.execute(
            "UPDATE ingest_runs SET status='COMMITTED', finished_at=? "
            "WHERE run_id=? AND status!='COMMITTED'",
            (_now(), new_run_id),
        )
        conn.commit()


def _apply_restore(conn, vector_store, doc_id: str,
                   old_run_id: Optional[str], old_ids: List[str]) -> None:
    old_ids = list(old_ids or [])
    _restore_chunks(conn, doc_id)
    generation = (_get_active(conn, doc_id) or {}).get("generation", 0)
    if old_run_id or old_ids:
        conn.execute(
            "INSERT OR REPLACE INTO document_versions "
            "(doc_id, active_run_id, generation, chunk_ids, updated_at) "
            "VALUES (?,?,?,?,datetime('now'))",
            (doc_id, old_run_id, generation, json.dumps(old_ids)),
        )
    else:
        conn.execute("DELETE FROM document_versions WHERE doc_id=?", (doc_id,))
    conn.commit()
    _restore_document_meta(conn, doc_id)
    _restore_vectors(conn, doc_id, vector_store)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def begin_replace(doc_id: str, conn, vector_store=None,
                  run_id: Optional[str] = None) -> VersionHandle:
    """Snapshot the active version and journal the attempt BEFORE any write.

    Raises :class:`VersioningError` if a previous attempt for the document
    cannot be recovered, so the caller aborts instead of overwriting the only
    recoverable pre-image.
    """
    conn.commit()  # close any implicit transaction

    # Finish any pending attempt for this document first (fail-closed).
    _recover_doc(doc_id, conn, vector_store)

    old_ids = _current_chunk_ids(conn, doc_id)
    active = _get_active(conn, doc_id)
    old_run = active["active_run_id"] if active else _latest_committed_run(conn, doc_id)
    had_previous = bool(old_ids) or bool(
        _table_exists(conn, "documents")
        and conn.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    )
    _snapshot_document(conn, doc_id, old_run, vector_store)

    payload = {
        "phase": "begin", "old_run": old_run, "old_ids": old_ids,
        "attempt_run": run_id,
    }
    op_id = _new_op(conn, doc_id, "WRITING", payload)
    conn.commit()
    logger.info(
        "versioning: begin_replace doc_id=%r previous_run=%s previous_chunks=%d op=%s",
        doc_id, old_run, len(old_ids), op_id,
    )
    return VersionHandle(doc_id=doc_id, old_run_id=old_run,
                         old_chunk_ids=list(old_ids), had_previous=had_previous,
                         op_id=op_id)


def commit_replace(handle: VersionHandle, conn, vector_store, new_run_id: str,
                   new_chunk_ids: List[str]) -> None:
    """Finalize a replacement: drop obsolete chunks, publish the new version."""
    doc_id = handle.doc_id
    new_ids = list(dict.fromkeys(new_chunk_ids or []))
    new_set = set(new_ids)
    obsolete = [c for c in handle.old_chunk_ids if c not in new_set]
    generation = (_get_active(conn, doc_id) or {}).get("generation", 0) + 1
    payload = {
        "phase": "commit", "old_run": handle.old_run_id, "old_ids": handle.old_chunk_ids,
        "new_run": new_run_id, "new_ids": new_ids, "obsolete": obsolete,
        "generation": generation,
    }
    op_id = handle.op_id
    if op_id is None:
        op_id = _new_op(conn, doc_id, "COMMITTING", payload)
    else:
        _set_op(conn, op_id, "COMMITTING", payload)
    _supersede_other_ops(conn, doc_id, op_id)
    conn.commit()

    try:
        _apply_commit(conn, vector_store, doc_id, new_run_id, new_ids, obsolete,
                      generation=generation)
    except Exception:
        logger.exception("versioning: commit_replace failed for %r; left for recovery", doc_id)
        raise
    _set_op(conn, op_id, "DONE", payload)
    logger.info(
        "versioning: commit_replace doc_id=%r run=%s new_chunks=%d obsolete=%d",
        doc_id, new_run_id, len(new_ids), len(obsolete),
    )


def rollback_replace(handle: VersionHandle, conn, vector_store) -> None:
    """Restore the pre-image captured by :func:`begin_replace` (idempotent)."""
    doc_id = handle.doc_id
    payload = {"phase": "rollback", "old_run": handle.old_run_id, "old_ids": handle.old_chunk_ids}
    op_id = handle.op_id
    if op_id is None:
        op_id = _new_op(conn, doc_id, "RESTORING", payload)
    else:
        _set_op(conn, op_id, "RESTORING", payload)
    # A completed rollback cancels any pending commit for the same document.
    _supersede_other_ops(conn, doc_id, op_id)
    conn.commit()

    try:
        _apply_restore(conn, vector_store, doc_id, handle.old_run_id, handle.old_chunk_ids)
    except Exception:
        logger.exception("versioning: rollback_replace failed for %r; left for recovery", doc_id)
        raise
    _set_op(conn, op_id, "DONE", payload)
    logger.warning(
        "versioning: rollback_replace doc_id=%r restored_run=%s chunks=%d",
        doc_id, handle.old_run_id, len(handle.old_chunk_ids),
    )


def admin_rollback(run_id: str, conn, vector_store) -> dict:
    """Roll back an ingest run, refusing to destroy a newer active version."""
    from rag_lab.ingest.transaction import IngestRunStore

    try:
        recover_pending_ops(conn, vector_store)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"cannot recover pending version operations: {exc}",
        }

    run = IngestRunStore(conn).get(run_id)
    if not run:
        return {"status": "error", "reason": f"run not found: {run_id}"}
    doc_id = run["doc_id"]

    active = _get_active(conn, doc_id)
    active_run = active["active_run_id"] if active else None
    if active_run is None:
        # Legacy (pre-versioning) document: the latest COMMITTED run owns the
        # data. If none exists, only the latest run of any status may be
        # considered owner of the (uncommitted) data.
        active_run = _latest_committed_run(conn, doc_id) or _latest_run_any_status(conn, doc_id)

    if active_run != run_id:
        return {
            "status": "refused",
            "reason": (
                f"run {run_id} is not the active version of {doc_id!r} "
                f"(active={active_run!r}); refusing to destroy newer data"
            ),
            "active_run": active_run,
        }

    snap_ids = [r[0] for r in conn.execute(
        "SELECT chunk_id FROM version_snapshot_chunks WHERE doc_id=?", (doc_id,)
    )]
    snap_doc = conn.execute(
        "SELECT run_id FROM version_snapshot_docs WHERE doc_id=?", (doc_id,)
    ).fetchone()
    handle = VersionHandle(
        doc_id=doc_id,
        old_run_id=snap_doc[0] if snap_doc else None,
        old_chunk_ids=snap_ids,
        had_previous=bool(snap_ids or snap_doc),
        op_id=None,
    )
    rollback_replace(handle, conn, vector_store)
    IngestRunStore(conn).update(run_id, status="ROLLED_BACK", finished_at=_now())
    return {
        "status": "rolled_back",
        "doc_id": doc_id,
        "restored_previous": bool(snap_ids or snap_doc),
    }


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def _finish_op(conn, op_id: int, vector_store, doc_id: str, state: str, payload: dict) -> None:
    if state == "WRITING":
        # The attempt never committed: restore the previous version.
        _apply_restore(conn, vector_store, doc_id,
                       payload.get("old_run"), payload.get("old_ids", []))
        _mark_run(conn, payload.get("attempt_run"), "ROLLED_BACK")
    elif state == "COMMITTING":
        _apply_commit(conn, vector_store, doc_id,
                      payload.get("new_run"), payload.get("new_ids", []),
                      payload.get("obsolete", []), generation=payload.get("generation"))
        _mark_run(conn, payload.get("new_run"), "COMMITTED")
    elif state == "RESTORING":
        _apply_restore(conn, vector_store, doc_id,
                       payload.get("old_run"), payload.get("old_ids", []))
        _mark_run(conn, payload.get("attempt_run"), "ROLLED_BACK")
    conn.execute("UPDATE version_ops SET state='DONE' WHERE id=?", (op_id,))
    conn.commit()


def recover_pending_ops(conn, vector_store=None) -> int:
    """Complete interrupted version operations idempotently.

    Raises :class:`VersioningError` if any pending operation could not be
    recovered, so callers fail closed instead of proceeding over an
    inconsistent document.
    """
    if not _table_exists(conn, "version_ops"):
        return 0
    rows = conn.execute(
        "SELECT id, doc_id, state, payload_json FROM version_ops "
        f"WHERE state IN ({','.join('?' * len(_PENDING_STATES))})",
        _PENDING_STATES,
    ).fetchall()
    n = 0
    failures = []
    for op_id, doc_id, state, payload_json in rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except (TypeError, ValueError):
            payload = {}
        try:
            _finish_op(conn, op_id, vector_store, doc_id, state, payload)
            n += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("versioning: recovery of op %s (%s) failed: %s", op_id, state, exc)
            failures.append((op_id, state, str(exc)))
    if n:
        logger.warning("versioning: recovered %d interrupted version operation(s)", n)
    if failures:
        raise VersioningError(
            f"{len(failures)} version operation(s) could not be recovered: {failures[:3]}"
        )
    return n


def _recover_doc(doc_id: str, conn, vector_store) -> None:
    """Recover pending operations for a single document (fail-closed)."""
    rows = _pending_ops(conn, doc_id)
    for op_id, state, payload_json in rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except (TypeError, ValueError):
            payload = {}
        try:
            _finish_op(conn, op_id, vector_store, doc_id, state, payload)
        except Exception as exc:
            raise VersioningError(
                f"cannot recover pending op {op_id} ({state}) for {doc_id!r}: {exc}"
            ) from exc
    remaining = _pending_ops(conn, doc_id)
    if remaining:
        raise VersioningError(
            f"pending version operations remain for {doc_id!r}: "
            f"{[r[0] for r in remaining]}"
        )
