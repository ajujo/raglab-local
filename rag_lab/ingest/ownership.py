"""Deterministic document-ownership authorization (P0-C / SOL-07).

This module replaces the ad-hoc ownership guard that compared loose hashes
across mutable tables.  It derives the confirmed owner from the *existing*
versioning state (``document_versions`` -> ``ingest_runs`` -> persisted
``chunks``/FTS/Chroma) and returns a structured :class:`Decision`.

Design split (important):

* :func:`decide` is a **pure** function over explicit snapshots
  (:class:`PersistedState`, :class:`IncomingState`).  It performs no I/O.
* :class:`OwnershipStore` is the read-only **adapter** that builds the snapshot
  from SQLite/FTS/Chroma.  It never mutates stores.
* The caller (``cli_ingest``) executes the decision; writes happen only through
  the existing ``IngestTransaction``/versioning path.

Policy (approved for P0-C):

* Only the confirmed owner may create a new generation.
* The owner with an identical persisted result is a NO-OP (no generation).
* Another path with an exact persisted equivalence is a NO-OP **without any
  documental write** (no content, vectors, metadata, manifest, owner, run or
  generation change).  Equality of content never grants ownership.
* Another path with different content/result is denied.
* Contradictory confirmed evidence or SQL/FTS/Chroma disagreement fails closed.
* A single loose hash never authorizes.
* The advisory manifest is never an authority.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Iterable, List, Optional, Sequence, Tuple


class DecisionKind(str, Enum):
    CREATE = "CREATE"
    REPLACE = "REPLACE"
    NOOP_OWNER = "NOOP_OWNER"
    NOOP_EQUIVALENT_OTHER_SOURCE = "NOOP_EQUIVALENT_OTHER_SOURCE"
    DENY_INCOHERENT = "DENY_INCOHERENT"
    DENY_NOT_OWNER = "DENY_NOT_OWNER"
    DENY_INSUFFICIENT_EVIDENCE = "DENY_INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reason: str
    owner_path: Optional[str] = None
    generation: Optional[int] = None
    active_run_id: Optional[str] = None

    def is_denied(self) -> bool:
        return self.kind in (
            DecisionKind.DENY_INCOHERENT,
            DecisionKind.DENY_NOT_OWNER,
            DecisionKind.DENY_INSUFFICIENT_EVIDENCE,
        )

    def is_noop(self) -> bool:
        return self.kind in (
            DecisionKind.NOOP_OWNER,
            DecisionKind.NOOP_EQUIVALENT_OTHER_SOURCE,
        )


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkRef:
    chunk_id: str
    text_sha256: str


Embedding = Tuple[str, str, int, int]  # name, version, dim, sparse_format_version
Classification = Tuple[Optional[str], Optional[str], Optional[str],
                       Optional[str], Optional[str], Tuple[str, ...]]


@dataclass(frozen=True)
class PersistedState:
    doc_id: str
    has_version: bool
    active_run_id: Optional[str]
    generation: int
    active_content_hash: Optional[str]
    owner_path: Optional[str]
    owner_run_status: Optional[str]
    documents_path: Optional[str]
    documents_content_hash: Optional[str]
    committed_hashes: Tuple[str, ...]
    chunks: Tuple[ChunkRef, ...]
    chunk_ids_version: Tuple[str, ...]
    chroma_verifiable: bool
    chroma_matches: bool
    fts_matches: bool
    embedding: Optional[Embedding]
    embedding_incoherent: bool
    classification: Classification
    has_documents: bool
    pending_ops: bool
    hash_column_available: bool = True

    @property
    def has_any_chunks(self) -> bool:
        return bool(self.chunks)


@dataclass(frozen=True)
class IncomingState:
    doc_id: str
    origin_path: str
    content_hash: str
    chunks: Tuple[ChunkRef, ...]
    embedding: Embedding
    classification: Classification

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_path(path) -> str:
    if path is None:
        return ""
    try:
        return os.path.realpath(os.path.expanduser(str(path)))
    except Exception:
        return str(path)


def text_sha256(text: Optional[str]) -> str:
    return sha256((text or "").encode("utf-8")).hexdigest()


def canonical_chunks(chunk_dicts: Iterable[dict]) -> Tuple[ChunkRef, ...]:
    """Canonical, order-independent chunk set sorted by ``chunk_id``."""
    refs = [ChunkRef(str(c.get("chunk_id", "")), text_sha256(c.get("text", "")))
            for c in chunk_dicts]
    refs.sort(key=lambda r: r.chunk_id)
    return tuple(refs)


def classification_from_fields(title, domain, source_type, language, version,
                               tags: Iterable[str]) -> Classification:
    return (
        title or None,
        domain or None,
        source_type or None,
        language or None,
        version or None,
        tuple(sorted({str(t) for t in (tags or [])})),
    )


def embedding_from_config() -> Embedding:
    from rag_lab import config
    return (
        str(getattr(config, "EMBEDDING_MODEL", "")),
        str(getattr(config, "EMBEDDING_MODEL_VERSION", "")),
        int(getattr(config, "EMBEDDING_DIM", 0)),
        int(getattr(config, "SPARSE_FORMAT_VERSION", 0)),
    )


def _table_exists(conn, name: str) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def _columns(conn, table: str) -> List[str]:
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except sqlite3.Error:
        return []


def _norm_hashes(values: Iterable[Optional[str]]) -> Tuple[str, ...]:
    return tuple(sorted({v for v in values if v}))


# ---------------------------------------------------------------------------
# Read-only adapter
# ---------------------------------------------------------------------------

class OwnershipStore:
    """Builds a :class:`PersistedState` snapshot from existing stores."""

    def __init__(self, conn, vector_store=None) -> None:
        self.conn = conn
        self.vector_store = vector_store

    # -- individual reads -------------------------------------------------

    def _chunks(self, doc_id: str) -> Tuple[Tuple[ChunkRef, ...], Optional[Embedding], bool]:
        try:
            rows = self.conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE doc_id=?", (doc_id,)
            ).fetchall()
        except sqlite3.Error:
            return (), None, True
        refs = tuple(sorted(
            (ChunkRef(str(r[0]), text_sha256(r[1])) for r in rows),
            key=lambda r: r.chunk_id,
        ))
        cols = set(_columns(self.conn, "chunks"))
        embedding = None
        incoherent = False
        if {"embedding_model_name", "embedding_model_version",
                "embedding_dim", "sparse_format_version"} <= cols:
            try:
                vals = self.conn.execute(
                    "SELECT DISTINCT embedding_model_name, embedding_model_version, "
                    "embedding_dim, sparse_format_version FROM chunks WHERE doc_id=?",
                    (doc_id,),
                ).fetchall()
            except sqlite3.Error:
                vals = []
            normalized = {
                (str(v[0] or ""), str(v[1] or ""), int(v[2] or 0), int(v[3] or 0))
                for v in vals
            }
            if len(normalized) > 1:
                incoherent = True
            elif len(normalized) == 1:
                value = normalized.pop()
                # Empty legacy metadata means "unknown", not "mismatch".
                if value != ("", "", 0, 0):
                    embedding = value
        return refs, embedding, incoherent

    def _fts(self, doc_id: str) -> Optional[Tuple[dict, int]]:
        """Return ({chunk_id: text_sha256}, row_count) for the FTS index.

        Comparing only ids is insufficient: a stale/corrupted FTS text with the
        same chunk id is an incoherent representation and must be rejected
        (P0C-SOL61-R1-F1).
        """
        if not _table_exists(self.conn, "chunks_fts"):
            return None
        try:
            rows = self.conn.execute(
                "SELECT chunk_id, text FROM chunks_fts WHERE doc_id=?", (doc_id,)
            ).fetchall()
        except sqlite3.Error:
            return None
        mapping = {str(r[0]): text_sha256(r[1]) for r in rows}
        return mapping, len(rows)

    def _chroma(self, doc_id: str, chunks: Tuple[ChunkRef, ...]):
        """Return (verifiable, matches)."""
        if self.vector_store is None or not hasattr(self.vector_store, "get_by_doc_id"):
            return False, False
        try:
            data = self.vector_store.get_by_doc_id(doc_id)
        except Exception:
            return False, False
        ids = {str(i) for i in (data.get("ids") or [])}
        documents = data.get("documents") or []
        texts = {}
        for cid, text in zip(data.get("ids") or [], documents):
            texts[str(cid)] = text_sha256(text)
        expected = {c.chunk_id: c.text_sha256 for c in chunks}
        return True, (ids == set(expected) and texts == expected)

    def _documents(self, doc_id: str):
        if not _table_exists(self.conn, "documents"):
            return None
        cols = set(_columns(self.conn, "documents"))
        if "path" not in cols:
            return None
        wanted = ["path", "content_hash", "title", "domain", "source_type",
                  "language", "version"]
        select = ", ".join(c for c in wanted if c in cols)
        try:
            row = self.conn.execute(
                f"SELECT {select} FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        values = dict(zip([c for c in wanted if c in cols], row))
        return values

    def _tags(self, doc_id: str) -> Tuple[str, ...]:
        if not (_table_exists(self.conn, "document_tags") and _table_exists(self.conn, "tags")):
            return ()
        try:
            rows = self.conn.execute(
                "SELECT t.name FROM document_tags dt JOIN tags t ON dt.tag_id=t.tag_id "
                "WHERE dt.doc_id=? ORDER BY t.name",
                (doc_id,),
            ).fetchall()
        except sqlite3.Error:
            return ()
        return tuple(str(r[0]) for r in rows)

    def _committed_hashes(self, doc_id: str, run_id: Optional[str]) -> Tuple[str, ...]:
        if not _table_exists(self.conn, "ingest_documents"):
            return ()
        try:
            if run_id is None:
                rows = self.conn.execute(
                    "SELECT content_hash FROM ingest_documents "
                    "WHERE doc_id=? AND status='COMMITTED'", (doc_id,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT content_hash FROM ingest_documents "
                    "WHERE doc_id=? AND run_id=? AND status='COMMITTED'",
                    (doc_id, run_id),
                ).fetchall()
        except sqlite3.Error:
            return ()
        return _norm_hashes(r[0] for r in rows)

    def _pending_ops(self, doc_id: str) -> bool:
        if not _table_exists(self.conn, "version_ops"):
            return False
        try:
            row = self.conn.execute(
                "SELECT 1 FROM version_ops WHERE doc_id=? AND state IN "
                "('WRITING','COMMITTING','RESTORING') LIMIT 1", (doc_id,),
            ).fetchone()
        except sqlite3.Error:
            return False
        return row is not None

    # -- public -----------------------------------------------------------

    def load(self, doc_id: str) -> PersistedState:
        chunks, embedding, embedding_incoherent = self._chunks(doc_id)
        fts = self._fts(doc_id)
        expected = {c.chunk_id: c.text_sha256 for c in chunks}
        if fts is None:
            # Missing FTS is incoherent if there is persisted content.
            fts_matches = not chunks
        else:
            fts_map, fts_count = fts
            fts_matches = (fts_map == expected and fts_count == len(chunks))
        chroma_verifiable, chroma_matches = self._chroma(doc_id, chunks)

        doc = self._documents(doc_id)
        tags = self._tags(doc_id)
        has_documents = doc is not None
        if doc is not None:
            classification = classification_from_fields(
                doc.get("title"), doc.get("domain"), doc.get("source_type"),
                doc.get("language"), doc.get("version"), tags,
            )
        else:
            classification = (None, None, None, None, None, ())

        has_version = False
        active_run_id = None
        generation = 0
        active_hash = None
        owner_path = None
        owner_status = None
        version_ids: Tuple[str, ...] = ()
        hash_column_available = True

        if _table_exists(self.conn, "document_versions"):
            cols = set(_columns(self.conn, "document_versions"))
            hash_column_available = "content_hash" in cols
            select = ["active_run_id", "generation", "chunk_ids"]
            if hash_column_available:
                select.append("content_hash")
            try:
                dv = self.conn.execute(
                    f"SELECT {', '.join(select)} FROM document_versions WHERE doc_id=?",
                    (doc_id,),
                ).fetchone()
            except sqlite3.Error:
                dv = None
            if dv is not None:
                has_version = True
                active_run_id = dv[0]
                generation = int(dv[1] or 0)
                try:
                    version_ids = tuple(json.loads(dv[2]) if dv[2] else [])
                except (TypeError, ValueError):
                    version_ids = ()
                if hash_column_available:
                    active_hash = dv[3] or None

        if has_version and active_run_id and _table_exists(self.conn, "ingest_runs"):
            try:
                run = self.conn.execute(
                    "SELECT source_path, status FROM ingest_runs WHERE run_id=?",
                    (active_run_id,),
                ).fetchone()
            except sqlite3.Error:
                run = None
            if run is not None:
                owner_path, owner_status = run[0], run[1]

        if not has_version and _table_exists(self.conn, "ingest_runs"):
            # Legacy (pre-versioning): the latest COMMITTED run owns the data.
            try:
                run = self.conn.execute(
                    "SELECT run_id, source_path, status FROM ingest_runs "
                    "WHERE doc_id=? AND status='COMMITTED' ORDER BY rowid DESC LIMIT 1",
                    (doc_id,),
                ).fetchone()
            except sqlite3.Error:
                run = None
            if run is not None:
                active_run_id, owner_path, owner_status = run[0], run[1], run[2]

        # Legacy fallback: documents.path may be the only owner evidence.
        if not has_version and owner_path is None and doc is not None:
            owner_path = doc.get("path")

        committed = self._committed_hashes(doc_id, active_run_id)

        return PersistedState(
            doc_id=doc_id,
            has_version=has_version,
            active_run_id=active_run_id,
            generation=generation,
            active_content_hash=active_hash,
            owner_path=owner_path,
            owner_run_status=owner_status,
            documents_path=(doc or {}).get("path"),
            documents_content_hash=(doc or {}).get("content_hash"),
            committed_hashes=committed,
            chunks=chunks,
            chunk_ids_version=version_ids,
            chroma_verifiable=chroma_verifiable,
            chroma_matches=chroma_matches,
            fts_matches=fts_matches,
            embedding=embedding,
            embedding_incoherent=embedding_incoherent,
            classification=classification,
            has_documents=has_documents,
            pending_ops=self._pending_ops(doc_id),
            hash_column_available=hash_column_available,
        )

    def evaluate(self, incoming: IncomingState, *, force: bool = False) -> Decision:
        return decide(self.load(incoming.doc_id), incoming, force=force)


# ---------------------------------------------------------------------------
# Pure decision
# ---------------------------------------------------------------------------

def decide(state: PersistedState, incoming: IncomingState, *,
           force: bool = False) -> Decision:
    """Return the authorization decision.  No I/O, no mutation."""
    # A pending journal operation means the confirmed state is not settled.
    if state.pending_ops:
        return Decision(DecisionKind.DENY_INCOHERENT,
                        "pending version operation; recovery required")

    same_path = bool(state.owner_path) and (
        normalize_path(state.owner_path) == normalize_path(incoming.origin_path))

    if state.has_version:
        if not state.owner_run_status or state.owner_run_status != "COMMITTED":
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "active run is not a COMMITTED run")
        if not state.hash_column_available:
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "confirmed content-hash column unavailable")
        if len(set(state.committed_hashes)) > 1:
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "contradictory COMMITTED hashes in the active run")
        if state.active_content_hash and state.committed_hashes and \
                state.active_content_hash not in state.committed_hashes:
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "active version hash is not corroborated")
        if state.documents_path and state.owner_path and \
                normalize_path(state.documents_path) != normalize_path(state.owner_path):
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "documents.path disagrees with the active run path")
        if state.documents_content_hash and state.committed_hashes and \
                state.documents_content_hash not in state.committed_hashes:
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "documents hash is not corroborated")
        if state.embedding_incoherent:
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "incoherent embedding metadata across chunks")
        if not state.chroma_verifiable:
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "SQL/Chroma consistency is not verifiable")
        if not state.chroma_matches or not state.fts_matches:
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "SQL/FTS/Chroma representations disagree")
        if state.chunk_ids_version and \
                set(state.chunk_ids_version) != {c.chunk_id for c in state.chunks}:
            return Decision(DecisionKind.DENY_INCOHERENT,
                            "document_versions.chunk_ids disagrees with chunks")

        representation_equal = state.chunks == incoming.chunks
        embedding_equal = (state.embedding is None) or (state.embedding == incoming.embedding)
        classification_equal = state.classification == incoming.classification
        pipeline_equal = embedding_equal and classification_equal

        if same_path:
            if representation_equal and pipeline_equal and not force:
                return Decision(DecisionKind.NOOP_OWNER,
                                "owner re-ingest with identical persisted result",
                                state.owner_path, state.generation, state.active_run_id)
            return Decision(DecisionKind.REPLACE,
                            "owner update (content/embedding/classification changed)",
                            state.owner_path, state.generation, state.active_run_id)
        if representation_equal and pipeline_equal:
            return Decision(DecisionKind.NOOP_EQUIVALENT_OTHER_SOURCE,
                            "exact persisted equivalence from another path; "
                            "no ownership change",
                            state.owner_path, state.generation, state.active_run_id)
        return Decision(DecisionKind.DENY_NOT_OWNER,
                        f"doc_id {incoming.doc_id!r} is owned by "
                        f"{state.owner_path!r}; different path with different content",
                        state.owner_path, state.generation, state.active_run_id)

    # No confirmed version: legacy or a genuinely new document.
    if not state.has_any_chunks and not state.has_documents:
        return Decision(DecisionKind.CREATE, "new document")

    if not state.owner_path:
        return Decision(DecisionKind.DENY_INSUFFICIENT_EVIDENCE,
                        "persisted data without a provable owner")
    if len(set(state.committed_hashes)) > 1:
        return Decision(DecisionKind.DENY_INCOHERENT,
                        "contradictory COMMITTED hashes in the legacy owner run")
    if state.documents_path and state.owner_path and \
            normalize_path(state.documents_path) != normalize_path(state.owner_path):
        return Decision(DecisionKind.DENY_INCOHERENT,
                        "documents.path disagrees with the legacy owner run")
    if state.embedding_incoherent:
        return Decision(DecisionKind.DENY_INCOHERENT,
                        "incoherent embedding metadata across chunks")
    if not state.chroma_verifiable or not state.chroma_matches or not state.fts_matches:
        return Decision(DecisionKind.DENY_INCOHERENT,
                        "SQL/FTS/Chroma representations disagree")

    representation_equal = state.chunks == incoming.chunks
    embedding_equal = (state.embedding is None) or (state.embedding == incoming.embedding)
    classification_equal = (not state.has_documents) or (
        state.classification == incoming.classification)
    if same_path:
        # The owner is confirmed by the COMMITTED run path; the owner may
        # replace content even when the pre-image has no exact representation.
        if representation_equal and embedding_equal and classification_equal:
            return Decision(DecisionKind.NOOP_OWNER,
                            "legacy owner re-ingest with identical persisted result",
                            state.owner_path, state.generation, state.active_run_id)
        return Decision(DecisionKind.REPLACE,
                        "legacy owner update (content/embedding/classification changed)",
                        state.owner_path, state.generation, state.active_run_id)
    if representation_equal and embedding_equal and classification_equal:
        return Decision(DecisionKind.NOOP_EQUIVALENT_OTHER_SOURCE,
                        "legacy exact persisted equivalence from another path; "
                        "no ownership change",
                        state.owner_path, state.generation, state.active_run_id)
    return Decision(DecisionKind.DENY_INSUFFICIENT_EVIDENCE,
                    "legacy representation differs and ownership cannot be proven",
                    state.owner_path, state.generation, state.active_run_id)


# ---------------------------------------------------------------------------
# Convenience façade for the ingest pipeline
# ---------------------------------------------------------------------------

def evaluate(conn, vector_store, *, doc_id: str, origin_path, content_hash: str,
             chunk_dicts: Sequence[dict], classification: Classification,
             force: bool = False) -> Decision:
    incoming = IncomingState(
        doc_id=doc_id,
        origin_path=str(origin_path),
        content_hash=content_hash,
        chunks=canonical_chunks(chunk_dicts),
        embedding=embedding_from_config(),
        classification=classification,
    )
    return OwnershipStore(conn, vector_store).evaluate(incoming, force=force)
