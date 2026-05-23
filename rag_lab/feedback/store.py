"""Structured chunk-level feedback store — v1.15+

Records per-chunk relevance signals from user interaction.
Does NOT affect retrieval ranking, scoring, or cache.

Backend: SQLite (defaults to docstore.sqlite so a single backup
captures corpus metadata and user feedback together).

Schema: feedback_events table — one row per (query, chunk) judgment.

Usage
-----
    store = FeedbackStore()
    store.initialize()
    store.add("What is SDMX?", chunk_id="abc123", doc_id="sdmx_glossary",
              rank=1, feedback="relevant")
    rows = store.list()
    print(store.stats())
    store.export_jsonl()
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from rag_lab.config import (
    BM25_RRF_WEIGHT,
    DENSE_RRF_WEIGHT,
    EMBEDDING_MODEL_VERSION,
    FEEDBACK_DB_PATH,
    HYDE_ENABLED,
    MMR_ENABLED,
    MMR_LAMBDA,
    RERANKER_USE_HEADING_CONTEXT,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    RRF_K,
    SPARSE_FORMAT_VERSION,
    SPARSE_RRF_WEIGHT,
)

logger = logging.getLogger("rag_lab")

VALID_FEEDBACK = frozenset([
    "relevant",
    "irrelevant",
    "useful",
    "not_useful",
    "wrong_doc",
    "outdated",
    "duplicate",
    "bad_citation",
])

_CREATE_FEEDBACK_EVENTS = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    query_text            TEXT NOT NULL,
    query_hash            TEXT NOT NULL,
    normalized_query      TEXT NOT NULL,
    chunk_id              TEXT NOT NULL,
    doc_id                TEXT NOT NULL,
    rank                  INTEGER,
    feedback              TEXT NOT NULL,
    rating                INTEGER,
    reason                TEXT,
    source                TEXT NOT NULL DEFAULT 'cli',
    pipeline_variant      TEXT NOT NULL DEFAULT '',
    cache_hit             INTEGER NOT NULL DEFAULT 0,
    cache_key             TEXT,
    corpus_fingerprint    TEXT NOT NULL DEFAULT '',
    retrieval_config_hash TEXT NOT NULL DEFAULT '',
    user_note             TEXT
)
"""


class FeedbackStore:
    """Chunk-level user feedback store backed by SQLite.

    IMPORTANT: Reading or writing to this store NEVER modifies retrieval
    rankings, scores, or the query cache. It is purely an append-only log.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or FEEDBACK_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(_CREATE_FEEDBACK_EVENTS)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fe_query_hash "
            "ON feedback_events(query_hash)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fe_chunk_id "
            "ON feedback_events(chunk_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fe_feedback "
            "ON feedback_events(feedback)"
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_open(self) -> None:
        if self._conn is None:
            self.initialize()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(
        self,
        query_text: str,
        chunk_id: str,
        doc_id: str,
        rank: Optional[int],
        feedback: str,
        *,
        rating: Optional[int] = None,
        reason: Optional[str] = None,
        source: str = "cli",
        pipeline_variant: str = "",
        cache_hit: bool = False,
        cache_key: Optional[str] = None,
        corpus_fingerprint: str = "",
        retrieval_config_hash: str = "",
        user_note: Optional[str] = None,
    ) -> int:
        """Record a chunk-level feedback event. Returns the new row id."""
        if feedback not in VALID_FEEDBACK:
            raise ValueError(
                f"Invalid feedback {feedback!r}. "
                f"Valid values: {sorted(VALID_FEEDBACK)}"
            )
        self._ensure_open()
        norm_q = _normalize_query(query_text)
        q_hash = make_query_hash(query_text)
        if not retrieval_config_hash:
            retrieval_config_hash = make_retrieval_config_hash()
        cur = self._conn.execute(
            """
            INSERT INTO feedback_events (
                query_text, query_hash, normalized_query,
                chunk_id, doc_id, rank, feedback,
                rating, reason, source, pipeline_variant,
                cache_hit, cache_key, corpus_fingerprint,
                retrieval_config_hash, user_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_text, q_hash, norm_q,
                chunk_id, doc_id, rank, feedback,
                rating, reason, source, pipeline_variant,
                int(cache_hit), cache_key, corpus_fingerprint,
                retrieval_config_hash, user_note,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def list(
        self,
        *,
        query_hash: Optional[str] = None,
        chunk_id: Optional[str] = None,
        feedback: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Return feedback events, most recent first."""
        self._ensure_open()
        clauses = []
        params: list = []
        if query_hash is not None:
            clauses.append("query_hash = ?")
            params.append(query_hash)
        if chunk_id is not None:
            clauses.append("chunk_id = ?")
            params.append(chunk_id)
        if feedback is not None:
            clauses.append("feedback = ?")
            params.append(feedback)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT id, created_at, query_text, query_hash, normalized_query,
                   chunk_id, doc_id, rank, feedback, rating, reason,
                   source, pipeline_variant, cache_hit, cache_key,
                   corpus_fingerprint, retrieval_config_hash, user_note
            FROM feedback_events
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        cols = [
            "id", "created_at", "query_text", "query_hash", "normalized_query",
            "chunk_id", "doc_id", "rank", "feedback", "rating", "reason",
            "source", "pipeline_variant", "cache_hit", "cache_key",
            "corpus_fingerprint", "retrieval_config_hash", "user_note",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def stats(self) -> dict:
        """Return aggregate counts by feedback type."""
        self._ensure_open()
        total = self._conn.execute(
            "SELECT COUNT(*) FROM feedback_events"
        ).fetchone()[0]
        rows = self._conn.execute(
            "SELECT feedback, COUNT(*) FROM feedback_events GROUP BY feedback ORDER BY feedback"
        ).fetchall()
        by_type = {r[0]: r[1] for r in rows}

        unique_queries = self._conn.execute(
            "SELECT COUNT(DISTINCT query_hash) FROM feedback_events"
        ).fetchone()[0]
        unique_chunks = self._conn.execute(
            "SELECT COUNT(DISTINCT chunk_id) FROM feedback_events"
        ).fetchone()[0]
        return {
            "total_events": total,
            "unique_queries": unique_queries,
            "unique_chunks": unique_chunks,
            "by_feedback_type": by_type,
        }

    def export_jsonl(self, path: Optional[Path] = None) -> str:
        """Serialize all events to JSONL. Returns the JSONL string.

        If path is given, also writes to that file.
        """
        rows = self.list(limit=10_000_000)
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        jsonl = "\n".join(lines)
        if path is not None:
            Path(path).write_text(jsonl, encoding="utf-8")
        return jsonl

    def clear(self) -> int:
        """Delete all feedback events. Returns number of rows deleted."""
        self._ensure_open()
        cur = self._conn.execute("DELETE FROM feedback_events")
        self._conn.commit()
        return cur.rowcount


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalize_query(text: str) -> str:
    return " ".join(text.strip().lower().split())


def make_query_hash(query_text: str) -> str:
    """Stable SHA-256 of normalized query text."""
    norm = _normalize_query(query_text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def make_retrieval_config_hash() -> str:
    """Stable SHA-256 of the pipeline config params that affect retrieval results."""
    components = {
        "top_k": RETRIEVAL_TOP_K,
        "rerank_top_k": RERANK_TOP_K,
        "rrf_k": RRF_K,
        "dw": DENSE_RRF_WEIGHT,
        "bw": BM25_RRF_WEIGHT,
        "sw": SPARSE_RRF_WEIGHT,
        "mmr": MMR_ENABLED,
        "mmr_l": MMR_LAMBDA,
        "rhc": RERANKER_USE_HEADING_CONTEXT,
        "hyde": HYDE_ENABLED,
        "emb_v": EMBEDDING_MODEL_VERSION,
        "sfv": SPARSE_FORMAT_VERSION,
    }
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Module-level singleton (lazy)
# ------------------------------------------------------------------

_store_instance: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Return the module-level FeedbackStore singleton, initialized on first call."""
    global _store_instance
    if _store_instance is None:
        _store_instance = FeedbackStore()
        _store_instance.initialize()
    return _store_instance


def reset_feedback_store_instance() -> None:
    """Reset the singleton (used in tests to get a fresh instance per tmp_path)."""
    global _store_instance
    if _store_instance is not None:
        _store_instance.close()
        _store_instance = None
