"""Persistent query cache for retrieval+reranking results.

Caches the ranked chunk list produced by hybrid_search + reranker for a given
(query, config, corpus) combination. Does NOT cache LLM responses.

Cache key is a SHA-256 of all inputs that affect retrieval determinism:
  - normalized query text
  - filter spec (doc_ids / tags)
  - pipeline parameters (top_k, rrf_k, weights, MMR, HyDE, reranker flags)
  - model versions (embedding, sparse format)
  - corpus fingerprint (chunk count + max ingest_run id)

The corpus fingerprint automatically invalidates entries when documents are
ingested or deleted — no manual invalidation required for corpus changes.

Backend: SQLite (data/query_cache.sqlite). WAL mode for concurrent reads.
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from rag_lab.config import (
    BM25_RRF_WEIGHT,
    DENSE_RRF_WEIGHT,
    EMBEDDING_MODEL_VERSION,
    HYDE_ENABLED,
    MMR_ENABLED,
    MMR_LAMBDA,
    QUERY_CACHE_ENABLED,
    QUERY_CACHE_PATH,
    QUERY_CACHE_TTL_SECONDS,
    RERANKER_USE_HEADING_CONTEXT,
    RRF_K,
    RETRIEVAL_TOP_K,
    SPARSE_FORMAT_VERSION,
    SPARSE_RRF_WEIGHT,
)

logger = logging.getLogger("rag_lab")

_SCHEMA_VERSION = 1


class QueryCache:
    """SQLite-backed persistent cache for retrieval results.

    Thread-safe for concurrent readers; single-writer (WAL mode).
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or QUERY_CACHE_PATH
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open the cache DB and create schema if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_cache (
                cache_key       TEXT PRIMARY KEY,
                corpus_fp       TEXT NOT NULL,
                created_at      INTEGER NOT NULL,
                last_accessed_at INTEGER NOT NULL,
                hit_count       INTEGER NOT NULL DEFAULT 0,
                query_norm      TEXT,
                result_json     TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qc_created ON query_cache(created_at)"
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

    def get(self, cache_key: str, corpus_fp: str) -> Optional[List[dict]]:
        """Return cached results if the key and corpus fingerprint match.

        Returns None on any miss (key absent, corpus changed, or TTL expired).
        Updates hit_count and last_accessed_at on a hit.
        """
        self._ensure_open()
        now = int(time.time())
        row = self._conn.execute(
            "SELECT result_json, corpus_fp, created_at FROM query_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None

        result_json, stored_fp, created_at = row

        if stored_fp != corpus_fp:
            return None

        if QUERY_CACHE_TTL_SECONDS > 0 and (now - created_at) >= QUERY_CACHE_TTL_SECONDS:
            return None

        self._conn.execute(
            "UPDATE query_cache SET hit_count = hit_count + 1, last_accessed_at = ? "
            "WHERE cache_key = ?",
            (now, cache_key),
        )
        self._conn.commit()
        return json.loads(result_json)

    def set(
        self,
        cache_key: str,
        corpus_fp: str,
        results: List[dict],
        query_norm: str = "",
    ) -> None:
        """Store retrieval results under the given key."""
        self._ensure_open()
        now = int(time.time())
        self._conn.execute(
            """
            INSERT OR REPLACE INTO query_cache
                (cache_key, corpus_fp, created_at, last_accessed_at, hit_count, query_norm, result_json)
            VALUES (?, ?, ?, ?, 0, ?, ?)
            """,
            (cache_key, corpus_fp, now, now, query_norm, json.dumps(results)),
        )
        self._conn.commit()

    def invalidate(self, cache_key: str) -> bool:
        """Remove a single entry. Returns True if it existed."""
        self._ensure_open()
        cur = self._conn.execute(
            "DELETE FROM query_cache WHERE cache_key = ?", (cache_key,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def clear(self) -> int:
        """Remove all entries. Returns number of rows deleted."""
        self._ensure_open()
        cur = self._conn.execute("DELETE FROM query_cache")
        self._conn.commit()
        return cur.rowcount

    def vacuum(self) -> None:
        """Remove expired or stale entries, then VACUUM the DB."""
        self._ensure_open()
        now = int(time.time())
        if QUERY_CACHE_TTL_SECONDS > 0:
            self._conn.execute(
                "DELETE FROM query_cache WHERE ? - created_at >= ?",
                (now, QUERY_CACHE_TTL_SECONDS),
            )
        self._conn.commit()
        self._conn.execute("VACUUM")

    def stats(self) -> dict:
        """Return summary statistics for the cache."""
        self._ensure_open()
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total_entries,
                COALESCE(SUM(hit_count), 0) AS total_hits,
                COALESCE(MIN(created_at), 0) AS oldest_entry,
                COALESCE(MAX(last_accessed_at), 0) AS latest_access
            FROM query_cache
            """
        ).fetchone()
        total_entries, total_hits, oldest_entry, latest_access = row
        size_bytes = self._path.stat().st_size if self._path.exists() else 0
        return {
            "total_entries": total_entries,
            "total_hits": int(total_hits),
            "oldest_entry_age_s": int(time.time()) - oldest_entry if oldest_entry else 0,
            "latest_access_age_s": int(time.time()) - latest_access if latest_access else 0,
            "db_size_bytes": size_bytes,
            "ttl_seconds": QUERY_CACHE_TTL_SECONDS,
            "enabled": QUERY_CACHE_ENABLED,
        }

    def inspect(self, cache_key: str) -> Optional[dict]:
        """Return metadata for one entry (without the full result payload)."""
        self._ensure_open()
        row = self._conn.execute(
            """
            SELECT cache_key, corpus_fp, created_at, last_accessed_at, hit_count, query_norm
            FROM query_cache WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        now = int(time.time())
        return {
            "cache_key": row[0],
            "corpus_fp": row[1],
            "created_at": row[2],
            "age_s": now - row[2],
            "last_accessed_at": row[3],
            "hit_count": row[4],
            "query_norm": row[5],
        }


# ------------------------------------------------------------------
# Cache key construction
# ------------------------------------------------------------------

def make_cache_key(
    query: str,
    *,
    filters: Optional[dict] = None,
    top_k: int = None,
    rrf_k: int = None,
    dense_weight: float = None,
    bm25_weight: float = None,
    sparse_weight: float = None,
    mmr_enabled: bool = None,
    mmr_lambda: float = None,
    reranker_use_heading_context: bool = None,
    hyde_enabled: bool = None,
    embedding_model_version: str = None,
    sparse_format_version: int = None,
    corpus_fingerprint: str = "",
) -> str:
    """Compute a stable SHA-256 cache key for a retrieval request.

    All parameters default to the live config values so callers need only
    pass overrides. The key changes if any input differs, including corpus state.
    """
    norm_query = " ".join(query.strip().lower().split())
    components = {
        "q": norm_query,
        "filters": filters or {},
        "top_k": top_k if top_k is not None else RETRIEVAL_TOP_K,
        "rrf_k": rrf_k if rrf_k is not None else RRF_K,
        "dw": dense_weight if dense_weight is not None else DENSE_RRF_WEIGHT,
        "bw": bm25_weight if bm25_weight is not None else BM25_RRF_WEIGHT,
        "sw": sparse_weight if sparse_weight is not None else SPARSE_RRF_WEIGHT,
        "mmr": mmr_enabled if mmr_enabled is not None else MMR_ENABLED,
        "mmr_l": mmr_lambda if mmr_lambda is not None else MMR_LAMBDA,
        "rhc": reranker_use_heading_context
        if reranker_use_heading_context is not None
        else RERANKER_USE_HEADING_CONTEXT,
        "hyde": hyde_enabled if hyde_enabled is not None else HYDE_ENABLED,
        "emb_v": embedding_model_version or EMBEDDING_MODEL_VERSION,
        "sfv": sparse_format_version if sparse_format_version is not None else SPARSE_FORMAT_VERSION,
        "corpus": corpus_fingerprint,
    }
    canonical = json.dumps(components, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_corpus_fingerprint(conn: sqlite3.Connection) -> str:
    """Return a lightweight fingerprint of the current corpus state.

    Changes whenever:
      - a document is ingested or deleted (n_chunks + max ingest_run id)
      - a tag is assigned, unassigned, renamed, or deleted (cache_revision)

    The cache_revision counter is bumped by MetadataStore on every mutating
    tag/document operation, ensuring stale tag-filtered cache entries are
    automatically invalidated.
    """
    try:
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except Exception:
        n_chunks = 0
    try:
        max_run = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM ingest_runs"
        ).fetchone()[0]
    except Exception:
        max_run = 0
    try:
        rev_row = conn.execute(
            "SELECT value FROM cache_revision WHERE key = 'retrieval'"
        ).fetchone()
        revision = rev_row[0] if rev_row else 0
    except Exception:
        revision = 0
    return f"{n_chunks}:{max_run}:{revision}"


# ------------------------------------------------------------------
# Module-level singleton (lazy)
# ------------------------------------------------------------------

_cache_instance: Optional[QueryCache] = None


def get_cache() -> QueryCache:
    """Return the module-level cache singleton, initialized on first call."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache()
        _cache_instance.initialize()
    return _cache_instance


def reset_cache_instance() -> None:
    """Reset the singleton (used in tests to get a fresh instance per tmp_path)."""
    global _cache_instance
    if _cache_instance is not None:
        _cache_instance.close()
        _cache_instance = None
