"""FTS5-based BM25 full-text search over chunks.

Uses SQLite's built-in FTS5 extension (available in CPython's sqlite3).
The virtual table `chunks_fts` lives in the same docstore.sqlite file and
is populated by DocStore.add() — this class is query-only.

Typical usage:
    fts = FTSStore()
    results = fts.query("SDMX metadata structure", top_k=30)
    # returns [{"id": chunk_id, "bm25_score": float}, ...]
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

from rag_lab.config import DOCDSTORE_SQLITE_PATH

logger = logging.getLogger("rag_lab")


class FTSStore:
    """BM25 full-text search backed by SQLite FTS5.

    The underlying table (`chunks_fts`) is created and populated by DocStore.
    FTSStore only queries it.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DOCDSTORE_SQLITE_PATH
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode = WAL")
        logger.debug(f"FTSStore opened {self.db_path}")

    def query(
        self,
        text: str,
        top_k: int = 30,
        doc_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """BM25 search using FTS5 MATCH.

        Args:
            text: User query string.
            top_k: Maximum results to return.
            doc_ids: If given, restrict to these document IDs.

        Returns:
            List of {"id": chunk_id, "bm25_score": float}, best first.
            bm25_score is positive; higher = more relevant.
        """
        if self._conn is None:
            self.initialize()

        escaped = self._escape_query(text)
        if not escaped:
            return []

        try:
            if doc_ids:
                placeholders = ",".join(["?"] * len(doc_ids))
                rows = self._conn.execute(
                    f"""
                    SELECT chunk_id, -rank AS bm25
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                      AND doc_id IN ({placeholders})
                    ORDER BY rank
                    LIMIT ?
                    """,
                    [escaped, *doc_ids, top_k],
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT chunk_id, -rank AS bm25
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (escaped, top_k),
                ).fetchall()
        except sqlite3.OperationalError as e:
            # chunks_fts table doesn't exist yet (pre-migration) — return empty
            logger.warning(f"FTSStore.query failed (table may not exist): {e}")
            return []

        results = [{"id": row[0], "bm25_score": float(row[1])} for row in rows]
        logger.debug(f"FTS5 BM25 returned {len(results)} results for: {text[:60]!r}")
        return results

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _escape_query(text: str) -> str:
        """Convert a free-text query into a safe FTS5 MATCH expression.

        Strategy: strip FTS5 operators and special chars, split on whitespace,
        wrap each token in double quotes (exact-token match, implicit AND).
        This is conservative but safe for any input.
        """
        # Remove chars that have special meaning in FTS5
        cleaned = re.sub(r'[^\w\s\-]', ' ', text, flags=re.UNICODE)
        tokens = [t for t in cleaned.split() if t and t != '-']
        if not tokens:
            return ""
        # Wrap in double quotes to treat each as a literal token
        return " ".join(f'"{tok}"' for tok in tokens)
