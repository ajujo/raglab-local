"""SQLite-based document store for chunk text and metadata.

Stores the full text of each chunk for later retrieval and generation.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from rag_lab.config import STORAGE_DIR, DOCDSTORE_SQLITE_PATH

logger = logging.getLogger("rag_lab")


class DocStore:
    """SQLite-based document store for chunk text and metadata."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
    ):
        """Initialize the document store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path or DOCDSTORE_SQLITE_PATH
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Initialize the database and create tables if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                heading_path TEXT,
                tipo TEXT,
                posicion_relativa REAL,
                n_tokens INTEGER,
                line_start INTEGER DEFAULT 0,
                line_end INTEGER DEFAULT 0
            )
            """
        )
        self._conn.commit()
        logger.info(f"Initialized docstore at {self.db_path}")

    def add(
        self,
        chunks: List[dict],
    ) -> None:
        """Add chunks to the document store.

        Args:
            chunks: List of chunk dicts with all metadata.
        """
        if self._conn is None:
            self.initialize()

        for chunk in chunks:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, text, heading_path, tipo, posicion_relativa, n_tokens, line_start, line_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.get("chunk_id", ""),
                    chunk.get("doc_id", ""),
                    chunk.get("text", ""),
                    chunk.get("heading_path", ""),
                    chunk.get("tipo", "texto"),
                    chunk.get("posicion_relativa", 0.0),
                    chunk.get("n_tokens", 0),
                    chunk.get("line_start", 0),
                    chunk.get("line_end", 0),
                ),
            )
        self._conn.commit()
        logger.info(f"Added {len(chunks)} chunks to docstore")

    def get_by_id(self, chunk_id: str) -> Optional[dict]:
        """Retrieve a chunk by its ID.

        Args:
            chunk_id: The chunk ID.

        Returns:
            Dict with chunk data or None if not found.
        """
        if self._conn is None:
            self.initialize()

        cursor = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "chunk_id": row[0],
                "doc_id": row[1],
                "text": row[2],
                "heading_path": row[3],
                "tipo": row[4],
                "posicion_relativa": row[5],
                "n_tokens": row[6],
                "line_start": row[7],
                "line_end": row[8],
            }
        return None

    def get_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """Retrieve multiple chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs.

        Returns:
            List of chunk dicts.
        """
        if self._conn is None:
            self.initialize()

        placeholders = ", ".join(["?"] * len(chunk_ids))
        cursor = self._conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "chunk_id": row[0],
                "doc_id": row[1],
                "text": row[2],
                "heading_path": row[3],
                "tipo": row[4],
                "posicion_relativa": row[5],
                "n_tokens": row[6],
                "line_start": row[7],
                "line_end": row[8],
            })
        return results

    def count(self) -> int:
        """Return the number of chunks stored."""
        if self._conn is None:
            self.initialize()
        cursor = self._conn.execute("SELECT COUNT(*) FROM chunks")
        return cursor.fetchone()[0]

    def count_chunks(self, doc_id: str) -> int:
        """Return the number of chunks for a specific document.

        Args:
            doc_id: The document ID to count chunks for.

        Returns:
            Number of chunks for the given document.
        """
        if self._conn is None:
            self.initialize()
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        )
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def delete_all(self) -> None:
        """Delete all chunks from the store."""
        if self._conn is None:
            self.initialize()
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()
        logger.info("Deleted all chunks from docstore")