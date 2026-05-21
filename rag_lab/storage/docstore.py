"""SQLite-based document store for chunk text and metadata.

Schema v2 additions (idempotent migration in initialize()):
  - sparse_tokens BLOB, sparse_weights BLOB  (BGE-M3 sparse vectors)
  - embedding_model_name, embedding_model_version, embedding_dim
  - sparse_format_version
  - chunks_fts virtual table (FTS5 BM25)
  - idx_chunks_doc_id, idx_chunks_model indexes
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from rag_lab.config import STORAGE_DIR, DOCDSTORE_SQLITE_PATH

logger = logging.getLogger("rag_lab")


class DocStore:
    """SQLite-based document store for chunk text and metadata."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DOCDSTORE_SQLITE_PATH
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Open DB, create/migrate schema, ensure FTS5 virtual table exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode = WAL")

        # Base table (v1)
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

        # Idempotent migrations
        self._migrate_v2()
        self._migrate_v3()
        self._migrate_v4()
        self._conn.commit()
        logger.info(f"Initialized docstore at {self.db_path}")

    def _migrate_v2(self) -> None:
        """Add v2 columns and FTS5 table if they don't exist yet."""
        existing = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        }

        new_cols = [
            ("sparse_tokens", "BLOB"),
            ("sparse_weights", "BLOB"),
            ("embedding_model_name", "TEXT DEFAULT ''"),
            ("embedding_model_version", "TEXT DEFAULT ''"),
            ("embedding_dim", "INTEGER DEFAULT 0"),
            ("sparse_format_version", "INTEGER DEFAULT 1"),
        ]
        for col, definition in new_cols:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {definition}")
                logger.info(f"DocStore: added column {col}")

        # FTS5 virtual table (same DB file)
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                doc_id   UNINDEXED,
                text,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )

        # Indexes
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_model "
            "ON chunks(embedding_model_name, embedding_model_version)"
        )

    def _migrate_v3(self) -> None:
        """Create v3 metadata tables if they don't exist yet."""
        from rag_lab.storage.metadata_store import MetadataStore
        MetadataStore(conn=self._conn).initialize()

    def _migrate_v4(self) -> None:
        """Create ingest_runs table for transaction tracking (v1.4)."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_runs (
                run_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                source_path TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'IN_PROGRESS',
                error_message TEXT,
                chunks_expected INTEGER DEFAULT 0,
                chunks_written_docstore INTEGER DEFAULT 0,
                chunks_written_fts5 INTEGER DEFAULT 0,
                chunks_written_chroma INTEGER DEFAULT 0,
                chunks_written_sparse INTEGER DEFAULT 0,
                metadata_written INTEGER DEFAULT 0
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingest_runs_doc_status "
            "ON ingest_runs(doc_id, status)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingest_runs_status "
            "ON ingest_runs(status)"
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, chunks: List[dict]) -> None:
        """Add chunks to the document store and FTS5 index.

        Each chunk dict may include:
          sparse_tokens, sparse_weights (bytes)  — from encode_chunks()
          embedding_model_name, embedding_model_version, embedding_dim
          sparse_format_version
        """
        if self._conn is None:
            self.initialize()

        for chunk in chunks:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, text, heading_path, tipo, posicion_relativa,
                 n_tokens, line_start, line_end,
                 sparse_tokens, sparse_weights,
                 embedding_model_name, embedding_model_version,
                 embedding_dim, sparse_format_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    chunk.get("sparse_tokens"),
                    chunk.get("sparse_weights"),
                    chunk.get("embedding_model_name", ""),
                    chunk.get("embedding_model_version", ""),
                    chunk.get("embedding_dim", 0),
                    chunk.get("sparse_format_version", 1),
                ),
            )
            # Sync FTS5
            self._conn.execute(
                "INSERT OR REPLACE INTO chunks_fts(chunk_id, doc_id, text) VALUES (?, ?, ?)",
                (chunk.get("chunk_id", ""), chunk.get("doc_id", ""), chunk.get("text", "")),
            )

        self._conn.commit()
        logger.info(f"Added {len(chunks)} chunks to docstore + FTS5")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, chunk_id: str) -> Optional[dict]:
        if self._conn is None:
            self.initialize()
        cursor = self._conn.execute(
            "SELECT chunk_id, doc_id, text, heading_path, tipo, "
            "posicion_relativa, n_tokens, line_start, line_end "
            "FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []
        if self._conn is None:
            self.initialize()
        placeholders = ", ".join(["?"] * len(chunk_ids))
        cursor = self._conn.execute(
            f"SELECT chunk_id, doc_id, text, heading_path, tipo, "
            f"posicion_relativa, n_tokens, line_start, line_end "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_dict(row) -> dict:
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

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def count(self) -> int:
        if self._conn is None:
            self.initialize()
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def count_chunks(self, doc_id: str) -> int:
        if self._conn is None:
            self.initialize()
        return self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_doc_id(self, doc_id: str) -> int:
        if self._conn is None:
            self.initialize()
        cursor = self._conn.execute(
            "DELETE FROM chunks WHERE doc_id = ?", (doc_id,)
        )
        count = cursor.rowcount
        self._conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        from rag_lab.storage.metadata_store import MetadataStore
        MetadataStore(conn=self._conn).delete_document(doc_id)
        self._conn.commit()
        logger.info(f"Deleted {count} chunks for doc_id={doc_id!r}")
        return count

    def delete_all(self) -> None:
        if self._conn is None:
            self.initialize()
        self._conn.execute("DELETE FROM chunks")
        self._conn.execute("DELETE FROM chunks_fts")
        self._conn.commit()
        logger.info("Deleted all chunks from docstore")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
