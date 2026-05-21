import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from rag_lab.config import DOCDSTORE_SQLITE_PATH

logger = logging.getLogger("rag_lab")

_CREATE_SOURCES = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    url TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_DATASETS = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    path TEXT,
    content_hash TEXT,
    source_id TEXT REFERENCES sources(source_id),
    dataset_id TEXT REFERENCES datasets(dataset_id),
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    embedding_model_version TEXT DEFAULT '',
    embedding_dim INTEGER DEFAULT 0,
    sparse_format_version INTEGER DEFAULT 0
)
"""

_CREATE_TAGS = """
CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
)
"""

_CREATE_DOCUMENT_TAGS = """
CREATE TABLE IF NOT EXISTS document_tags (
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
    PRIMARY KEY (doc_id, tag_id)
)
"""


class MetadataStore:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        conn: Optional[sqlite3.Connection] = None,
    ):
        if conn is not None:
            self._conn = conn
            self._own_conn = False
        else:
            self._db_path = db_path or DOCDSTORE_SQLITE_PATH
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._own_conn = True

    def initialize(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self._own_conn:
            self._conn.execute("PRAGMA journal_mode = WAL")

        for ddl in (
            _CREATE_SOURCES,
            _CREATE_DATASETS,
            _CREATE_DOCUMENTS,
            _CREATE_TAGS,
            _CREATE_DOCUMENT_TAGS,
        ):
            self._conn.execute(ddl)

        if self._own_conn:
            self._conn.commit()

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    def upsert_document(
        self,
        doc_id: str,
        *,
        title: Optional[str] = None,
        path: Optional[str] = None,
        content_hash: Optional[str] = None,
        source_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        status: str = "active",
        embedding_model_version: str = "",
        embedding_dim: int = 0,
        sparse_format_version: int = 0,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO documents
            (doc_id, title, path, content_hash, source_id, dataset_id, status,
             updated_at, embedding_model_version, embedding_dim, sparse_format_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?)
            """,
            (
                doc_id,
                title,
                path,
                content_hash,
                source_id,
                dataset_id,
                status,
                embedding_model_version,
                embedding_dim,
                sparse_format_version,
            ),
        )
        if self._own_conn:
            self._conn.commit()

    def get_document(self, doc_id: str) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT doc_id, title, path, content_hash, source_id, dataset_id,
                   status, created_at, updated_at, ingested_at,
                   embedding_model_version, embedding_dim, sparse_format_version
            FROM documents WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        doc = {
            "doc_id": row[0],
            "title": row[1],
            "path": row[2],
            "content_hash": row[3],
            "source_id": row[4],
            "dataset_id": row[5],
            "status": row[6],
            "created_at": row[7],
            "updated_at": row[8],
            "ingested_at": row[9],
            "embedding_model_version": row[10],
            "embedding_dim": row[11],
            "sparse_format_version": row[12],
            "tags": self.get_tags_for_doc(doc_id),
        }
        return doc

    def _chunks_table_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        return row is not None

    def list_documents(
        self,
        *,
        tag: Optional[str] = None,
        source_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        status: Optional[str] = "active",
    ) -> List[dict]:
        params: list = []
        where_clauses: list = []

        if status is not None:
            where_clauses.append("d.status = ?")
            params.append(status)
        if source_id is not None:
            where_clauses.append("d.source_id = ?")
            params.append(source_id)
        if dataset_id is not None:
            where_clauses.append("d.dataset_id = ?")
            params.append(dataset_id)
        if tag is not None:
            where_clauses.append(
                "d.doc_id IN ("
                "  SELECT dt.doc_id FROM document_tags dt"
                "  JOIN tags t ON dt.tag_id = t.tag_id"
                "  WHERE t.name = ?"
                ")"
            )
            params.append(tag)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Only join chunks when the table exists (MetadataStore may be used standalone)
        if self._chunks_table_exists():
            chunk_join = "LEFT JOIN chunks c ON d.doc_id = c.doc_id"
            chunk_col = "COUNT(DISTINCT c.chunk_id) AS chunk_count"
        else:
            chunk_join = ""
            chunk_col = "0 AS chunk_count"

        rows = self._conn.execute(
            f"""
            SELECT d.doc_id, d.title, d.path, d.content_hash,
                   d.source_id, d.dataset_id, d.status,
                   d.created_at, d.updated_at, d.ingested_at,
                   d.embedding_model_version, d.embedding_dim,
                   d.sparse_format_version,
                   GROUP_CONCAT(t.name, '|||') AS tag_names,
                   {chunk_col}
            FROM documents d
            LEFT JOIN document_tags dt ON d.doc_id = dt.doc_id
            LEFT JOIN tags t ON dt.tag_id = t.tag_id
            {chunk_join}
            {where_sql}
            GROUP BY d.doc_id
            ORDER BY d.created_at DESC
            """,
            params,
        ).fetchall()

        results = []
        for row in rows:
            raw_tags = row[13]
            tags = sorted(set(raw_tags.split("|||"))) if raw_tags else []
            results.append(
                {
                    "doc_id": row[0],
                    "title": row[1],
                    "path": row[2],
                    "content_hash": row[3],
                    "source_id": row[4],
                    "dataset_id": row[5],
                    "status": row[6],
                    "created_at": row[7],
                    "updated_at": row[8],
                    "ingested_at": row[9],
                    "embedding_model_version": row[10],
                    "embedding_dim": row[11],
                    "sparse_format_version": row[12],
                    "tags": tags,
                    "chunk_count": row[14],
                }
            )
        return results

    def delete_document(self, doc_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
        )
        if self._own_conn:
            self._conn.commit()
        return cursor.rowcount > 0

    def count_documents(self, *, status: Optional[str] = "active") -> int:
        if status is not None:
            return self._conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status = ?", (status,)
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    # ------------------------------------------------------------------
    # Tag operations
    # ------------------------------------------------------------------

    def get_or_create_tag(self, name: str) -> int:
        row = self._conn.execute(
            "SELECT tag_id FROM tags WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return row[0]
        cursor = self._conn.execute(
            "INSERT INTO tags (name) VALUES (?)", (name,)
        )
        if self._own_conn:
            self._conn.commit()
        return cursor.lastrowid

    def assign_tag(self, doc_id: str, tag_name: str) -> None:
        tag_id = self.get_or_create_tag(tag_name)
        self._conn.execute(
            "INSERT OR IGNORE INTO document_tags (doc_id, tag_id) VALUES (?, ?)",
            (doc_id, tag_id),
        )
        if self._own_conn:
            self._conn.commit()

    def unassign_tag(self, doc_id: str, tag_name: str) -> None:
        row = self._conn.execute(
            "SELECT tag_id FROM tags WHERE name = ?", (tag_name,)
        ).fetchone()
        if row is None:
            return
        self._conn.execute(
            "DELETE FROM document_tags WHERE doc_id = ? AND tag_id = ?",
            (doc_id, row[0]),
        )
        if self._own_conn:
            self._conn.commit()

    def get_tags_for_doc(self, doc_id: str) -> List[str]:
        rows = self._conn.execute(
            """
            SELECT t.name FROM tags t
            JOIN document_tags dt ON t.tag_id = dt.tag_id
            WHERE dt.doc_id = ?
            ORDER BY t.name
            """,
            (doc_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def list_tags(self) -> List[dict]:
        rows = self._conn.execute(
            """
            SELECT t.tag_id, t.name, COUNT(dt.doc_id) AS doc_count
            FROM tags t
            LEFT JOIN document_tags dt ON t.tag_id = dt.tag_id
            GROUP BY t.tag_id, t.name
            ORDER BY t.name
            """
        ).fetchall()
        return [{"tag_id": r[0], "name": r[1], "doc_count": r[2]} for r in rows]

    def rename_tag(self, old_name: str, new_name: str) -> bool:
        cursor = self._conn.execute(
            "UPDATE tags SET name = ? WHERE name = ?", (new_name, old_name)
        )
        if self._own_conn:
            self._conn.commit()
        return cursor.rowcount > 0

    def delete_tag(self, name: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM tags WHERE name = ?", (name,)
        )
        if self._own_conn:
            self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Source operations
    # ------------------------------------------------------------------

    def upsert_source(
        self,
        source_id: str,
        name: str,
        *,
        description: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO sources (source_id, name, description, url)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, name, description, url),
        )
        if self._own_conn:
            self._conn.commit()

    def list_sources(self) -> List[dict]:
        rows = self._conn.execute(
            """
            SELECT s.source_id, s.name, s.description, s.url, s.created_at,
                   COUNT(d.doc_id) AS doc_count
            FROM sources s
            LEFT JOIN documents d ON s.source_id = d.source_id
            GROUP BY s.source_id
            ORDER BY s.name
            """
        ).fetchall()
        return [
            {
                "source_id": r[0],
                "name": r[1],
                "description": r[2],
                "url": r[3],
                "created_at": r[4],
                "doc_count": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def upsert_dataset(
        self,
        dataset_id: str,
        name: str,
        *,
        description: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO datasets (dataset_id, name, description)
            VALUES (?, ?, ?)
            """,
            (dataset_id, name, description),
        )
        if self._own_conn:
            self._conn.commit()

    def list_datasets(self) -> List[dict]:
        rows = self._conn.execute(
            """
            SELECT ds.dataset_id, ds.name, ds.description, ds.created_at,
                   COUNT(d.doc_id) AS doc_count
            FROM datasets ds
            LEFT JOIN documents d ON ds.dataset_id = d.dataset_id
            GROUP BY ds.dataset_id
            ORDER BY ds.name
            """
        ).fetchall()
        return [
            {
                "dataset_id": r[0],
                "name": r[1],
                "description": r[2],
                "created_at": r[3],
                "doc_count": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._own_conn and self._conn:
            self._conn.close()
            self._conn = None
