"""Document manager for RAG-Lab.

Provides SQLite-based storage for document metadata and tag management.
"""

import hashlib
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rag_lab.config import DATA_DIR, STORAGE_DIR
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")

# Database path
DB_PATH = STORAGE_DIR / "doc_manager.db"


class DocManager:
    """Manages ingested documents with metadata and tags."""

    def __init__(self, db_path: Path = None):
        """Initialize the document manager.

        Args:
            db_path: Path to the SQLite database. Defaults to STORAGE_DIR/doc_manager.db.
        """
        self.db_path = db_path or DB_PATH
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure the database exists and tables are created."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                ingested_at TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                doc_id TEXT NOT NULL,
                tag_name TEXT NOT NULL,
                PRIMARY KEY (doc_id, tag_name),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"DocManager initialized at {self.db_path}")

    def compute_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file.

        Args:
            file_path: Path to the file.

        Returns:
            MD5 hex digest string.
        """
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def add_document(self, file_path: Path, chunk_count: int = 0) -> bool:
        """Add a new document to the manager.

        Args:
            file_path: Path to the document file.
            chunk_count: Number of chunks this document was split into.

        Returns:
            True if the document was added, False if it already exists.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = file_path.stem
        file_hash = self.compute_hash(file_path)
        file_size = file_path.stat().st_size
        ingested_at = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(self.db_path))

        # Check for duplicate by hash
        existing = conn.execute(
            "SELECT doc_id FROM documents WHERE hash = ?", (file_hash,)
        ).fetchone()

        if existing:
            conn.close()
            logger.info(f"Document already exists with same hash: {existing[0]}")
            return False

        # Insert document
        conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, path, hash, size, ingested_at, chunk_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, str(file_path), file_hash, file_size, ingested_at, chunk_count),
        )
        conn.commit()
        conn.close()
        logger.info(f"Added document: {doc_id} (hash: {file_hash[:8]}...)")
        return True

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its tags.

        Args:
            doc_id: The document ID to delete.

        Returns:
            True if the document was found and deleted.
        """
        conn = sqlite3.connect(str(self.db_path))
        existing = conn.execute(
            "SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()

        if not existing:
            conn.close()
            return False

        # Delete chunks from ChromaDB
        self._delete_chunks_from_chroma(doc_id)

        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM tags WHERE doc_id = ?", (doc_id,))
        conn.commit()
        conn.close()
        logger.info(f"Deleted document: {doc_id}")
        return True

    def _delete_chunks_from_chroma(self, doc_id: str) -> None:
        """Delete all chunks for a document from ChromaDB.

        Args:
            doc_id: The document ID to delete chunks for.
        """
        try:
            vector_store = VectorStore()
            vector_store.initialize()

            # Get all chunk IDs for this doc_id
            results = vector_store.query(
                query_embedding=vector_store._collection.get(include=["metadatas"])[0],
                top_k=1000,
                doc_ids=[doc_id],
            )

            chunk_ids = results["ids"]
            if chunk_ids:
                vector_store._collection.delete(ids=chunk_ids)
                logger.info(f"Deleted {len(chunk_ids)} chunks for {doc_id} from ChromaDB")
        except Exception as e:
            logger.warning(f"Failed to delete chunks from ChromaDB: {e}")

    def list_documents(self) -> List[dict]:
        """List all documents with their metadata.

        Returns:
            List of dicts with document info.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT d.doc_id, d.path, d.hash, d.size, d.ingested_at, d.chunk_count,
                   GROUP_CONCAT(t.tag_name, ',') as tags
            FROM documents d
            LEFT JOIN tags t ON d.doc_id = t.doc_id
            GROUP BY d.doc_id
        """).fetchall()
        conn.close()

        documents = []
        for row in rows:
            documents.append({
                "doc_id": row["doc_id"],
                "path": row["path"],
                "hash": row["hash"][:8] + "...",
                "size": row["size"],
                "ingested_at": row["ingested_at"],
                "chunk_count": row["chunk_count"],
                "tags": row["tags"] or "",
            })
        return documents

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get detailed information about a specific document.

        Args:
            doc_id: The document ID.

        Returns:
            Dict with document info or None if not found.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            conn.close()
            return None

        tags = [t["tag_name"] for t in conn.execute(
            "SELECT tag_name FROM tags WHERE doc_id = ?", (doc_id,)
        ).fetchall()]
        conn.close()

        return {
            "doc_id": row["doc_id"],
            "path": row["path"],
            "hash": row["hash"],
            "size": row["size"],
            "ingested_at": row["ingested_at"],
            "chunk_count": row["chunk_count"],
            "tags": tags,
        }

    # --- Tag Management ---

    def assign_tag(self, doc_id: str, tag_name: str) -> bool:
        """Assign a tag to a document.

        Args:
            doc_id: The document ID.
            tag_name: The tag to assign.

        Returns:
            True if the tag was assigned.
        """
        conn = sqlite3.connect(str(self.db_path))
        # Check if document exists
        doc = conn.execute(
            "SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not doc:
            conn.close()
            return False

        conn.execute(
            "INSERT OR IGNORE INTO tags (doc_id, tag_name) VALUES (?, ?)",
            (doc_id, tag_name),
        )
        conn.commit()
        conn.close()
        logger.info(f"Assigned tag '{tag_name}' to document {doc_id}")
        return True

    def remove_tag(self, doc_id: str, tag_name: str) -> bool:
        """Remove a tag from a document.

        Args:
            doc_id: The document ID.
            tag_name: The tag to remove.

        Returns:
            True if the tag was removed.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "DELETE FROM tags WHERE doc_id = ? AND tag_name = ?",
            (doc_id, tag_name),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        if deleted:
            logger.info(f"Removed tag '{tag_name}' from document {doc_id}")
        return deleted

    def get_tags(self, doc_id: str) -> List[str]:
        """Get all tags for a document.

        Args:
            doc_id: The document ID.

        Returns:
            List of tag names.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        tags = [row["tag_name"] for row in conn.execute(
            "SELECT tag_name FROM tags WHERE doc_id = ?", (doc_id,)
        ).fetchall()]
        conn.close()
        return tags

    def list_all_tags(self) -> List[str]:
        """List all unique tags in the system.

        Returns:
            List of unique tag names.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        tags = [row["tag_name"] for row in conn.execute(
            "SELECT DISTINCT tag_name FROM tags ORDER BY tag_name"
        ).fetchall()]
        conn.close()
        return tags

    def search_documents(self, query: str) -> List[dict]:
        """Search documents by name, path, or tags.

        Args:
            query: Search query string.

        Returns:
            List of matching documents.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        pattern = f"%{query}%"
        rows = conn.execute("""
            SELECT DISTINCT d.doc_id, d.path, d.hash, d.size, d.ingested_at, d.chunk_count,
                   GROUP_CONCAT(t.tag_name, ',') as tags
            FROM documents d
            LEFT JOIN tags t ON d.doc_id = t.doc_id
            WHERE d.doc_id LIKE ? OR d.path LIKE ? OR t.tag_name LIKE ?
            GROUP BY d.doc_id
        """, (pattern, pattern, pattern)).fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "doc_id": row["doc_id"],
                "path": row["path"],
                "hash": row["hash"][:8] + "...",
                "size": row["size"],
                "ingested_at": row["ingested_at"],
                "chunk_count": row["chunk_count"],
                "tags": row["tags"] or "",
            })
        return results

    def update_chunk_count(self, doc_id: str, chunk_count: int) -> None:
        """Update the chunk count for a document.

        Args:
            doc_id: The document ID.
            chunk_count: New chunk count.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "UPDATE documents SET chunk_count = ? WHERE doc_id = ?",
            (chunk_count, doc_id),
        )
        conn.commit()
        conn.close()
