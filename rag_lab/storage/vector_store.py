"""ChromaDB wrapper for dense vector storage.

Provides methods to store and retrieve dense embeddings using ChromaDB
with HNSW and cosine similarity.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from rag_lab.config import EMBEDDING_MODEL, STORAGE_DIR, VECTOR_STORE_PATH

logger = logging.getLogger("rag_lab")


class VectorStore:
    """ChromaDB wrapper for dense vector storage."""

    def __init__(
        self,
        collection_name: str = "sdmx_rag",
        storage_path: Optional[Path] = None,
    ):
        """Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            storage_path: Path to store ChromaDB data.
        """
        self.collection_name = collection_name
        self.storage_path = storage_path or VECTOR_STORE_PATH
        self._collection = None

    def initialize(self) -> None:
        """Initialize the ChromaDB connection and collection."""
        try:
            import chromadb
            self._client = chromadb.PersistentClient(
                path=str(self.storage_path)
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": 16,
                    "hnsw:construction_ef": 100,
                },
            )
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Install with: pip install chromadb"
            )

    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """Add vectors to the store.

        Args:
            ids: List of unique IDs for each vector.
            embeddings: Numpy array of shape (n, 1024).
            documents: List of original text documents.
            metadatas: Optional list of metadata dicts.
        """
        if self._collection is None:
            self.initialize()

        kwargs = {
            "ids": ids,
            "embeddings": embeddings.tolist(),
            "documents": documents,
        }
        if metadatas:
            kwargs["metadatas"] = metadatas
            
        self._collection.upsert(**kwargs)
        logger.info(f"Added {len(ids)} vectors to {self.collection_name}")

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        doc_ids: Optional[List[str]] = None,
    ) -> dict:
        """Query the vector store.

        Args:
            query_embedding: Single embedding vector.
            top_k: Number of results to return.
            doc_ids: Optional list of doc_ids to filter by.

        Returns:
            Dict with 'ids', 'distances', 'documents', 'metadatas'.
        """
        if self._collection is None:
            self.initialize()

        query_2d = query_embedding.reshape(1, -1)
        count = self._collection.count()
        if count == 0:
            return {
                "ids": [],
                "distances": [],
                "documents": [],
                "metadatas": [],
            }

        # Build where clause for doc_id filtering
        where_clause = None
        if doc_ids:
            where_clause = {"doc_id": {"$in": doc_ids}}

        results = self._collection.query(
            query_embeddings=query_2d.tolist(),
            n_results=min(top_k, count),
            where=where_clause,
        )

        return {
            "ids": results["ids"][0],
            "distances": results["distances"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
        }

    def count(self) -> int:
        """Return the number of vectors in the store."""
        if self._collection is None:
            self.initialize()
        return self._collection.count()

    def delete_all(self) -> None:
        """Delete all vectors from the store."""
        if self._collection is None:
            self.initialize()
        # Delete all items by iterating over all IDs
        all_ids = self._collection.get(include=[])["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
        logger.info(f"Deleted all vectors from {self.collection_name}")