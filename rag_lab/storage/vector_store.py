"""ChromaDB wrapper for dense vector storage.

Provides methods to store and retrieve dense embeddings using ChromaDB
with HNSW and cosine similarity.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from rag_lab.config import (
    EMBEDDING_MODEL,
    STORAGE_DIR,
    VECTOR_STORE_PATH,
    VECTOR_HNSW_SPACE,
    VECTOR_HNSW_M,
    VECTOR_HNSW_CONSTRUCTION_EF,
    VECTOR_HNSW_SEARCH_EF,
)

logger = logging.getLogger("rag_lab")

# Keys used in ChromaDB collection metadata for HNSW configuration
_HNSW_META_SPACE = "hnsw:space"
_HNSW_META_M = "hnsw:M"
_HNSW_META_CONSTRUCTION_EF = "hnsw:construction_ef"
_HNSW_META_SEARCH_EF = "hnsw:search_ef"


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

    def _hnsw_creation_metadata(self) -> dict:
        """Return the HNSW metadata dict for collection creation."""
        return {
            _HNSW_META_SPACE: VECTOR_HNSW_SPACE,
            _HNSW_META_M: VECTOR_HNSW_M,
            _HNSW_META_CONSTRUCTION_EF: VECTOR_HNSW_CONSTRUCTION_EF,
            _HNSW_META_SEARCH_EF: VECTOR_HNSW_SEARCH_EF,
        }

    def _hnsw_effective_params(self, collection) -> dict:
        """Return the actual build-time HNSW params for an existing collection.

        Reads from collection.configuration_json (the authoritative source written
        by the hnswlib index at creation time). Falls back to collection.metadata
        for collections created before ChromaDB 1.5 when configuration_json was
        unavailable.

        Returns a dict keyed by the standard hnsw: metadata keys so it can be
        passed directly to _check_hnsw_mismatch.

        Note: collection.metadata may contain stale annotations from past
        modify() calls (e.g. hnsw:search_ef=500) that do NOT reflect the running
        index. configuration_json is the ground truth.
        """
        try:
            cfg = collection.configuration_json
            hnsw = cfg.get("hnsw") or {}
            if hnsw:
                params = {}
                if hnsw.get("space") is not None:
                    params[_HNSW_META_SPACE] = hnsw["space"]
                if hnsw.get("max_neighbors") is not None:
                    params[_HNSW_META_M] = hnsw["max_neighbors"]
                if hnsw.get("ef_construction") is not None:
                    params[_HNSW_META_CONSTRUCTION_EF] = hnsw["ef_construction"]
                if hnsw.get("ef_search") is not None:
                    params[_HNSW_META_SEARCH_EF] = hnsw["ef_search"]
                return params
        except Exception:
            pass
        return collection.metadata or {}

    def _check_hnsw_mismatch(self, existing_meta: dict) -> None:
        """Warn if the existing collection's HNSW params differ from config.

        All HNSW parameters are build-time in ChromaDB 1.x — changing them in
        config.py without rebuilding the collection has no effect on the running
        index. This method logs a clear warning but never modifies or destroys
        the existing collection.
        """
        mismatches = []

        space = existing_meta.get(_HNSW_META_SPACE)
        if space and space != VECTOR_HNSW_SPACE:
            mismatches.append(
                f"hnsw:space existing={space!r} config={VECTOR_HNSW_SPACE!r}"
            )

        m = existing_meta.get(_HNSW_META_M)
        if m is not None and m != VECTOR_HNSW_M:
            mismatches.append(
                f"hnsw:M existing={m} config={VECTOR_HNSW_M}"
            )

        ef_c = existing_meta.get(_HNSW_META_CONSTRUCTION_EF)
        if ef_c is not None and ef_c != VECTOR_HNSW_CONSTRUCTION_EF:
            mismatches.append(
                f"hnsw:construction_ef existing={ef_c} config={VECTOR_HNSW_CONSTRUCTION_EF}"
            )

        ef_s = existing_meta.get(_HNSW_META_SEARCH_EF)
        if ef_s is not None and ef_s != VECTOR_HNSW_SEARCH_EF:
            mismatches.append(
                f"hnsw:search_ef existing={ef_s} config={VECTOR_HNSW_SEARCH_EF}"
            )

        if mismatches:
            logger.warning(
                "HNSW build-time parameter mismatch — collection %r was built with "
                "different parameters. Config changes have no effect on the running "
                "index; a full rebuild is required to apply them.\n"
                "  Mismatches: %s\n"
                "  Rebuild: delete storage/chroma_db and run `python -m rag_lab.cli ingest`.",
                self.collection_name,
                "; ".join(mismatches),
            )

    def initialize(self) -> None:
        """Initialize the ChromaDB connection and collection.

        If the collection already exists, reads its actual build-time HNSW
        parameters from configuration_json and warns if they differ from config.
        Does not modify or destroy the existing collection.

        If the collection does not exist, creates it with the HNSW parameters
        from config.py (VECTOR_HNSW_SPACE, VECTOR_HNSW_M,
        VECTOR_HNSW_CONSTRUCTION_EF, VECTOR_HNSW_SEARCH_EF).
        """
        try:
            import chromadb
            self._client = chromadb.PersistentClient(
                path=str(self.storage_path)
            )

            existing_names = [c.name for c in self._client.list_collections()]

            if self.collection_name in existing_names:
                self._collection = self._client.get_collection(
                    name=self.collection_name
                )
                effective = self._hnsw_effective_params(self._collection)
                self._check_hnsw_mismatch(effective)
            else:
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata=self._hnsw_creation_metadata(),
                )

            logger.info(
                "Initialized ChromaDB collection: %s (%d vectors, M=%d, ef_c=%d, ef_s=%d, space=%s)",
                self.collection_name,
                self._collection.count(),
                VECTOR_HNSW_M,
                VECTOR_HNSW_CONSTRUCTION_EF,
                VECTOR_HNSW_SEARCH_EF,
                VECTOR_HNSW_SPACE,
            )
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

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all vectors for a given doc_id. Returns count deleted."""
        if self._collection is None:
            self.initialize()
        result = self._collection.get(where={"doc_id": {"$eq": doc_id}}, include=[])
        ids = result.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} vectors for doc_id={doc_id!r}")
        return len(ids)

    def delete_all(self) -> None:
        """Delete all vectors from the store."""
        if self._collection is None:
            self.initialize()
        all_ids = self._collection.get(include=[])["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
        logger.info(f"Deleted all vectors from {self.collection_name}")