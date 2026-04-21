"""Tests for storage/vector_store.py

Tests:
- VectorStore initialization
- add and query methods
- count and delete_all
"""

import pytest
import tempfile
from pathlib import Path

from rag_lab.storage.vector_store import VectorStore
import numpy as np


class TestVectorStore:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for ChromaDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create a VectorStore instance."""
        return VectorStore(
            collection_name="test_collection",
            storage_path=temp_dir / "chroma_db",
        )

    def test_initialization(self, store):
        store.initialize()
        assert store._collection is not None

    def test_add_and_query(self, store):
        store.initialize()
        embeddings = np.random.rand(3, 1024)
        ids = ["1", "2", "3"]
        docs = ["doc1", "doc2", "doc3"]
        metadatas = [
            {"heading_path": "Section 1"},
            {"heading_path": "Section 2"},
            {"heading_path": "Section 3"},
        ]
        store.add(ids, embeddings, docs, metadatas)
        assert store.count() == 3

        query = np.random.rand(1024)
        results = store.query(query, top_k=2)
        assert len(results["ids"]) == 2
        assert len(results["documents"]) == 2

    def test_add_empty(self, store):
        store.initialize()
        # ChromaDB requires non-empty lists, so we test that an empty list
        # is handled gracefully
        embeddings = np.random.rand(0, 1024)
        ids = []
        docs = []
        metadatas = []
        # ChromaDB raises ValueError for empty lists, so we expect that
        with pytest.raises(ValueError, match="non-empty"):
            store.add(ids, embeddings, docs, metadatas)

    def test_query_empty_store(self, store):
        store.initialize()
        query = np.random.rand(1024)
        # When store is empty, query should return empty results
        results = store.query(query, top_k=1)
        assert len(results["ids"]) == 0

    def test_delete_all(self, store):
        store.initialize()
        embeddings = np.random.rand(2, 1024)
        metadatas = [
            {"heading_path": "Section 1"},
            {"heading_path": "Section 2"},
        ]
        store.add(["1", "2"], embeddings, ["doc1", "doc2"], metadatas)
        assert store.count() == 2
        # Delete all by using a wildcard filter
        store.delete_all()
        assert store.count() == 0

    def test_with_metadatas(self, store):
        store.initialize()
        embeddings = np.random.rand(2, 1024)
        metadatas = [
            {"heading_path": "Section 1"},
            {"heading_path": "Section 2"},
        ]
        store.add(["1", "2"], embeddings, ["doc1", "doc2"], metadatas)
        results = store.query(np.random.rand(1024), top_k=2)
        assert len(results["metadatas"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
