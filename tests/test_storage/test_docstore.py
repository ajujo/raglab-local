"""Tests for storage/docstore.py

Tests:
- initialize, add, get_by_id, get_by_ids, count, delete_all
"""

import pytest
from pathlib import Path
import tempfile
import sqlite3

from rag_lab.storage.docstore import DocStore


class TestDocStore:
    @pytest.fixture
    def store(self, tmp_path):
        """Create a DocStore with a temp SQLite DB."""
        db_path = tmp_path / "test_docstore.sqlite"
        return DocStore(db_path=db_path)

    def test_initialization(self, store):
        store.initialize()
        assert store._conn is not None

    def test_add_and_get_by_id(self, store, tmp_path):
        store.initialize()
        chunks = [
            {
                "chunk_id": "1",
                "doc_id": "doc1",
                "text": "Some text",
                "heading_path": "Heading 1",
                "tipo": "texto",
                "posicion_relativa": 0.1,
                "n_tokens": 10,
            }
        ]
        store.add(chunks)
        assert store.count() == 1

        result = store.get_by_id("1")
        assert result is not None
        assert result["text"] == "Some text"

    def test_get_by_id_not_found(self, store):
        store.initialize()
        result = store.get_by_id("nonexistent")
        assert result is None

    def test_get_by_ids(self, store, tmp_path):
        store.initialize()
        chunks = [
            {
                "chunk_id": "1",
                "doc_id": "doc1",
                "text": "Text 1",
                "heading_path": "H1",
                "tipo": "texto",
                "posicion_relativa": 0.1,
                "n_tokens": 10,
            },
            {
                "chunk_id": "2",
                "doc_id": "doc1",
                "text": "Text 2",
                "heading_path": "H2",
                "tipo": "texto",
                "posicion_relativa": 0.2,
                "n_tokens": 15,
            },
        ]
        store.add(chunks)
        results = store.get_by_ids(["1", "2"])
        assert len(results) == 2

    def test_count(self, store, tmp_path):
        store.initialize()
        assert store.count() == 0
        store.add([{"chunk_id": "1", "doc_id": "doc1", "text": "Text", "heading_path": "H1", "tipo": "texto", "posicion_relativa": 0.1, "n_tokens": 10}])
        assert store.count() == 1

    def test_delete_all(self, store, tmp_path):
        store.initialize()
        store.add([{"chunk_id": "1", "doc_id": "doc1", "text": "Text", "heading_path": "H1", "tipo": "texto", "posicion_relativa": 0.1, "n_tokens": 10}])
        store.delete_all()
        assert store.count() == 0

    def test_add_multiple(self, store, tmp_path):
        store.initialize()
        chunks = [
            {
                "chunk_id": str(i),
                "doc_id": "doc1",
                "text": f"Text {i}",
                "heading_path": f"H{i}",
                "tipo": "texto",
                "posicion_relativa": float(i) * 0.1,
                "n_tokens": 10 + i,
            }
            for i in range(5)
        ]
        store.add(chunks)
        assert store.count() == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
