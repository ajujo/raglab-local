"""Tests for storage/sparse_store.py

Tests:
- load, save, add, query methods
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from rag_lab.storage.sparse_store import SparseStore


class TestSparseStore:
    @pytest.fixture
    def store(self, tmp_path):
        return SparseStore(storage_path=tmp_path / "nonexistent.json")

    def test_load_existing(self, store, tmp_path):
        # Create a test file
        test_file = tmp_path / "sparse_index.json"
        test_file.write_text(json.dumps({"id1": {"sparse": {"a": 1.0}}}))
        store = SparseStore(storage_path=test_file)
        store.load()
        assert store.count() == 1

    def test_load_nonexistent(self, store):
        store.load()
        assert store.count() == 0

    def test_save_and_load(self, store, tmp_path):
        test_file = tmp_path / "sparse_index.json"
        store.storage_path = test_file
        store._data = {"id1": {"sparse": {"a": 1.0}}, "id2": {"sparse": {"b": 2.0}}}
        store.save()
        assert test_file.exists()

        store2 = SparseStore(storage_path=test_file)
        store2.load()
        assert store2.count() == 2

    def test_add(self, store):
        ids = ["id1", "id2"]
        sparse_vectors = [
            {"sparse": {"a": 1.0}},
            {"sparse": {"b": 2.0}},
        ]
        store.add(ids, sparse_vectors)
        assert store.count() == 2

    def test_query(self, store):
        store._data = {
            "id1": {"sparse": {"a": 1.0, "b": 0.5}},
            "id2": {"sparse": {"a": 0.5, "b": 2.0}}, # Higher score
        }
        results = store.query({"a": 1.0, "b": 1.0}, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "id2"

    def test_query_empty(self, store):
        results = store.query({"a": 1.0}, top_k=10)
        assert len(results) == 0

    def test_compute_similarity(self, store):
        score = store._compute_similarity({"a": 1.0}, {"a": 2.0, "b": 3.0})
        assert score == 2.0

    def test_compute_similarity_no_overlap(self, store):
        score = store._compute_similarity({"a": 1.0}, {"b": 2.0})
        assert score == 0.0

    def test_compute_similarity_partial_overlap(self, store):
        score = store._compute_similarity({"a": 1.0, "b": 0.5}, {"a": 2.0, "c": 3.0})
        assert score == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
