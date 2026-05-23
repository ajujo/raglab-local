"""Tests for VectorStore HNSW configuration (v1.13).

Covers:
- HNSW config values read from config.py
- New collection created with correct HNSW metadata
- Existing collection with matching metadata → no warning
- Existing collection with mismatching M/ef → warning emitted
- Existing collection with mismatching space → warning emitted
- Existing collection with mismatching search_ef → warning emitted
- Existing collection never destroyed on mismatch
- VectorStore._hnsw_creation_metadata() returns correct keys/values
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from rag_lab.config import (
    VECTOR_HNSW_CONSTRUCTION_EF,
    VECTOR_HNSW_M,
    VECTOR_HNSW_SEARCH_EF,
    VECTOR_HNSW_SPACE,
)
from rag_lab.storage.vector_store import (
    VectorStore,
    _HNSW_META_CONSTRUCTION_EF,
    _HNSW_META_M,
    _HNSW_META_SEARCH_EF,
    _HNSW_META_SPACE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> VectorStore:
    return VectorStore(collection_name="test_hnsw", storage_path=tmp_path)


# ---------------------------------------------------------------------------
# Config values
# ---------------------------------------------------------------------------

class TestHnswConfigValues:
    def test_space_is_cosine(self):
        assert VECTOR_HNSW_SPACE == "cosine"

    def test_m_is_positive_integer(self):
        assert isinstance(VECTOR_HNSW_M, int)
        assert VECTOR_HNSW_M > 0

    def test_construction_ef_is_positive_integer(self):
        assert isinstance(VECTOR_HNSW_CONSTRUCTION_EF, int)
        assert VECTOR_HNSW_CONSTRUCTION_EF > 0

    def test_search_ef_is_positive_integer(self):
        assert isinstance(VECTOR_HNSW_SEARCH_EF, int)
        assert VECTOR_HNSW_SEARCH_EF > 0

    def test_search_ef_ge_m(self):
        # Reasonable invariant: ef_search should be >= M for quality guarantees
        assert VECTOR_HNSW_SEARCH_EF >= VECTOR_HNSW_M

    def test_defaults_match_existing_collection(self):
        """Config defaults must match the existing production collection (M=16, ef_c=100)."""
        assert VECTOR_HNSW_M == 16
        assert VECTOR_HNSW_CONSTRUCTION_EF == 100
        assert VECTOR_HNSW_SEARCH_EF == 100


# ---------------------------------------------------------------------------
# VectorStore._hnsw_creation_metadata
# ---------------------------------------------------------------------------

class TestHnswCreationMetadata:
    def test_returns_all_four_keys(self):
        store = VectorStore(collection_name="test", storage_path=Path("/tmp"))
        meta = store._hnsw_creation_metadata()
        assert _HNSW_META_SPACE in meta
        assert _HNSW_META_M in meta
        assert _HNSW_META_CONSTRUCTION_EF in meta
        assert _HNSW_META_SEARCH_EF in meta

    def test_values_match_config(self):
        store = VectorStore(collection_name="test", storage_path=Path("/tmp"))
        meta = store._hnsw_creation_metadata()
        assert meta[_HNSW_META_SPACE] == VECTOR_HNSW_SPACE
        assert meta[_HNSW_META_M] == VECTOR_HNSW_M
        assert meta[_HNSW_META_CONSTRUCTION_EF] == VECTOR_HNSW_CONSTRUCTION_EF
        assert meta[_HNSW_META_SEARCH_EF] == VECTOR_HNSW_SEARCH_EF


# ---------------------------------------------------------------------------
# Mismatch detection
# ---------------------------------------------------------------------------

class TestHnswMismatchDetection:
    def test_no_warning_on_matching_metadata(self, caplog):
        """Exact match → no warning emitted."""
        store = VectorStore(collection_name="test", storage_path=Path("/tmp"))
        matching_meta = {
            _HNSW_META_SPACE: VECTOR_HNSW_SPACE,
            _HNSW_META_M: VECTOR_HNSW_M,
            _HNSW_META_CONSTRUCTION_EF: VECTOR_HNSW_CONSTRUCTION_EF,
            _HNSW_META_SEARCH_EF: VECTOR_HNSW_SEARCH_EF,
        }
        import logging
        with caplog.at_level(logging.WARNING, logger="rag_lab"):
            store._check_hnsw_mismatch(matching_meta)
        assert "mismatch" not in caplog.text.lower()

    def test_no_warning_on_empty_metadata(self, caplog):
        """Empty metadata (legacy collection without HNSW tags) → no warning."""
        store = VectorStore(collection_name="test", storage_path=Path("/tmp"))
        import logging
        with caplog.at_level(logging.WARNING, logger="rag_lab"):
            store._check_hnsw_mismatch({})
        assert "mismatch" not in caplog.text.lower()

    def test_warning_on_m_mismatch(self, caplog):
        store = VectorStore(collection_name="test", storage_path=Path("/tmp"))
        meta = {
            _HNSW_META_SPACE: VECTOR_HNSW_SPACE,
            _HNSW_META_M: VECTOR_HNSW_M + 16,  # different M
            _HNSW_META_CONSTRUCTION_EF: VECTOR_HNSW_CONSTRUCTION_EF,
        }
        import logging
        with caplog.at_level(logging.WARNING, logger="rag_lab"):
            store._check_hnsw_mismatch(meta)
        assert "mismatch" in caplog.text.lower()
        assert "hnsw:M" in caplog.text

    def test_warning_on_construction_ef_mismatch(self, caplog):
        store = VectorStore(collection_name="test", storage_path=Path("/tmp"))
        meta = {_HNSW_META_CONSTRUCTION_EF: VECTOR_HNSW_CONSTRUCTION_EF + 50}
        import logging
        with caplog.at_level(logging.WARNING, logger="rag_lab"):
            store._check_hnsw_mismatch(meta)
        assert "mismatch" in caplog.text.lower()
        assert "construction_ef" in caplog.text

    def test_warning_on_space_mismatch(self, caplog):
        store = VectorStore(collection_name="test", storage_path=Path("/tmp"))
        meta = {_HNSW_META_SPACE: "l2"}
        import logging
        with caplog.at_level(logging.WARNING, logger="rag_lab"):
            store._check_hnsw_mismatch(meta)
        assert "mismatch" in caplog.text.lower()
        assert "space" in caplog.text

    def test_warning_on_search_ef_mismatch(self, caplog):
        store = VectorStore(collection_name="test", storage_path=Path("/tmp"))
        meta = {_HNSW_META_SEARCH_EF: VECTOR_HNSW_SEARCH_EF + 100}
        import logging
        with caplog.at_level(logging.WARNING, logger="rag_lab"):
            store._check_hnsw_mismatch(meta)
        assert "mismatch" in caplog.text.lower()
        assert "search_ef" in caplog.text

    def test_warning_mentions_rebuild(self, caplog):
        """Mismatch warning must mention rebuild."""
        store = VectorStore(collection_name="test", storage_path=Path("/tmp"))
        meta = {_HNSW_META_M: VECTOR_HNSW_M + 8}
        import logging
        with caplog.at_level(logging.WARNING, logger="rag_lab"):
            store._check_hnsw_mismatch(meta)
        assert "rebuild" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Collection lifecycle: create new vs open existing
# ---------------------------------------------------------------------------

class TestVectorStoreHnswLifecycle:
    def test_new_collection_gets_hnsw_metadata(self, tmp_path):
        """A freshly created collection must have the configured HNSW metadata."""
        store = _make_store(tmp_path)
        store.initialize()
        assert store._collection is not None
        meta = store._collection.metadata or {}
        assert meta.get(_HNSW_META_SPACE) == VECTOR_HNSW_SPACE
        assert meta.get(_HNSW_META_M) == VECTOR_HNSW_M
        assert meta.get(_HNSW_META_CONSTRUCTION_EF) == VECTOR_HNSW_CONSTRUCTION_EF

    def test_existing_collection_not_destroyed_on_mismatch(self, tmp_path):
        """When an existing collection has different HNSW params, it is NOT destroyed."""
        import chromadb

        # Create a collection with non-default M
        client = chromadb.PersistentClient(path=str(tmp_path))
        client.create_collection(
            name="test_hnsw",
            metadata={
                _HNSW_META_SPACE: "cosine",
                _HNSW_META_M: 8,          # differs from default 16
                _HNSW_META_CONSTRUCTION_EF: 64,
            },
        )
        # Add some vectors so we can verify count is preserved
        col = client.get_collection("test_hnsw")
        col.add(
            ids=["v1", "v2"],
            embeddings=np.random.rand(2, 1024).tolist(),
            documents=["doc1", "doc2"],
        )
        assert col.count() == 2

        # Now initialize with default config (M=16, different from existing M=8)
        store = _make_store(tmp_path)
        store.initialize()

        # Collection still has 2 vectors — NOT destroyed
        assert store.count() == 2

    def test_existing_collection_metadata_preserved_on_init(self, tmp_path):
        """Initializing on existing collection with mismatched config must not alter its metadata."""
        import chromadb

        original_meta = {
            _HNSW_META_SPACE: "cosine",
            _HNSW_META_M: 8,
            _HNSW_META_CONSTRUCTION_EF: 64,
        }
        client = chromadb.PersistentClient(path=str(tmp_path))
        client.create_collection(name="test_hnsw", metadata=original_meta)

        store = _make_store(tmp_path)
        store.initialize()

        # Reopen and check metadata unchanged
        col = chromadb.PersistentClient(path=str(tmp_path)).get_collection("test_hnsw")
        # M should still be 8, not overwritten to 16
        assert col.metadata.get(_HNSW_META_M) == 8

    def test_reconcile_unaffected_by_hnsw_config(self, tmp_path):
        """HNSW config changes must not break the reconcile logic."""
        # Reconcile counts vectors; HNSW params are transparent to count()
        store = _make_store(tmp_path)
        store.initialize()
        assert store.count() == 0
        # Add and count
        store.add(
            ids=["x"],
            embeddings=np.zeros((1, 1024), dtype=np.float32),
            documents=["doc"],
        )
        assert store.count() == 1

    def test_no_warning_for_stale_metadata_annotation(self, tmp_path, caplog):
        """Stale metadata from a past modify() call must not produce a false mismatch warning.

        Scenario: collection built with default params (matching config), then
        metadata annotated via col.modify({"hnsw:search_ef": <different>}).
        The running index is unchanged. initialize() must read configuration_json
        (actual index params), not metadata — so no warning is emitted.
        """
        import chromadb
        import logging

        client = chromadb.PersistentClient(path=str(tmp_path))
        col = client.create_collection(
            name="test_hnsw",
            metadata={
                _HNSW_META_SPACE: VECTOR_HNSW_SPACE,
                _HNSW_META_M: VECTOR_HNSW_M,
                _HNSW_META_CONSTRUCTION_EF: VECTOR_HNSW_CONSTRUCTION_EF,
                _HNSW_META_SEARCH_EF: VECTOR_HNSW_SEARCH_EF,
            },
        )
        # Simulate the stale annotation that was on the production collection
        col.modify(metadata={_HNSW_META_SEARCH_EF: VECTOR_HNSW_SEARCH_EF + 400})

        store = _make_store(tmp_path)
        with caplog.at_level(logging.WARNING, logger="rag_lab"):
            store.initialize()
        assert "mismatch" not in caplog.text.lower()


# ---------------------------------------------------------------------------
# Profile constant sanity
# ---------------------------------------------------------------------------

class TestHnswProfileConstants:
    def test_all_profiles_present(self):
        from rag_lab.maintenance.hnsw_profiles import HNSW_PROFILES
        assert "current" in HNSW_PROFILES
        assert "fast" in HNSW_PROFILES
        assert "balanced" in HNSW_PROFILES
        assert "recall" in HNSW_PROFILES

    def test_current_profile_matches_config(self):
        from rag_lab.maintenance.hnsw_profiles import HNSW_PROFILES
        p = HNSW_PROFILES["current"]
        assert p["hnsw:M"] == VECTOR_HNSW_M
        assert p["hnsw:construction_ef"] == VECTOR_HNSW_CONSTRUCTION_EF
        assert p["hnsw:search_ef"] == VECTOR_HNSW_SEARCH_EF
        assert p["hnsw:space"] == VECTOR_HNSW_SPACE

    def test_recall_profile_higher_m_than_fast(self):
        from rag_lab.maintenance.hnsw_profiles import HNSW_PROFILES
        assert HNSW_PROFILES["recall"]["hnsw:M"] > HNSW_PROFILES["fast"]["hnsw:M"]

    def test_recall_profile_higher_ef_than_fast(self):
        from rag_lab.maintenance.hnsw_profiles import HNSW_PROFILES
        assert HNSW_PROFILES["recall"]["hnsw:search_ef"] > HNSW_PROFILES["fast"]["hnsw:search_ef"]

    def test_all_profiles_have_cosine_space(self):
        from rag_lab.maintenance.hnsw_profiles import HNSW_PROFILES
        for name, p in HNSW_PROFILES.items():
            assert p["hnsw:space"] == "cosine", f"Profile {name!r} has non-cosine space"
