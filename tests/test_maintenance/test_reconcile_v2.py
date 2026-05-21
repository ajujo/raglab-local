"""Tests for the extended reconcile functionality (v1.2)."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_lab.maintenance.reconcile import (
    _has_issues,
    reconcile,
    save_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_result():
    return {
        "docstore_count": 10,
        "chroma_count": 10,
        "fts_count": 10,
        "sparse_blob_count": 10,
        "chroma_orphans": [],
        "missing_from_chroma": [],
        "duplicate_chunk_ids": [],
        "model_version_mismatches": [],
        "embedding_dim_mismatches": [],
        "sparse_format_version_mismatches": [],
        "repaired": False,
    }


# ---------------------------------------------------------------------------
# Unit tests for _has_issues
# ---------------------------------------------------------------------------

class TestHasIssues:
    def test_clean_returns_false(self):
        assert _has_issues(_make_clean_result()) is False

    def test_chroma_orphans_returns_true(self):
        r = _make_clean_result()
        r["chroma_orphans"] = ["orphan1"]
        assert _has_issues(r) is True

    def test_missing_from_chroma_returns_true(self):
        r = _make_clean_result()
        r["missing_from_chroma"] = ["missing1"]
        assert _has_issues(r) is True

    def test_duplicate_chunk_ids_returns_true(self):
        r = _make_clean_result()
        r["duplicate_chunk_ids"] = ["dup1"]
        assert _has_issues(r) is True

    def test_model_version_mismatch_returns_true(self):
        r = _make_clean_result()
        r["model_version_mismatches"] = [{"chunk_id": "x", "stored_version": "old", "config_version": "new"}]
        assert _has_issues(r) is True

    def test_embedding_dim_mismatch_returns_true(self):
        r = _make_clean_result()
        r["embedding_dim_mismatches"] = [{"chunk_id": "x", "stored_dim": 512, "config_dim": 1024}]
        assert _has_issues(r) is True

    def test_sparse_format_version_mismatch_returns_true(self):
        r = _make_clean_result()
        r["sparse_format_version_mismatches"] = [{"chunk_id": "x", "stored_version": 0, "config_version": 1}]
        assert _has_issues(r) is True


# ---------------------------------------------------------------------------
# Unit tests for save_report
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_saves_json(self, tmp_path):
        r = _make_clean_result()
        out = tmp_path / "report.json"
        save_report(r, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["docstore_count"] == 10
        assert data["chroma_orphans"] == []

    def test_creates_parent_dirs(self, tmp_path):
        r = _make_clean_result()
        out = tmp_path / "sub" / "deep" / "report.json"
        save_report(r, out)
        assert out.exists()

    def test_overwrite_existing(self, tmp_path):
        r1 = _make_clean_result()
        r2 = _make_clean_result()
        r2["docstore_count"] = 99
        out = tmp_path / "report.json"
        save_report(r1, out)
        save_report(r2, out)
        data = json.loads(out.read_text())
        assert data["docstore_count"] == 99


# ---------------------------------------------------------------------------
# Integration tests for reconcile() with a real SQLite DocStore
# ---------------------------------------------------------------------------

@pytest.fixture
def ds_with_chroma(tmp_path):
    """Creates a minimal DocStore with 2 chunks and returns (DocStore, mock_chroma_ids)."""
    import numpy as np
    from rag_lab.storage.docstore import DocStore

    db_path = tmp_path / "test.sqlite"
    ds = DocStore(db_path=db_path)
    ds.initialize()

    def _blob(tokens, weights):
        return (
            np.array(tokens, dtype=np.int32).tobytes(),
            np.array(weights, dtype=np.float32).tobytes(),
        )

    t, w = _blob([1, 2], [0.8, 0.5])
    ds.add([
        {"chunk_id": "c1", "doc_id": "doc1", "text": "hello",
         "heading_path": "", "tipo": "texto", "posicion_relativa": 0.1,
         "n_tokens": 2, "line_start": 1, "line_end": 2,
         "sparse_tokens": t, "sparse_weights": w,
         "embedding_model_name": "bge", "embedding_model_version": "2024-09",
         "embedding_dim": 1024, "sparse_format_version": 1},
        {"chunk_id": "c2", "doc_id": "doc1", "text": "world",
         "heading_path": "", "tipo": "texto", "posicion_relativa": 0.5,
         "n_tokens": 2, "line_start": 3, "line_end": 4,
         "sparse_tokens": t, "sparse_weights": w,
         "embedding_model_name": "bge", "embedding_model_version": "2024-09",
         "embedding_dim": 1024, "sparse_format_version": 1},
    ])

    yield ds, db_path

    ds.close()


class TestReconcileQuietMode:
    def test_quiet_suppresses_output(self, ds_with_chroma, capsys):
        ds, db_path = ds_with_chroma

        with patch("rag_lab.maintenance.reconcile.DocStore") as MockDS, \
             patch("rag_lab.maintenance.reconcile.VectorStore") as MockVS:

            mock_ds_inst = MagicMock()
            mock_ds_inst._conn = ds._conn
            MockDS.return_value = mock_ds_inst

            mock_collection = MagicMock()
            mock_collection.get.return_value = {"ids": ["c1", "c2"]}
            mock_vs_inst = MagicMock()
            mock_vs_inst._collection = mock_collection
            MockVS.return_value = mock_vs_inst

            result = reconcile(quiet=True)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_quiet_false_prints_report(self, ds_with_chroma, capsys):
        ds, db_path = ds_with_chroma

        with patch("rag_lab.maintenance.reconcile.DocStore") as MockDS, \
             patch("rag_lab.maintenance.reconcile.VectorStore") as MockVS:

            mock_ds_inst = MagicMock()
            mock_ds_inst._conn = ds._conn
            MockDS.return_value = mock_ds_inst

            mock_collection = MagicMock()
            mock_collection.get.return_value = {"ids": ["c1", "c2"]}
            mock_vs_inst = MagicMock()
            mock_vs_inst._collection = mock_collection
            MockVS.return_value = mock_vs_inst

            reconcile(quiet=False)

        captured = capsys.readouterr()
        assert "Reconcile Report" in captured.out


class TestReconcileOrphans:
    def test_detects_chroma_orphan(self, ds_with_chroma):
        ds, _ = ds_with_chroma

        with patch("rag_lab.maintenance.reconcile.DocStore") as MockDS, \
             patch("rag_lab.maintenance.reconcile.VectorStore") as MockVS:

            mock_ds_inst = MagicMock()
            mock_ds_inst._conn = ds._conn
            MockDS.return_value = mock_ds_inst

            mock_collection = MagicMock()
            # ChromaDB has an extra ID that DocStore doesn't have
            mock_collection.get.return_value = {"ids": ["c1", "c2", "orphan99"]}
            mock_vs_inst = MagicMock()
            mock_vs_inst._collection = mock_collection
            MockVS.return_value = mock_vs_inst

            result = reconcile(quiet=True)

        assert "orphan99" in result["chroma_orphans"]
        assert _has_issues(result) is True

    def test_detects_missing_from_chroma(self, ds_with_chroma):
        ds, _ = ds_with_chroma

        with patch("rag_lab.maintenance.reconcile.DocStore") as MockDS, \
             patch("rag_lab.maintenance.reconcile.VectorStore") as MockVS:

            mock_ds_inst = MagicMock()
            mock_ds_inst._conn = ds._conn
            MockDS.return_value = mock_ds_inst

            mock_collection = MagicMock()
            # ChromaDB only has c1, not c2
            mock_collection.get.return_value = {"ids": ["c1"]}
            mock_vs_inst = MagicMock()
            mock_vs_inst._collection = mock_collection
            MockVS.return_value = mock_vs_inst

            result = reconcile(quiet=True)

        assert "c2" in result["missing_from_chroma"]
        assert _has_issues(result) is True

    def test_clean_state_no_issues(self, ds_with_chroma):
        ds, _ = ds_with_chroma

        with patch("rag_lab.maintenance.reconcile.DocStore") as MockDS, \
             patch("rag_lab.maintenance.reconcile.VectorStore") as MockVS:

            mock_ds_inst = MagicMock()
            mock_ds_inst._conn = ds._conn
            MockDS.return_value = mock_ds_inst

            mock_collection = MagicMock()
            mock_collection.get.return_value = {"ids": ["c1", "c2"]}
            mock_vs_inst = MagicMock()
            mock_vs_inst._collection = mock_collection
            MockVS.return_value = mock_vs_inst

            result = reconcile(quiet=True)

        assert _has_issues(result) is False
        assert result["docstore_count"] == 2
        assert result["chroma_count"] == 2


class TestReconcileReportJson:
    def test_report_json_written(self, ds_with_chroma, tmp_path):
        ds, _ = ds_with_chroma
        out = tmp_path / "report.json"

        with patch("rag_lab.maintenance.reconcile.DocStore") as MockDS, \
             patch("rag_lab.maintenance.reconcile.VectorStore") as MockVS:

            mock_ds_inst = MagicMock()
            mock_ds_inst._conn = ds._conn
            MockDS.return_value = mock_ds_inst

            mock_collection = MagicMock()
            mock_collection.get.return_value = {"ids": ["c1", "c2"]}
            mock_vs_inst = MagicMock()
            mock_vs_inst._collection = mock_collection
            MockVS.return_value = mock_vs_inst

            result = reconcile(quiet=True)
            save_report(result, out)

        data = json.loads(out.read_text())
        assert "docstore_count" in data
        assert "chroma_orphans" in data
        assert "model_version_mismatches" in data
        assert "embedding_dim_mismatches" in data
        assert "sparse_format_version_mismatches" in data
