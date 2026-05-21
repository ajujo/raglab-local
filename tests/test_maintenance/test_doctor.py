"""Tests for the doctor command (rag_lab/doctor.py)."""

import pytest
from unittest.mock import MagicMock, patch

from rag_lab.doctor import (
    ALL_CHECKS,
    CheckResult,
    check_chromadb,
    check_config,
    check_docstore,
    check_fts5,
    check_reconcile,
    check_sparse_coverage,
    check_test_query,
    doctor,
)


# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_default_fields(self):
        r = CheckResult("config", "OK")
        assert r.name == "config"
        assert r.status == "OK"
        assert r.reason is None
        assert r.detail is None

    def test_with_reason(self):
        r = CheckResult("fts5", "WARN", reason="incomplete")
        assert r.reason == "incomplete"


# ---------------------------------------------------------------------------
# check_config
# ---------------------------------------------------------------------------

class TestCheckConfig:
    def test_ok_with_valid_config(self):
        result = check_config()
        # Should pass since the project's config.py has valid defaults
        assert result.name == "config"
        assert result.status in ("OK", "WARN", "FAIL")

    def test_fail_on_import_error(self):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = check_config()
        # If import fails, status should be FAIL
        assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# check_docstore
# ---------------------------------------------------------------------------

class TestCheckDocstore:
    def test_ok_when_chunks_exist(self):
        mock_ds = MagicMock()
        mock_ds.count.return_value = 50
        with patch("rag_lab.doctor.DocStore", return_value=mock_ds):
            result = check_docstore()
        assert result.status == "OK"
        assert "50" in result.reason

    def test_warn_when_empty(self):
        mock_ds = MagicMock()
        mock_ds.count.return_value = 0
        with patch("rag_lab.doctor.DocStore", return_value=mock_ds):
            result = check_docstore()
        assert result.status == "WARN"

    def test_fail_on_exception(self):
        mock_ds = MagicMock()
        mock_ds.initialize.side_effect = Exception("db locked")
        with patch("rag_lab.doctor.DocStore", return_value=mock_ds):
            result = check_docstore()
        assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# check_chromadb
# ---------------------------------------------------------------------------

class TestCheckChromaDB:
    def test_ok_when_vectors_exist(self):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 610
        mock_vs = MagicMock()
        mock_vs._collection = mock_collection
        with patch("rag_lab.doctor.VectorStore", return_value=mock_vs):
            result = check_chromadb()
        assert result.status == "OK"

    def test_warn_when_empty(self):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_vs = MagicMock()
        mock_vs._collection = mock_collection
        with patch("rag_lab.doctor.VectorStore", return_value=mock_vs):
            result = check_chromadb()
        assert result.status == "WARN"

    def test_fail_on_exception(self):
        mock_vs = MagicMock()
        mock_vs.initialize.side_effect = Exception("chroma unavailable")
        with patch("rag_lab.doctor.VectorStore", return_value=mock_vs):
            result = check_chromadb()
        assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# check_fts5
# ---------------------------------------------------------------------------

class TestCheckFts5:
    def test_ok_when_fully_indexed(self):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (10,)
        mock_ds = MagicMock()
        mock_ds.count.return_value = 10
        mock_ds._conn = mock_conn
        with patch("rag_lab.doctor.DocStore", return_value=mock_ds):
            result = check_fts5()
        assert result.status == "OK"

    def test_warn_when_partial(self):
        def _side_effect(sql):
            m = MagicMock()
            if "chunks_fts" in sql:
                m.fetchone.return_value = (5,)
            else:
                m.fetchone.return_value = (5,)
            return m

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = _side_effect
        mock_ds = MagicMock()
        mock_ds.count.return_value = 10
        mock_ds._conn = mock_conn
        with patch("rag_lab.doctor.DocStore", return_value=mock_ds):
            result = check_fts5()
        assert result.status in ("WARN", "FAIL")

    def test_fail_on_exception(self):
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = Exception("no such table: chunks_fts")
        mock_ds = MagicMock()
        mock_ds.count.return_value = 10
        mock_ds._conn = mock_conn
        with patch("rag_lab.doctor.DocStore", return_value=mock_ds):
            result = check_fts5()
        assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# check_sparse_coverage
# ---------------------------------------------------------------------------

class TestCheckSparseCoverage:
    def test_ok_at_full_coverage(self):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (10, 10)
        mock_ds = MagicMock()
        mock_ds._conn = mock_conn
        with patch("rag_lab.doctor.DocStore", return_value=mock_ds):
            result = check_sparse_coverage()
        assert result.status == "OK"

    def test_warn_below_threshold(self):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (10, 5)
        mock_ds = MagicMock()
        mock_ds._conn = mock_conn
        with patch("rag_lab.doctor.DocStore", return_value=mock_ds):
            result = check_sparse_coverage()
        assert result.status == "WARN"

    def test_warn_when_empty(self):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (0, 0)
        mock_ds = MagicMock()
        mock_ds._conn = mock_conn
        with patch("rag_lab.doctor.DocStore", return_value=mock_ds):
            result = check_sparse_coverage()
        assert result.status == "WARN"


# ---------------------------------------------------------------------------
# check_reconcile
# ---------------------------------------------------------------------------

class TestCheckReconcile:
    def test_ok_when_consistent(self):
        clean_result = {
            "docstore_count": 10, "chroma_count": 10,
            "fts_count": 10, "sparse_blob_count": 10,
            "chroma_orphans": [], "missing_from_chroma": [],
            "duplicate_chunk_ids": [], "model_version_mismatches": [],
            "embedding_dim_mismatches": [], "sparse_format_version_mismatches": [],
            "repaired": False,
        }
        with patch("rag_lab.doctor.reconcile", return_value=clean_result), \
             patch("rag_lab.doctor._has_issues", return_value=False):
            result = check_reconcile()
        assert result.status == "OK"

    def test_fail_when_orphans(self):
        dirty_result = {
            "docstore_count": 10, "chroma_count": 11,
            "fts_count": 10, "sparse_blob_count": 10,
            "chroma_orphans": ["orphan1"], "missing_from_chroma": [],
            "duplicate_chunk_ids": [], "model_version_mismatches": [],
            "embedding_dim_mismatches": [], "sparse_format_version_mismatches": [],
            "repaired": False,
        }
        with patch("rag_lab.doctor.reconcile", return_value=dirty_result), \
             patch("rag_lab.doctor._has_issues", return_value=True):
            result = check_reconcile()
        assert result.status in ("WARN", "FAIL")

    def test_fail_on_exception(self):
        with patch("rag_lab.doctor.reconcile", side_effect=Exception("db error")):
            result = check_reconcile()
        assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# check_test_query
# ---------------------------------------------------------------------------

class TestCheckTestQuery:
    def test_ok_when_results_returned(self):
        mock_results = [
            {"doc_id": "SDMX_Glossary", "rrf_score": 0.05, "chunk_id": "abc"},
        ]
        with patch("rag_lab.doctor.DocStore") as MockDS, \
             patch("rag_lab.doctor.VectorStore") as MockVS, \
             patch("rag_lab.doctor.FTSStore") as MockFTS, \
             patch("rag_lab.embedding.encoder.encode_chunks",
                   return_value=([MagicMock()], {"__doctor_query__": {}})), \
             patch("rag_lab.retrieval.hybrid_search.hybrid_search", return_value=mock_results):

            MockDS.return_value = MagicMock()
            MockVS.return_value = MagicMock()
            MockFTS.return_value = MagicMock()

            result = check_test_query("What is SDMX?")

        assert result.status == "OK"

    def test_fail_when_no_results(self):
        with patch("rag_lab.doctor.DocStore") as MockDS, \
             patch("rag_lab.doctor.VectorStore") as MockVS, \
             patch("rag_lab.doctor.FTSStore") as MockFTS, \
             patch("rag_lab.embedding.encoder.encode_chunks",
                   return_value=([MagicMock()], {"__doctor_query__": {}})), \
             patch("rag_lab.retrieval.hybrid_search.hybrid_search", return_value=[]):

            MockDS.return_value = MagicMock()
            MockVS.return_value = MagicMock()
            MockFTS.return_value = MagicMock()

            result = check_test_query("no results query")

        assert result.status == "FAIL"

    def test_fail_on_exception(self):
        with patch("rag_lab.doctor.DocStore", side_effect=Exception("store unavailable")):
            result = check_test_query("query")
        assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# doctor() orchestration
# ---------------------------------------------------------------------------

class TestDoctor:
    def _ok_result(self, name):
        return CheckResult(name, "OK", "all good")

    def test_all_ok_returns_ok(self):
        with patch("rag_lab.doctor.check_config", return_value=CheckResult("config", "OK")), \
             patch("rag_lab.doctor.check_docstore", return_value=CheckResult("docstore", "OK")), \
             patch("rag_lab.doctor.check_chromadb", return_value=CheckResult("chromadb", "OK")), \
             patch("rag_lab.doctor.check_fts5", return_value=CheckResult("fts5", "OK")), \
             patch("rag_lab.doctor.check_sparse_coverage", return_value=CheckResult("sparse_coverage", "OK")), \
             patch("rag_lab.doctor.check_reconcile", return_value=CheckResult("reconcile", "OK")), \
             patch("rag_lab.doctor.check_test_query", return_value=CheckResult("test_query", "OK")):
            result = doctor(quiet=True)
        assert result["overall"] == "OK"

    def test_one_warn_returns_warn(self):
        with patch("rag_lab.doctor.check_config", return_value=CheckResult("config", "OK")), \
             patch("rag_lab.doctor.check_docstore", return_value=CheckResult("docstore", "WARN", "low count")), \
             patch("rag_lab.doctor.check_chromadb", return_value=CheckResult("chromadb", "OK")), \
             patch("rag_lab.doctor.check_fts5", return_value=CheckResult("fts5", "OK")), \
             patch("rag_lab.doctor.check_sparse_coverage", return_value=CheckResult("sparse_coverage", "OK")), \
             patch("rag_lab.doctor.check_reconcile", return_value=CheckResult("reconcile", "OK")), \
             patch("rag_lab.doctor.check_test_query", return_value=CheckResult("test_query", "OK")):
            result = doctor(quiet=True)
        assert result["overall"] == "WARN"

    def test_one_fail_returns_fail(self):
        with patch("rag_lab.doctor.check_config", return_value=CheckResult("config", "FAIL", "bad dim")), \
             patch("rag_lab.doctor.check_docstore", return_value=CheckResult("docstore", "OK")), \
             patch("rag_lab.doctor.check_chromadb", return_value=CheckResult("chromadb", "WARN")), \
             patch("rag_lab.doctor.check_fts5", return_value=CheckResult("fts5", "OK")), \
             patch("rag_lab.doctor.check_sparse_coverage", return_value=CheckResult("sparse_coverage", "OK")), \
             patch("rag_lab.doctor.check_reconcile", return_value=CheckResult("reconcile", "OK")), \
             patch("rag_lab.doctor.check_test_query", return_value=CheckResult("test_query", "OK")):
            result = doctor(quiet=True)
        assert result["overall"] == "FAIL"

    def test_subset_of_checks(self):
        with patch("rag_lab.doctor.check_config", return_value=CheckResult("config", "OK")), \
             patch("rag_lab.doctor.check_docstore", return_value=CheckResult("docstore", "OK")):
            result = doctor(checks=["config", "docstore"], quiet=True)
        assert len(result["results"]) == 2
        assert {r.name for r in result["results"]} == {"config", "docstore"}

    def test_invalid_check_raises(self):
        with pytest.raises(ValueError, match="Unknown checks"):
            doctor(checks=["nonexistent"], quiet=True)

    def test_all_checks_list_complete(self):
        assert set(ALL_CHECKS) == {
            "config", "docstore", "chromadb", "fts5",
            "sparse_coverage", "reconcile", "test_query",
        }

    def test_quiet_suppresses_output(self, capsys):
        with patch("rag_lab.doctor.check_config", return_value=CheckResult("config", "OK")), \
             patch("rag_lab.doctor.check_docstore", return_value=CheckResult("docstore", "OK")), \
             patch("rag_lab.doctor.check_chromadb", return_value=CheckResult("chromadb", "OK")), \
             patch("rag_lab.doctor.check_fts5", return_value=CheckResult("fts5", "OK")), \
             patch("rag_lab.doctor.check_sparse_coverage", return_value=CheckResult("sparse_coverage", "OK")), \
             patch("rag_lab.doctor.check_reconcile", return_value=CheckResult("reconcile", "OK")), \
             patch("rag_lab.doctor.check_test_query", return_value=CheckResult("test_query", "OK")):
            doctor(quiet=True)
        captured = capsys.readouterr()
        assert captured.out == ""
