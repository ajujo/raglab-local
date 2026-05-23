"""Minimal tests for scripts/audit_answer_verifier.py.

Tests verify the module structure and dry-run execution; they do NOT call
the LLM or require a running server.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Import helper — load the script as a module without running main()
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "audit_answer_verifier.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_answer_verifier", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Must register in sys.modules before exec so that @dataclass with
    # `from __future__ import annotations` can resolve the module namespace.
    sys.modules["audit_answer_verifier"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit_mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------

class TestModuleStructure:

    def test_script_exists(self):
        assert _SCRIPT_PATH.exists(), f"Script not found: {_SCRIPT_PATH}"

    def test_query_suites_defined(self, audit_mod):
        assert hasattr(audit_mod, "QUERY_SUITES")
        assert isinstance(audit_mod.QUERY_SUITES, dict)

    def test_answer_e2e_suite_present(self, audit_mod):
        assert "answer_e2e" in audit_mod.QUERY_SUITES

    def test_suite_has_ten_queries(self, audit_mod):
        suite = audit_mod.QUERY_SUITES["answer_e2e"]
        assert len(suite) == 10

    def test_each_query_has_required_fields(self, audit_mod):
        required = {"id", "text", "category", "expect_citations", "expect_in_corpus"}
        for q in audit_mod.QUERY_SUITES["answer_e2e"]:
            missing = required - set(q.keys())
            assert not missing, f"Query {q.get('id')} missing fields: {missing}"

    def test_category_distribution(self, audit_mod):
        suite = audit_mod.QUERY_SUITES["answer_e2e"]
        categories = [q["category"] for q in suite]
        assert categories.count("easy_direct") == 3
        assert categories.count("technical_sdmx") == 3
        assert categories.count("spanish") == 2
        assert categories.count("ambiguous") == 1
        assert categories.count("out_of_corpus") == 1

    def test_main_callable(self, audit_mod):
        assert callable(audit_mod.main)

    def test_assess_callable(self, audit_mod):
        assert callable(audit_mod.assess)

    def test_audit_entry_dataclass(self, audit_mod):
        from dataclasses import fields
        field_names = {f.name for f in fields(audit_mod.AuditEntry)}
        assert "verdict" in field_names
        assert "evidence_map" in field_names
        assert "warnings" in field_names


# ---------------------------------------------------------------------------
# assess() logic (no LLM, no stores)
# ---------------------------------------------------------------------------

class TestAssessLogic:
    """Unit-test the verdict logic without any real pipeline calls."""

    def _make_vr(self, audit_mod, n_valid=2, n_total=2, final_score=0.85,
                 hallucinations=False, parse_success=True):
        """Build a minimal VerificationResult-like mock."""
        from unittest.mock import MagicMock
        from rag_lab.verification.verifier import CitationResult, CitationStatus
        from rag_lab.verification.consistency import ConsistencyResult
        from rag_lab.verification.scoring import ScoreResult, ConfidenceLevel

        cr_list = [
            CitationResult(
                citation_text=f"[[{i+1}] Fuente: x | Sección: y | Líneas: 1-2]",
                status=CitationStatus.VALID if i < n_valid else CitationStatus.INVALID,
            )
            for i in range(n_total)
        ]
        consistency = ConsistencyResult(
            has_unsupported_claims=False,
            has_contradictions=False,
            has_hallucinations=hallucinations,
            details="",
            score=0.0 if hallucinations else 1.0,
            parse_success=parse_success,
        )
        confidence = ConfidenceLevel.HIGH if final_score >= 0.75 else (
            ConfidenceLevel.MEDIUM if final_score >= 0.5 else ConfidenceLevel.LOW
        )
        score = ScoreResult(
            citation_score=1.0,
            retrieval_score=1.0,
            consistency_score=1.0,
            coverage_score=1.0,
            final_score=final_score,
            confidence_level=confidence,
        )
        vr = MagicMock()
        vr.citation_results = cr_list
        vr.consistency_result = consistency
        vr.score_result = score
        return vr

    def test_pass_when_all_valid(self, audit_mod):
        q = {"category": "easy_direct", "expect_citations": True, "expect_in_corpus": True}
        vr = self._make_vr(audit_mod, n_valid=3, n_total=3)
        verdict, reason = audit_mod.assess(q, "answer with citations", vr)
        assert verdict == "PASS"

    def test_fail_when_zero_citations_expected(self, audit_mod):
        q = {"category": "easy_direct", "expect_citations": True, "expect_in_corpus": True}
        vr = self._make_vr(audit_mod, n_valid=0, n_total=0)
        verdict, _ = audit_mod.assess(q, "answer without citations", vr)
        assert verdict == "FAIL"

    def test_fail_when_all_invalid(self, audit_mod):
        q = {"category": "easy_direct", "expect_citations": True, "expect_in_corpus": True}
        vr = self._make_vr(audit_mod, n_valid=0, n_total=2)
        verdict, _ = audit_mod.assess(q, "answer", vr)
        assert verdict == "FAIL"

    def test_fail_when_hallucinations(self, audit_mod):
        q = {"category": "easy_direct", "expect_citations": True, "expect_in_corpus": True}
        vr = self._make_vr(audit_mod, n_valid=2, n_total=2,
                           hallucinations=True, final_score=0.2)
        verdict, _ = audit_mod.assess(q, "answer", vr)
        assert verdict == "FAIL"

    def test_warn_when_low_confidence(self, audit_mod):
        q = {"category": "easy_direct", "expect_citations": True, "expect_in_corpus": True}
        vr = self._make_vr(audit_mod, n_valid=1, n_total=1, final_score=0.3)
        verdict, _ = audit_mod.assess(q, "answer", vr)
        assert verdict == "WARN"

    def test_out_of_corpus_pass_when_declared(self, audit_mod):
        q = {"category": "out_of_corpus", "expect_citations": False, "expect_in_corpus": False}
        vr = self._make_vr(audit_mod, n_valid=0, n_total=0)
        verdict, _ = audit_mod.assess(
            q, "No encuentro esta información en los documentos proporcionados.", vr
        )
        assert verdict == "PASS"

    def test_out_of_corpus_warn_when_not_declared(self, audit_mod):
        q = {"category": "out_of_corpus", "expect_citations": False, "expect_in_corpus": False}
        vr = self._make_vr(audit_mod, n_valid=0, n_total=0)
        verdict, _ = audit_mod.assess(q, "Here is an answer without acknowledging limits.", vr)
        assert verdict == "WARN"

    def test_ambiguous_pass_when_has_valid_citation(self, audit_mod):
        q = {"category": "ambiguous", "expect_citations": True, "expect_in_corpus": True}
        vr = self._make_vr(audit_mod, n_valid=2, n_total=2)
        verdict, _ = audit_mod.assess(q, "answer", vr)
        assert verdict == "PASS"


# ---------------------------------------------------------------------------
# Dry-run mode — full pipeline without LLM (embedding + retrieval required)
# ---------------------------------------------------------------------------

class TestDryRunExecution:
    """Run main() in dry-run mode and verify the JSON output structure."""

    def test_dry_run_produces_json(self, audit_mod, tmp_path):
        out = tmp_path / "test_audit.json"
        # dry-run exit code may be 1 (FAILs expected — synthetic citation won't match)
        audit_mod.main(["--dry-run", "--output", str(out)])
        assert out.exists(), "JSON output file not created."
        data = json.loads(out.read_text())
        assert "entries" in data
        assert len(data["entries"]) == 10

    def test_dry_run_entry_has_required_keys(self, audit_mod, tmp_path):
        out = tmp_path / "test_audit.json"
        audit_mod.main(["--dry-run", "--output", str(out)])
        data = json.loads(out.read_text())
        required = {
            "id", "category", "query", "response",
            "citation_results", "warnings", "verification_block",
            "evidence_map", "confidence_score", "confidence_level",
            "verdict", "verdict_reason",
        }
        for entry in data["entries"]:
            missing = required - set(entry.keys())
            assert not missing, f"Entry {entry.get('id')} missing: {missing}"

    def test_dry_run_v1_18_invariants_pass(self, audit_mod, tmp_path):
        """The 4 v1.18 bug-fix invariants must hold even in dry-run mode."""
        out = tmp_path / "test_audit.json"
        audit_mod.main(["--dry-run", "--output", str(out)])
        data = json.loads(out.read_text())

        for entry in data["entries"]:
            # BUG-1: no 0/0 ✓ false positive
            assert "0/0 ✓" not in entry["verification_block"], (
                f"BUG-1 regression in {entry['id']}: 0/0 ✓ still present."
            )
            # BUG-3: no N/A on consistency line
            for line in entry["verification_block"].splitlines():
                if "Consistencia" in line:
                    assert "N/A" not in line, (
                        f"BUG-3 regression in {entry['id']}: N/A on consistency line."
                    )

    def test_dry_run_out_of_corpus_gets_zero_citations(self, audit_mod, tmp_path):
        """The out-of-corpus query should have 0 citations in dry-run
        (synthetic response is about SDMX, but verifier checks against retrieved
        chunks for that query).
        """
        out = tmp_path / "test_audit.json"
        audit_mod.main(["--dry-run", "--output", str(out)])
        data = json.loads(out.read_text())
        o01 = next(e for e in data["entries"] if e["id"] == "o01")
        # All citations are INVALID in dry-run (synthetic citation from SDMX Training
        # doc won't match what was retrieved for "What is the capital of France?")
        # so valid_citations == 0 — the verifier correctly cannot confirm the citation
        for c in o01["citation_results"]:
            assert c["status"] != "VALID", (
                "Dry-run out-of-corpus entry unexpectedly has a VALID citation."
            )
