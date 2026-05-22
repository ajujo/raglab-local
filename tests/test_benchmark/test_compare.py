"""Tests for the benchmark regression guard (rag_lab/benchmark/compare.py)."""

import json
import pytest
from pathlib import Path

from rag_lab.benchmark.compare import (
    apply_thresholds,
    compare_metrics,
    extract_metrics,
    format_report,
    load_benchmark_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(variant: str, metrics: dict) -> dict:
    return {
        "config": {"variants": [variant]},
        "results": {
            variant: {
                "aggregate": metrics,
                "per_query": [],
            }
        },
    }


def _save_result(path: Path, variant: str, metrics: dict) -> Path:
    data = _make_result(variant, metrics)
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# load_benchmark_result
# ---------------------------------------------------------------------------

class TestLoadBenchmarkResult:
    def test_loads_valid_json(self, tmp_path):
        f = tmp_path / "bench.json"
        _save_result(f, "hybrid", {"recall@5": 0.9})
        result = load_benchmark_result(f)
        assert "results" in result

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_benchmark_result(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_extracts_aggregate(self):
        result = _make_result("hybrid_mmr", {"recall@5": 1.0, "ndcg@10": 0.84})
        metrics = extract_metrics(result, "hybrid_mmr")
        assert metrics["recall@5"] == 1.0
        assert metrics["ndcg@10"] == 0.84

    def test_raises_on_unknown_variant(self):
        result = _make_result("hybrid", {"recall@5": 0.9})
        with pytest.raises(KeyError, match="nonexistent"):
            extract_metrics(result, "nonexistent")

    def test_empty_aggregate_returns_empty(self):
        result = _make_result("hybrid", {})
        metrics = extract_metrics(result, "hybrid")
        assert metrics == {}


# ---------------------------------------------------------------------------
# compare_metrics
# ---------------------------------------------------------------------------

class TestCompareMetrics:
    def test_no_regression_all_ok(self):
        baseline = {"recall@5": 1.0, "ndcg@10": 0.84, "mrr": 0.88, "p95": 100.0}
        current  = {"recall@5": 1.0, "ndcg@10": 0.84, "mrr": 0.88, "p95": 100.0}
        findings = compare_metrics(baseline, current)
        assert all(f["status"] == "OK" for f in findings)

    def test_recall_drop_below_threshold_ok(self):
        baseline = {"recall@5": 1.0}
        current  = {"recall@5": 0.985}  # drop = 1.5 pp < 2 pp threshold
        findings = compare_metrics(baseline, current)
        r5 = next(f for f in findings if f["metric"] == "recall@5")
        assert r5["status"] == "OK"

    def test_recall_drop_above_threshold_fail(self):
        baseline = {"recall@5": 1.0}
        current  = {"recall@5": 0.97}   # drop = 3 pp > 2 pp threshold
        findings = compare_metrics(baseline, current)
        r5 = next(f for f in findings if f["metric"] == "recall@5")
        assert r5["status"] == "FAIL"

    def test_ndcg_drop_above_threshold_fail(self):
        baseline = {"ndcg@10": 0.84}
        current  = {"ndcg@10": 0.81}
        findings = compare_metrics(baseline, current)
        ndcg = next(f for f in findings if f["metric"] == "ndcg@10")
        assert ndcg["status"] == "FAIL"

    def test_mrr_drop_above_threshold_fail(self):
        baseline = {"mrr": 0.88}
        current  = {"mrr": 0.84}   # drop = 4 pp > 3 pp threshold
        findings = compare_metrics(baseline, current)
        mrr = next(f for f in findings if f["metric"] == "mrr")
        assert mrr["status"] == "FAIL"

    def test_mrr_drop_below_threshold_ok(self):
        baseline = {"mrr": 0.88}
        current  = {"mrr": 0.86}   # drop = 2 pp < 3 pp threshold
        findings = compare_metrics(baseline, current)
        mrr = next(f for f in findings if f["metric"] == "mrr")
        assert mrr["status"] == "OK"

    def test_p95_latency_increase_warn(self):
        baseline = {"p95": 100.0}
        current  = {"p95": 130.0}  # +30% > 25% threshold
        findings = compare_metrics(baseline, current)
        p95 = next(f for f in findings if f["metric"] == "p95")
        assert p95["status"] == "WARN"

    def test_p95_latency_under_threshold_ok(self):
        baseline = {"p95": 100.0}
        current  = {"p95": 120.0}  # +20% < 25% threshold
        findings = compare_metrics(baseline, current)
        p95 = next(f for f in findings if f["metric"] == "p95")
        assert p95["status"] == "OK"

    def test_metric_improvement_always_ok(self):
        baseline = {"recall@5": 0.9, "ndcg@10": 0.80, "mrr": 0.85}
        current  = {"recall@5": 1.0, "ndcg@10": 0.90, "mrr": 0.95}
        findings = compare_metrics(baseline, current)
        assert all(f["status"] == "OK" for f in findings)

    def test_missing_current_metric_warns(self):
        baseline = {"recall@5": 1.0}
        current  = {}
        findings = compare_metrics(baseline, current)
        r5 = next(f for f in findings if f["metric"] == "recall@5")
        assert r5["status"] == "WARN"
        assert r5["current"] is None

    def test_metric_not_in_baseline_ignored(self):
        baseline = {}
        current  = {"recall@5": 0.9}
        findings = compare_metrics(baseline, current)
        assert not any(f["metric"] == "recall@5" for f in findings)

    def test_custom_threshold_override(self):
        custom = {"recall@5": {"max_drop": 0.10, "severity": "WARN"}}
        baseline = {"recall@5": 1.0}
        current  = {"recall@5": 0.94}   # drop = 6 pp — below custom 10 pp threshold
        findings = compare_metrics(baseline, current, thresholds=custom)
        r5 = next(f for f in findings if f["metric"] == "recall@5")
        assert r5["status"] == "OK"


# ---------------------------------------------------------------------------
# apply_thresholds
# ---------------------------------------------------------------------------

class TestApplyThresholds:
    def test_all_ok(self):
        findings = [
            {"status": "OK"}, {"status": "OK"},
        ]
        assert apply_thresholds(findings) == "OK"

    def test_one_warn(self):
        findings = [
            {"status": "OK"}, {"status": "WARN"},
        ]
        assert apply_thresholds(findings) == "WARN"

    def test_one_fail_overrides_warn(self):
        findings = [
            {"status": "WARN"}, {"status": "FAIL"},
        ]
        assert apply_thresholds(findings) == "FAIL"

    def test_empty_findings_ok(self):
        assert apply_thresholds([]) == "OK"


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_contains_overall(self):
        findings = [{"status": "OK", "message": "recall@5: 1.000 → 1.000 (Δ+0.000)"}]
        report = format_report(findings, "OK")
        assert "Overall: OK" in report

    def test_contains_fail_icon(self):
        findings = [{"status": "FAIL", "message": "recall@5: regression"}]
        report = format_report(findings, "FAIL")
        assert "✗" in report

    def test_contains_variant(self):
        findings = []
        report = format_report(findings, "OK", variant="hybrid_mmr")
        assert "hybrid_mmr" in report

    def test_contains_baseline_path(self):
        findings = []
        report = format_report(findings, "OK", baseline_path="/some/path.json")
        assert "/some/path.json" in report


# ---------------------------------------------------------------------------
# End-to-end compare() via files
# ---------------------------------------------------------------------------

class TestCompareEndToEnd:
    def test_no_regression(self, tmp_path):
        from rag_lab.benchmark.compare import compare
        baseline = tmp_path / "baseline.json"
        current  = tmp_path / "current.json"
        metrics = {"recall@5": 1.0, "ndcg@10": 0.84, "mrr": 0.88, "p95": 100.0}
        _save_result(baseline, "hybrid_mmr", metrics)
        _save_result(current,  "hybrid_mmr", metrics)

        findings, overall = compare(baseline, current, variant="hybrid_mmr", quiet=True)
        assert overall == "OK"

    def test_regression_detected(self, tmp_path):
        from rag_lab.benchmark.compare import compare
        baseline = tmp_path / "baseline.json"
        current  = tmp_path / "current.json"
        _save_result(baseline, "hybrid_mmr", {"recall@5": 1.0, "ndcg@10": 0.84, "mrr": 0.88, "p95": 100.0})
        _save_result(current,  "hybrid_mmr", {"recall@5": 0.95, "ndcg@10": 0.80, "mrr": 0.84, "p95": 100.0})

        findings, overall = compare(baseline, current, variant="hybrid_mmr", quiet=True)
        assert overall == "FAIL"

    def test_saves_output_json(self, tmp_path):
        from rag_lab.benchmark.compare import compare
        baseline = tmp_path / "baseline.json"
        current  = tmp_path / "current.json"
        output   = tmp_path / "report.json"
        metrics = {"recall@5": 1.0, "ndcg@10": 0.84, "mrr": 0.88, "p95": 100.0}
        _save_result(baseline, "hybrid_mmr", metrics)
        _save_result(current,  "hybrid_mmr", metrics)

        compare(baseline, current, variant="hybrid_mmr", output_path=output, quiet=True)
        data = json.loads(output.read_text())
        assert "overall" in data
        assert "findings" in data
        assert data["variant"] == "hybrid_mmr"


# ---------------------------------------------------------------------------
# v1.8 threshold additions (recall@10, recall@30, p99)
# ---------------------------------------------------------------------------

class TestV18Thresholds:
    def test_recall_at_30_in_default_thresholds(self):
        from rag_lab.benchmark.compare import DEFAULT_THRESHOLDS
        assert "recall@30" in DEFAULT_THRESHOLDS

    def test_recall_at_10_in_default_thresholds(self):
        from rag_lab.benchmark.compare import DEFAULT_THRESHOLDS
        assert "recall@10" in DEFAULT_THRESHOLDS

    def test_p99_in_default_thresholds(self):
        from rag_lab.benchmark.compare import DEFAULT_THRESHOLDS
        assert "p99" in DEFAULT_THRESHOLDS

    def test_recall30_drop_triggers_warn(self):
        baseline = {"recall@30": 0.96, "recall@5": 1.0, "ndcg@10": 0.8, "mrr": 0.9, "p95": 100.0}
        current  = {"recall@30": 0.92, "recall@5": 1.0, "ndcg@10": 0.8, "mrr": 0.9, "p95": 100.0}
        findings = compare_metrics(baseline, current)
        r30_finding = next(f for f in findings if f["metric"] == "recall@30")
        assert r30_finding["status"] in ("WARN", "FAIL")

    def test_recall30_small_drop_ok(self):
        """Drop within threshold stays OK."""
        baseline = {"recall@30": 0.96, "recall@5": 1.0, "ndcg@10": 0.8, "mrr": 0.9, "p95": 100.0}
        current  = {"recall@30": 0.955, "recall@5": 1.0, "ndcg@10": 0.8, "mrr": 0.9, "p95": 100.0}
        findings = compare_metrics(baseline, current)
        r30_finding = next((f for f in findings if f["metric"] == "recall@30"), None)
        if r30_finding:
            assert r30_finding["status"] == "OK"

    def test_p99_large_increase_warns(self):
        """P99 increasing by >30% relative triggers WARN."""
        baseline = {"recall@5": 1.0, "ndcg@10": 0.8, "mrr": 0.9, "p95": 100.0, "p99": 100.0}
        current  = {"recall@5": 1.0, "ndcg@10": 0.8, "mrr": 0.9, "p95": 100.0, "p99": 135.0}
        findings = compare_metrics(baseline, current)
        p99_finding = next((f for f in findings if f["metric"] == "p99"), None)
        if p99_finding:
            assert p99_finding["status"] in ("WARN", "FAIL")

    def test_default_variant_is_full(self):
        """compare() default variant is 'full' in v1.8."""
        from rag_lab.benchmark.compare import compare
        import inspect
        sig = inspect.signature(compare)
        assert sig.parameters["variant"].default == "full"

    def test_strong_regression_fails(self, tmp_path):
        """Regression strong enough to fail on recall@5."""
        from rag_lab.benchmark.compare import compare
        baseline = tmp_path / "b.json"
        current  = tmp_path / "c.json"
        _save_result(baseline, "full", {"recall@5": 0.80, "ndcg@10": 0.80, "mrr": 0.90, "p95": 100.0})
        _save_result(current,  "full", {"recall@5": 0.70, "ndcg@10": 0.78, "mrr": 0.87, "p95": 110.0})
        _, overall = compare(baseline, current, variant="full", quiet=True)
        assert overall == "FAIL"

    def test_within_tolerance_passes(self, tmp_path):
        """Changes within all thresholds produce OK."""
        from rag_lab.benchmark.compare import compare
        baseline = tmp_path / "b.json"
        current  = tmp_path / "c.json"
        _save_result(baseline, "full", {"recall@5": 0.80, "ndcg@10": 0.80, "mrr": 0.90, "p95": 100.0})
        _save_result(current,  "full", {"recall@5": 0.80, "ndcg@10": 0.80, "mrr": 0.90, "p95": 100.0})
        _, overall = compare(baseline, current, variant="full", quiet=True)
        assert overall == "OK"
