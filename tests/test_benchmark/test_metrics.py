"""Unit tests for rag_lab.benchmark.metrics.

All tests use synthetic data — no stores, no models, no I/O.
"""

import math
import pytest

from rag_lab.benchmark.metrics import (
    grade_results,
    mrr,
    ndcg_at_k,
    recall_at_k,
    signal_stats,
    latency_percentiles,
    aggregate_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(doc_ids):
    """Build minimal chunk dicts from a list of doc_ids."""
    return [{"chunk_id": f"c{i}", "doc_id": d} for i, d in enumerate(doc_ids)]


QUERY_BASIC = {
    "doc_relevance": {
        "docA": 3,
        "docB": 2,
    }
}

QUERY_GRADED = {
    "doc_relevance": {
        "docA": 3,
        "docB": 1,
        "docC": 2,
    }
}


# ---------------------------------------------------------------------------
# grade_results
# ---------------------------------------------------------------------------

class TestGradeResults:
    def test_first_appearance_gets_grade(self):
        results = _chunks(["docA", "docA", "docB"])
        grades = grade_results(results, QUERY_BASIC)
        assert grades[0] == 3   # docA first time
        assert grades[1] == 0   # docA second time (already seen)
        assert grades[2] == 2   # docB

    def test_unknown_doc_gets_zero(self):
        results = _chunks(["docZ", "docA"])
        grades = grade_results(results, QUERY_BASIC)
        assert grades[0] == 0
        assert grades[1] == 3

    def test_chunk_relevance_overrides_doc(self):
        results = [{"chunk_id": "special", "doc_id": "docA"}]
        query = {
            "doc_relevance": {"docA": 1},
            "chunk_relevance": {"special": 3},
        }
        grades = grade_results(results, query)
        assert grades[0] == 3

    def test_expected_doc_ids_fallback(self):
        """expected_doc_ids with no doc_relevance → grade 3."""
        results = _chunks(["docX"])
        query = {"expected_doc_ids": ["docX"]}
        grades = grade_results(results, query)
        assert grades[0] == 3

    def test_empty_results(self):
        assert grade_results([], QUERY_BASIC) == []


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        # Both relevant docs in top-2
        results = _chunks(["docA", "docB"])
        assert recall_at_k(results, QUERY_BASIC, k=2) == 1.0

    def test_partial_recall(self):
        # Only docA found in top-1, docB not in top-1
        results = _chunks(["docA", "docZ", "docB"])
        r1 = recall_at_k(results, QUERY_BASIC, k=1)
        r3 = recall_at_k(results, QUERY_BASIC, k=3)
        assert r1 == 0.5   # 1 of 2 relevant docs
        assert r3 == 1.0   # 2 of 2 relevant docs

    def test_zero_recall(self):
        results = _chunks(["docZ", "docW"])
        assert recall_at_k(results, QUERY_BASIC, k=5) == 0.0

    def test_no_relevant_docs(self):
        results = _chunks(["docA"])
        assert recall_at_k(results, {}, k=5) == 0.0

    def test_duplicate_docs_count_once(self):
        # docA appears 3 times but counts as 1 found doc
        results = _chunks(["docA", "docA", "docA"])
        assert recall_at_k(results, QUERY_BASIC, k=3) == 0.5  # 1/2 relevant docs


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_result_relevant(self):
        results = _chunks(["docA"])
        assert mrr(results, QUERY_BASIC) == 1.0

    def test_second_result_relevant(self):
        results = _chunks(["docZ", "docA"])
        assert mrr(results, QUERY_BASIC) == pytest.approx(0.5)

    def test_not_found(self):
        results = _chunks(["docZ"])
        assert mrr(results, QUERY_BASIC) == 0.0

    def test_empty_results(self):
        assert mrr([], QUERY_BASIC) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNDCGAtK:
    def test_perfect_ndcg(self):
        # Highest grade first → nDCG = 1.0
        results = _chunks(["docA", "docB"])  # 3 then 2
        score = ndcg_at_k(results, QUERY_BASIC, k=2)
        assert score == pytest.approx(1.0, rel=1e-3)

    def test_reversed_order_lower_ndcg(self):
        # Lower grade first → nDCG < 1.0
        results = _chunks(["docB", "docA"])  # 2 then 3
        score = ndcg_at_k(results, QUERY_BASIC, k=2)
        assert 0 < score < 1.0

    def test_no_relevant_in_results(self):
        results = _chunks(["docZ", "docW"])
        assert ndcg_at_k(results, QUERY_BASIC, k=10) == 0.0

    def test_no_relevant_annotated(self):
        results = _chunks(["docA"])
        assert ndcg_at_k(results, {}, k=5) == 0.0

    def test_ndcg_degrades_with_rank(self):
        # Found at rank 5 vs rank 1 → lower nDCG
        results_early = _chunks(["docA"] + ["docZ"] * 9)
        results_late = _chunks(["docZ"] * 4 + ["docA"] + ["docZ"] * 5)
        early = ndcg_at_k(results_early, {"doc_relevance": {"docA": 3}}, k=10)
        late = ndcg_at_k(results_late, {"doc_relevance": {"docA": 3}}, k=10)
        assert early > late


# ---------------------------------------------------------------------------
# latency_percentiles
# ---------------------------------------------------------------------------

class TestLatencyPercentiles:
    def test_basic(self):
        times = [10.0, 20.0, 30.0, 100.0, 200.0]
        result = latency_percentiles(times)
        assert "p50" in result
        assert "p95" in result
        assert "p99" in result
        assert "mean" in result
        assert result["p50"] <= result["p95"] <= result["p99"]

    def test_single_element(self):
        result = latency_percentiles([42.0])
        assert result["p50"] == 42.0
        assert result["mean"] == 42.0

    def test_empty(self):
        result = latency_percentiles([])
        assert result["p50"] == 0.0
        assert result["mean"] == 0.0


# ---------------------------------------------------------------------------
# signal_stats
# ---------------------------------------------------------------------------

class TestSignalStats:
    def test_all_dense(self):
        results = [
            {"chunk_id": "c1", "doc_id": "d", "in_dense_topk": True, "in_bm25_topk": False, "in_sparse_topk": False},
            {"chunk_id": "c2", "doc_id": "d", "in_dense_topk": True, "in_bm25_topk": False, "in_sparse_topk": False},
        ]
        stats = signal_stats(results)
        assert stats["dense_coverage"] == 1.0
        assert stats["bm25_coverage"] == 0.0
        assert stats["sparse_coverage"] == 0.0

    def test_empty(self):
        stats = signal_stats([])
        assert stats["sparse_coverage"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def test_mean_computation(self):
        per_query = [
            {"recall@5": 1.0, "mrr": 1.0},
            {"recall@5": 0.5, "mrr": 0.5},
        ]
        agg = aggregate_metrics(per_query)
        assert agg["recall@5"] == pytest.approx(0.75)
        assert agg["mrr"] == pytest.approx(0.75)

    def test_empty(self):
        assert aggregate_metrics([]) == {}
