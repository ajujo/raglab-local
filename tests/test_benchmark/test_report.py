"""Tests for rag_lab/benchmark/report.py."""

import json
import pytest

from rag_lab.benchmark.report import format_json, format_markdown, generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_per_query(qid, category, r5, r10, r30, mrr, ndcg, lat):
    return {
        "query_id": qid,
        "query": f"Query {qid}",
        "category": category,
        "language": "en",
        "suite": "official",
        "recall@5": r5,
        "recall@10": r10,
        "recall@30": r30,
        "mrr": mrr,
        "ndcg@10": ndcg,
        "latency_ms": lat,
        "n_results": 30,
        "candidate_pool_size": 90,
        "n_dense": 90, "n_bm25": 5, "n_sparse": 90,
        "dense_coverage": 1.0, "bm25_coverage": 0.05, "sparse_coverage": 1.0,
        "unique_docs@5": 2, "max_chunks_same_doc@5": 3,
        "unique_docs@10": 4, "max_chunks_same_doc@10": 3,
        "top5_doc_ids": [],
    }


def _make_result(variant="full", queries=None):
    if queries is None:
        queries = [
            _make_per_query("q001", "glossary_definition", 1.0, 1.0, 1.0, 1.0, 0.9, 10.0),
            _make_per_query("q002", "technical_standard",  0.5, 1.0, 1.0, 0.5, 0.7, 12.0),
            _make_per_query("q003", "glossary_definition", 0.0, 0.5, 1.0, 0.0, 0.4, 11.0),
        ]

    # Build per_category from queries
    from collections import defaultdict
    by_cat = defaultdict(list)
    for q in queries:
        by_cat[q["category"]].append(q)

    per_category = {}
    for cat, qs in by_cat.items():
        r5s = [q["recall@5"] for q in qs]
        per_category[cat] = {
            "n_queries": len(qs),
            "aggregate": {
                "recall@5": sum(r5s) / len(r5s),
                "recall@10": sum(q["recall@10"] for q in qs) / len(qs),
                "recall@30": sum(q["recall@30"] for q in qs) / len(qs),
                "mrr": sum(q["mrr"] for q in qs) / len(qs),
                "ndcg@10": sum(q["ndcg@10"] for q in qs) / len(qs),
                "p50": 10.0, "p95": 12.0, "p99": 12.5,
                "latency_ms": 11.0,
            },
        }

    return {
        "meta": {
            "git_tag": "v1.7",
            "git_sha": "abc123",
            "generated_at": "2026-05-22",
            "corpus_chunks": 610,
        },
        "config": {"top_k": 50, "rrf_k": 20, "n_queries": len(queries), "variants": [variant]},
        "results": {
            variant: {
                "aggregate": {
                    "recall@5": sum(q["recall@5"] for q in queries) / len(queries),
                    "recall@10": 0.85,
                    "recall@30": 0.95,
                    "mrr": 0.75,
                    "ndcg@10": 0.70,
                    "p50": 10.5, "p95": 12.5, "p99": 13.0,
                    "latency_ms": 11.0,
                },
                "per_category": per_category,
                "per_query": queries,
            }
        },
    }


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_returns_dict_with_expected_keys(self):
        result = _make_result()
        report = generate_report(result, variant="full")
        assert "meta" in report
        assert "overall" in report
        assert "per_category" in report
        assert "weak_queries" in report
        assert "strong_queries" in report

    def test_overall_has_metric_keys(self):
        result = _make_result()
        report = generate_report(result)
        overall = report["overall"]
        assert "recall@5" in overall
        assert "ndcg@10" in overall
        assert "mrr" in overall

    def test_per_category_present_when_categories_set(self):
        result = _make_result()
        report = generate_report(result)
        assert len(report["per_category"]) > 0

    def test_per_category_keys_match_categories(self):
        result = _make_result()
        report = generate_report(result)
        cats = set(report["per_category"].keys())
        assert "glossary_definition" in cats
        assert "technical_standard" in cats

    def test_weak_queries_have_recall_below_05(self):
        result = _make_result()
        report = generate_report(result)
        for q in report["weak_queries"]:
            assert q["recall@5"] < 0.5

    def test_strong_queries_have_recall_1(self):
        result = _make_result()
        report = generate_report(result)
        for q in report["strong_queries"]:
            assert q["recall@5"] == 1.0

    def test_meta_carries_git_tag(self):
        result = _make_result()
        report = generate_report(result)
        assert report["meta"]["git_tag"] == "v1.7"

    def test_raises_on_unknown_variant(self):
        result = _make_result(variant="full")
        with pytest.raises(KeyError):
            generate_report(result, variant="nonexistent")

    def test_n_queries_in_meta(self):
        result = _make_result()
        report = generate_report(result)
        assert report["meta"]["n_queries"] == 3


# ---------------------------------------------------------------------------
# format_markdown
# ---------------------------------------------------------------------------

class TestFormatMarkdown:
    def test_returns_string(self):
        result = _make_result()
        report = generate_report(result)
        md = format_markdown(report)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_contains_overall_metrics_section(self):
        result = _make_result()
        report = generate_report(result)
        md = format_markdown(report)
        assert "## Overall Metrics" in md

    def test_contains_per_category_section(self):
        result = _make_result()
        report = generate_report(result)
        md = format_markdown(report)
        assert "## Per-Category Breakdown" in md

    def test_contains_category_names(self):
        result = _make_result()
        report = generate_report(result)
        md = format_markdown(report)
        assert "glossary_definition" in md
        assert "technical_standard" in md

    def test_contains_weak_queries_section_when_present(self):
        result = _make_result()
        report = generate_report(result)
        md = format_markdown(report)
        assert "Weak Queries" in md

    def test_no_weak_section_when_none(self):
        """When all queries have recall@5 >= 0.5, no Weak Queries section."""
        queries = [
            _make_per_query("q001", "glossary_definition", 1.0, 1.0, 1.0, 1.0, 0.9, 10.0),
            _make_per_query("q002", "technical_standard",  0.5, 1.0, 1.0, 0.5, 0.7, 12.0),
        ]
        result = _make_result(queries=queries)
        report = generate_report(result)
        md = format_markdown(report)
        assert "Weak Queries" not in md

    def test_contains_git_tag(self):
        result = _make_result()
        report = generate_report(result)
        md = format_markdown(report)
        assert "v1.7" in md

    def test_metric_values_present(self):
        result = _make_result()
        report = generate_report(result)
        md = format_markdown(report)
        assert "Recall@5" in md
        assert "nDCG@10" in md
        assert "MRR" in md


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------

class TestFormatJson:
    def test_returns_valid_json(self):
        result = _make_result()
        report = generate_report(result)
        raw = format_json(report)
        parsed = json.loads(raw)
        assert "overall" in parsed

    def test_round_trips(self):
        result = _make_result()
        report = generate_report(result)
        raw = format_json(report)
        parsed = json.loads(raw)
        assert parsed["overall"]["recall@5"] == pytest.approx(report["overall"]["recall@5"])
