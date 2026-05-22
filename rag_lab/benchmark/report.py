"""Benchmark report generation: structured JSON + human-readable Markdown.

Usage
-----
    from rag_lab.benchmark.report import generate_report, format_markdown, format_json

    result = BenchmarkRunner.load(...)
    report = generate_report(result, variant="full")
    print(format_markdown(report))

CLI
---
    python -m rag_lab.benchmark.report data/baselines/v1.7_official.json [--variant full]
"""

import json
import sys
from pathlib import Path
from typing import Optional


_METRIC_KEYS = ["recall@5", "recall@10", "recall@30", "mrr", "ndcg@10",
                "p50", "p95", "p99", "latency_ms"]


def generate_report(result: dict, variant: str = "full") -> dict:
    """Build a structured report dict from a benchmark result.

    Args:
        result: Dict produced by BenchmarkRunner.save() / BenchmarkRunner.run().
        variant: Which variant to extract metrics from.

    Returns:
        Report dict with 'meta', 'overall', 'per_category', 'improving', 'worsening'.
        'improving' / 'worsening' are populated only when a baseline is embedded
        in the result (not used in single-file reports).
    """
    if variant not in result.get("results", {}):
        available = list(result.get("results", {}).keys())
        raise KeyError(f"Variant {variant!r} not found. Available: {available}")

    variant_data = result["results"][variant]
    agg = variant_data.get("aggregate", {})
    per_category = variant_data.get("per_category", {})
    per_query = variant_data.get("per_query", [])
    cfg = result.get("config", {})
    meta = result.get("meta", {})

    overall_metrics = {k: agg.get(k) for k in _METRIC_KEYS}

    # Per-category summary
    category_rows = {}
    for cat, cat_data in sorted(per_category.items()):
        cat_agg = cat_data.get("aggregate", {})
        category_rows[cat] = {
            "n_queries": cat_data.get("n_queries", 0),
            **{k: cat_agg.get(k) for k in _METRIC_KEYS},
        }

    # Identify weak queries (recall@5 < 0.5) and strong queries (recall@5 == 1.0)
    weak_queries = []
    strong_queries = []
    for pq in per_query:
        r5 = pq.get("recall@5", 0)
        entry = {
            "query_id": pq.get("query_id"),
            "query": pq.get("query", ""),
            "category": pq.get("category", "uncategorized"),
            "recall@5": r5,
            "recall@10": pq.get("recall@10", 0),
            "mrr": pq.get("mrr", 0),
            "ndcg@10": pq.get("ndcg@10", 0),
        }
        if r5 < 0.5:
            weak_queries.append(entry)
        elif r5 == 1.0:
            strong_queries.append(entry)

    return {
        "meta": {
            "variant": variant,
            "n_queries": cfg.get("n_queries", len(per_query)),
            "top_k": cfg.get("top_k"),
            "rrf_k": cfg.get("rrf_k"),
            "git_tag": meta.get("git_tag"),
            "git_sha": meta.get("git_sha"),
            "generated_at": meta.get("generated_at"),
            "corpus_chunks": meta.get("corpus_chunks"),
        },
        "overall": overall_metrics,
        "per_category": category_rows,
        "weak_queries": sorted(weak_queries, key=lambda x: x["recall@5"]),
        "strong_queries": strong_queries,
    }


def format_markdown(report: dict) -> str:
    """Render a report dict as a Markdown document."""
    meta = report.get("meta", {})
    overall = report.get("overall", {})
    per_category = report.get("per_category", {})
    weak = report.get("weak_queries", [])
    strong = report.get("strong_queries", [])

    lines = [
        "# Benchmark Report",
        "",
        "## Configuration",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Variant | `{meta.get('variant', '?')}` |",
        f"| Queries | {meta.get('n_queries', '?')} |",
        f"| top_k | {meta.get('top_k', '?')} |",
        f"| rrf_k | {meta.get('rrf_k', '?')} |",
        f"| Corpus chunks | {meta.get('corpus_chunks', '?')} |",
        f"| Git tag | {meta.get('git_tag', '?')} |",
        f"| Generated | {meta.get('generated_at', '?')} |",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    metric_labels = {
        "recall@5": "Recall@5", "recall@10": "Recall@10", "recall@30": "Recall@30",
        "mrr": "MRR", "ndcg@10": "nDCG@10",
        "p50": "P50 (ms)", "p95": "P95 (ms)", "p99": "P99 (ms)",
        "latency_ms": "Mean latency (ms)",
    }
    for key, label in metric_labels.items():
        val = overall.get(key)
        if val is not None:
            fmt = f"{val:.4f}" if key in ("recall@5","recall@10","recall@30","mrr","ndcg@10") else f"{val:.1f}"
            lines.append(f"| {label} | {fmt} |")

    if per_category:
        lines += [
            "",
            "## Per-Category Breakdown",
            "",
            "| Category | N | R@5 | R@10 | MRR | nDCG@10 | P95ms |",
            "|----------|---|-----|------|-----|---------|-------|",
        ]
        for cat, row in sorted(per_category.items()):
            n = row.get("n_queries", 0)
            r5 = row.get("recall@5") or 0
            r10 = row.get("recall@10") or 0
            mrr_val = row.get("mrr") or 0
            ndcg = row.get("ndcg@10") or 0
            p95 = row.get("p95") or 0
            lines.append(
                f"| `{cat}` | {n} | {r5:.3f} | {r10:.3f} | {mrr_val:.3f} | {ndcg:.3f} | {p95:.0f} |"
            )

    if weak:
        lines += [
            "",
            "## Weak Queries (Recall@5 < 0.5)",
            "",
            "| ID | Category | R@5 | R@10 | MRR | Query |",
            "|----|----------|-----|------|-----|-------|",
        ]
        for q in weak:
            lines.append(
                f"| {q['query_id']} | `{q['category']}` | "
                f"{q['recall@5']:.3f} | {q['recall@10']:.3f} | {q['mrr']:.3f} | "
                f"{q['query'][:60]} |"
            )

    lines += ["", f"_Report generated from variant `{meta.get('variant', '?')}`._", ""]
    return "\n".join(lines)


def format_json(report: dict) -> str:
    """Serialize a report dict to a JSON string."""
    return json.dumps(report, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a structured benchmark report from a result JSON file."
    )
    parser.add_argument("result_file", help="Benchmark result JSON (from BenchmarkRunner)")
    parser.add_argument("--variant", default="full",
                        help="Variant to report on (default: full)")
    parser.add_argument("--output", default=None,
                        help="Save Markdown report to this path")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of Markdown")
    args = parser.parse_args()

    path = Path(args.result_file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    try:
        report = generate_report(result, variant=args.variant)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output = format_json(report)
    else:
        output = format_markdown(report)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output)
