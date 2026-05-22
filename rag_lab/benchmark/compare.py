"""Benchmark regression guard: compare a current run against a baseline JSON.

Thresholds (default, all absolute percentage-point drops unless noted):
    R@5    drop > 2 pp  → FAIL
    nDCG@10 drop > 2 pp → FAIL
    MRR    drop > 3 pp  → FAIL
    P95    increase > 25% (relative) → WARN

Usage:
    python -m rag_lab.benchmark.compare \\
        --baseline data/benchmark_v1_1_mmr_20260521.json \\
        --current  data/benchmark_latest.json \\
        --variant  hybrid_mmr \\
        --output   data/regression_report.json

Exit codes:
    0 = no regressions (all within thresholds)
    1 = at least one WARN
    2 = at least one FAIL
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "recall@5":  {"max_drop": 0.02, "severity": "FAIL"},
    "recall@10": {"max_drop": 0.03, "severity": "WARN"},
    "recall@30": {"max_drop": 0.03, "severity": "WARN"},
    "ndcg@10":   {"max_drop": 0.02, "severity": "FAIL"},
    "mrr":       {"max_drop": 0.03, "severity": "FAIL"},
    "p95":       {"max_relative_increase": 0.25, "severity": "WARN"},
    "p99":       {"max_relative_increase": 0.30, "severity": "WARN"},
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_benchmark_result(path) -> dict:
    """Load a benchmark JSON file produced by BenchmarkRunner.save()."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(result: dict, variant: str) -> dict:
    """Extract aggregate metrics for one variant from a benchmark result dict.

    Returns a flat dict of metric_name → value.
    """
    variants_data = result.get("results", {})
    if variant not in variants_data:
        available = list(variants_data.keys())
        raise KeyError(
            f"Variant {variant!r} not found in benchmark. Available: {available}"
        )
    agg = variants_data[variant].get("aggregate", {})
    return dict(agg)


def compare_metrics(
    baseline: dict,
    current: dict,
    thresholds: Optional[dict] = None,
) -> List[dict]:
    """Compare current metrics against baseline and apply thresholds.

    Args:
        baseline: Flat dict of metric_name → float (from extract_metrics).
        current:  Flat dict of metric_name → float (from extract_metrics).
        thresholds: Override DEFAULT_THRESHOLDS (partial override OK).

    Returns:
        List of finding dicts, one per metric checked:
        {
            "metric":    str,
            "baseline":  float,
            "current":   float,
            "delta":     float,   # current - baseline
            "status":    "OK" | "WARN" | "FAIL",
            "message":   str,
        }
    """
    active_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    findings = []

    for metric, spec in active_thresholds.items():
        if metric not in baseline:
            continue
        if metric not in current:
            findings.append({
                "metric": metric,
                "baseline": baseline[metric],
                "current": None,
                "delta": None,
                "status": "WARN",
                "message": f"{metric} missing from current run",
            })
            continue

        b_val = float(baseline[metric])
        c_val = float(current[metric])
        delta = c_val - b_val

        status = "OK"
        message = f"{metric}: {b_val:.4f} → {c_val:.4f} (Δ{delta:+.4f})"

        if "max_drop" in spec:
            if delta < -spec["max_drop"]:
                status = spec["severity"]
                message += f"  ← drop {-delta:.4f} > threshold {spec['max_drop']:.4f}"
        elif "max_relative_increase" in spec:
            if b_val > 0 and delta > 0:
                rel_increase = delta / b_val
                if rel_increase > spec["max_relative_increase"]:
                    status = spec["severity"]
                    message += (
                        f"  ← relative increase {rel_increase:.1%} > "
                        f"threshold {spec['max_relative_increase']:.0%}"
                    )

        findings.append({
            "metric": metric,
            "baseline": b_val,
            "current": c_val,
            "delta": delta,
            "status": status,
            "message": message,
        })

    return findings


def apply_thresholds(findings: List[dict]) -> str:
    """Return overall status: 'OK', 'WARN', or 'FAIL'."""
    overall = "OK"
    for f in findings:
        if f["status"] == "FAIL":
            return "FAIL"
        if f["status"] == "WARN":
            overall = "WARN"
    return overall


def format_report(
    findings: List[dict],
    overall: str,
    baseline_path: str = "",
    current_path: str = "",
    variant: str = "",
) -> str:
    """Render a human-readable regression report."""
    sep = "─" * 55
    lines = [
        "",
        sep,
        "Benchmark Regression Report",
        sep,
    ]
    if variant:
        lines.append(f"  Variant  : {variant}")
    if baseline_path:
        lines.append(f"  Baseline : {baseline_path}")
    if current_path:
        lines.append(f"  Current  : {current_path}")
    lines.append("")

    icons = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}
    for f in findings:
        icon = icons.get(f["status"], "?")
        lines.append(f"  {icon} {f['message']}")

    lines.append("")
    overall_icon = icons[overall]
    lines.append(f"  {overall_icon} Overall: {overall}")
    lines.append(f"{sep}\n")
    return "\n".join(lines)


def compare(
    baseline_path,
    current_path,
    variant: str = "full",
    thresholds: Optional[dict] = None,
    output_path=None,
    quiet: bool = False,
) -> Tuple[List[dict], str]:
    """Full comparison pipeline.

    Args:
        baseline_path: Path to baseline JSON.
        current_path:  Path to current JSON.
        variant:       Variant name to compare.
        thresholds:    Optional threshold overrides.
        output_path:   If set, save JSON report here.
        quiet:         Suppress stdout output.

    Returns:
        (findings, overall_status)
    """
    baseline_result = load_benchmark_result(baseline_path)
    current_result = load_benchmark_result(current_path)

    baseline_metrics = extract_metrics(baseline_result, variant)
    current_metrics = extract_metrics(current_result, variant)

    findings = compare_metrics(baseline_metrics, current_metrics, thresholds)
    overall = apply_thresholds(findings)

    if not quiet:
        report = format_report(
            findings, overall,
            baseline_path=str(baseline_path),
            current_path=str(current_path),
            variant=variant,
        )
        print(report)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "variant": variant,
            "baseline_path": str(baseline_path),
            "current_path": str(current_path),
            "overall": overall,
            "findings": findings,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return findings, overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark regression guard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0 = no regressions\n"
            "  1 = at least one WARN\n"
            "  2 = at least one FAIL\n"
        ),
    )
    parser.add_argument("--baseline", required=True, metavar="PATH",
                        help="Baseline benchmark JSON (e.g. data/benchmark_v1_1_mmr_20260521.json)")
    parser.add_argument("--current", required=True, metavar="PATH",
                        help="Current benchmark JSON to evaluate")
    parser.add_argument("--variant", default="full",
                        help="Variant name to compare (default: full)")
    parser.add_argument("--output", metavar="PATH", default=None,
                        help="Save JSON regression report to this path")
    args = parser.parse_args()

    try:
        _, overall = compare(
            baseline_path=args.baseline,
            current_path=args.current,
            variant=args.variant,
            output_path=args.output,
        )
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if overall == "FAIL":
        sys.exit(2)
    elif overall == "WARN":
        sys.exit(1)
    else:
        sys.exit(0)
