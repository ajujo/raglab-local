"""Performance report generation for RAG-Lab.

Formats phase timing data into a human-readable table with visual bars.
"""

import json
import os
from datetime import datetime
from typing import Dict


BAR_WIDTH = 20


def _score_bar(score: float, width: int = BAR_WIDTH) -> str:
    """Generate a visual bar for a score between 0 and 1."""
    filled = round(score * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def generate_report(durations: Dict[str, float]) -> str:
    """Generate a formatted performance report from phase durations.

    Args:
        durations: Dictionary mapping phase names to durations in seconds.

    Returns:
        Formatted report string.
    """
    if not durations:
        return "No hay métricas de rendimiento registradas."

    # Find max duration for scaling bars
    max_duration = max(durations.values())

    block = "─" * 45
    block += "\nMétricas de rendimiento\n"

    for phase, duration in durations.items():
        normalized = duration / max_duration if max_duration > 0 else 0
        bar = _score_bar(normalized)
        block += f"  {phase:<20} : {duration:.2f}s  {bar}\n"

    block += "  " + "─" * 40
    total = sum(durations.values())
    block += f"\n  {'Total':<20} : {total:.2f}s\n"
    block += "─" * 45

    return block


def save_report_json(durations: Dict[str, float], path: str = "performance/metrics.json") -> None:
    """Save performance metrics to a JSON file for historical analysis."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Load existing data
    data = []
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)

    # Append new entry
    data.append({
        "timestamp": datetime.now().isoformat(),
        "phases": durations,
        "total": sum(durations.values()),
    })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def analyze_percentiles(path: str = "performance/metrics.json") -> Dict[str, float]:
    """Analyze historical metrics and calculate percentiles.

    Args:
        path: Path to the metrics JSON file.

    Returns:
        Dictionary with min, p50, p95, p99, max for total duration.
    """
    if not os.path.exists(path):
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    if not data:
        return {}

    totals = [entry["total"] for entry in data]
    totals.sort()
    n = len(totals)

    def percentile(pct):
        idx = int(pct / 100 * (n - 1))
        return totals[idx]

    return {
        "min": totals[0],
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": totals[-1],
        "count": n,
    }
