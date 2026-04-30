"""Analyze feedback data and display summary statistics.

Usage:
    python -m rag_lab.feedback.analyze_feedback
"""

import json
from collections import Counter
from pathlib import Path

from rag_lab.feedback.feedback_store import load_feedback, FeedbackEntry


def analyze(db_path: str = None) -> None:
    """Display feedback analysis summary."""
    entries = load_feedback(db_path)

    if not entries:
        print("No hay entradas de feedback registradas.")
        return

    total = len(entries)
    useful = sum(1 for e in entries if e.useful)
    not_useful = total - useful

    useful_pct = (useful / total) * 100
    not_useful_pct = 100 - useful_pct

    # Average scores
    useful_scores = [e.final_score for e in entries if e.useful]
    not_useful_scores = [e.final_score for e in entries if not e.useful]

    avg_useful = sum(useful_scores) / len(useful_scores) if useful_scores else 0.0
    avg_not_useful = sum(not_useful_scores) / len(not_useful_scores) if not_useful_scores else 0.0

    # Most frequent chunks in useful responses
    chunk_counter = Counter()
    for entry in entries:
        if entry.useful:
            try:
                chunks = json.loads(entry.chunks_retrieved)
                for chunk_meta in chunks:
                    key = f"{chunk_meta.get('doc_id', '?')} | Líneas {chunk_meta.get('line_start', '?')}-{chunk_meta.get('line_end', '?')}"
                    chunk_counter[key] += 1
            except (json.JSONDecodeError, TypeError):
                pass

    # Build output
    block = "─" * 45
    block += "\nAnálisis de feedback\n"
    block += f"  Total de respuestas evaluadas: {total}\n"
    block += f"  Útiles  : {useful} ({useful_pct:.0f}%)\n"
    block += f"  No útiles: {not_useful} ({not_useful_pct:.0f}%)\n"
    block += f"\n  Score medio en respuestas útiles   : {avg_useful:.2f}\n"
    block += f"  Score medio en respuestas no útiles: {avg_not_useful:.2f}\n"

    if chunk_counter:
        block += "\n  Chunks más frecuentes en respuestas útiles:\n"
        for chunk_key, count in chunk_counter.most_common(5):
            block += f"    • {chunk_key} → {count} veces\n"

    block += "─" * 45
    print(block)


if __name__ == "__main__":
    analyze()
