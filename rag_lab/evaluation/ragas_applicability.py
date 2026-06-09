"""RAGAS applicability metadata for benchmark queries.

Reads the ``ragas.answer_relevancy_applicable``,
``ragas.applicability_reason``, and ``ragas.decision`` fields from
benchmark_queries.yaml and provides helpers for splitting metric reports
into applicable vs. not-applicable subsets.

Designed to run in the ``rag-lab`` environment (no ragas dependency).
``scripts/ragas_eval.py`` imports this module by adding the repo root to
sys.path.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import yaml

VALID_REASONS: frozenset[str] = frozenset(
    {
        "normal_in_corpus",
        "out_of_corpus",
        "ambiguity_test",
        "meta_synthesis",
        "structured_reference_missing_corpus",
        "ragas_evaluator_limitation",
    }
)

VALID_DECISIONS: frozenset[str] = frozenset(
    {
        "keep_as_stress_test",
        "needs_corpus_expansion",
        "needs_query_rewrite",
        "evaluator_limitation",
    }
)

DEFAULT_BENCHMARK_YAML = (
    Path(__file__).parent.parent.parent / "data" / "benchmark_queries.yaml"
)


class ApplicabilityEntry(NamedTuple):
    applicable: bool
    reason: str
    decision: str = ""


def load_applicability_map(
    yaml_path: "Path | str | None" = None,
) -> dict[str, ApplicabilityEntry]:
    """Return ``{query_id: ApplicabilityEntry}`` from benchmark_queries.yaml.

    Queries without a ``ragas`` block default to
    ``applicable=True, reason='normal_in_corpus', decision=''``.

    Raises ``ValueError`` for unrecognised reason strings.
    """
    path = Path(yaml_path) if yaml_path else DEFAULT_BENCHMARK_YAML
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    result: dict[str, ApplicabilityEntry] = {}
    for q in data.get("queries", []):
        qid = q["id"]
        ragas_meta = q.get("ragas") or {}
        applicable = ragas_meta.get("answer_relevancy_applicable", True)
        reason = ragas_meta.get("applicability_reason", "normal_in_corpus")
        decision = ragas_meta.get("decision", "")
        if reason not in VALID_REASONS:
            raise ValueError(
                f"Query {qid}: invalid applicability_reason '{reason}'. "
                f"Valid values: {sorted(VALID_REASONS)}"
            )
        if decision and decision not in VALID_DECISIONS:
            raise ValueError(
                f"Query {qid}: invalid decision '{decision}'. "
                f"Valid values: {sorted(VALID_DECISIONS)}"
            )
        result[qid] = ApplicabilityEntry(
            applicable=bool(applicable), reason=reason, decision=decision
        )

    return result


def split_by_applicability(
    per_query: list[dict],
    applicability_map: dict[str, ApplicabilityEntry],
) -> tuple[list[dict], list[dict]]:
    """Split per-query result rows into ``(applicable, not_applicable)``.

    Rows whose ``query_id`` is absent from *applicability_map* default to
    applicable=True.  Original dicts are not modified — scores are preserved
    as-is in both lists.
    """
    applicable: list[dict] = []
    not_applicable: list[dict] = []
    for row in per_query:
        qid = row.get("query_id", "")
        entry = applicability_map.get(
            qid, ApplicabilityEntry(applicable=True, reason="normal_in_corpus")
        )
        if entry.applicable:
            applicable.append(row)
        else:
            not_applicable.append(row)
    return applicable, not_applicable


def mean_score(rows: list[dict], metric: str) -> float | None:
    """Mean of *metric* across *rows*; ``None`` if list is empty."""
    vals = [r[metric] for r in rows if metric in r and r[metric] is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_applicability_report(
    per_query: list[dict],
    applicability_map: dict[str, ApplicabilityEntry],
    metric_names: list[str],
) -> dict:
    """Return a structured report with all/applicable/not-applicable splits.

    Per-query scores are never modified.  Not-applicable rows remain visible
    in ``not_applicable_queries`` with their original scores.
    """
    applicable_rows, not_applicable_rows = split_by_applicability(
        per_query, applicability_map
    )

    report: dict = {
        "n_all": len(per_query),
        "n_applicable": len(applicable_rows),
        "n_not_applicable": len(not_applicable_rows),
        "scores_all": {},
        "scores_applicable": {},
        "scores_not_applicable": {},
        "not_applicable_queries": [],
    }

    for m in metric_names:
        report["scores_all"][m] = mean_score(per_query, m)
        report["scores_applicable"][m] = mean_score(applicable_rows, m)
        report["scores_not_applicable"][m] = mean_score(not_applicable_rows, m)

    for row in not_applicable_rows:
        qid = row.get("query_id", "")
        entry = applicability_map.get(
            qid, ApplicabilityEntry(applicable=True, reason="normal_in_corpus")
        )
        record: dict = {
            "query_id": qid,
            "question": row.get("question", ""),
            "category": row.get("category", ""),
            "reason": entry.reason,
            "decision": entry.decision,
        }
        for m in metric_names:
            record[m] = row.get(m)
        report["not_applicable_queries"].append(record)

    return report
