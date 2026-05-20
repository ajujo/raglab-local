"""Retrieval evaluation metrics: recall@k, MRR, nDCG@k, latency percentiles.

All functions accept:
  results   — list of chunk dicts (must have 'chunk_id' and 'doc_id' keys)
  query_item — dict from the queries YAML:
                 doc_relevance:  {doc_id: grade(0-3)}
                 chunk_relevance: {chunk_id: grade(0-3)}  (optional override)

Relevance resolution order per result:
  1. chunk_relevance[chunk_id]  (most specific)
  2. doc_relevance[doc_id]
  3. 0  (not annotated → not relevant)
"""

import math
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_grade_maps(query_item: dict) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return (doc_relevance, chunk_relevance) grade dicts."""
    doc_rel: Dict[str, int] = dict(query_item.get("doc_relevance", {}))
    # Fall back to expected_doc_ids with grade 3 if no explicit grades
    if not doc_rel:
        for d in query_item.get("expected_doc_ids", []):
            doc_rel[d] = 3
    chunk_rel: Dict[str, int] = dict(query_item.get("chunk_relevance", {}))
    return doc_rel, chunk_rel


def _result_id(r: dict) -> str:
    return r.get("chunk_id") or r.get("id", "")


def grade_results(results: List[dict], query_item: dict) -> List[int]:
    """Assign relevance grade (0-3) to each result.

    A doc contributes its grade only at its FIRST appearance (subsequent
    chunks from the same doc score 0) so that nDCG rewards diversity.
    chunk_relevance overrides doc_relevance for specific chunk IDs.
    """
    doc_rel, chunk_rel = _build_grade_maps(query_item)
    seen_docs: set = set()
    grades: List[int] = []

    for r in results:
        cid = _result_id(r)
        did = r.get("doc_id", "")

        if cid in chunk_rel:
            grades.append(chunk_rel[cid])
            seen_docs.add(did)
        elif did in doc_rel and did not in seen_docs:
            grades.append(doc_rel[did])
            seen_docs.add(did)
        else:
            grades.append(0)

    return grades


# ---------------------------------------------------------------------------
# IR metrics
# ---------------------------------------------------------------------------

def recall_at_k(results: List[dict], query_item: dict, k: int) -> float:
    """Doc-level recall@k: fraction of annotated relevant docs with ≥1 chunk in top-k.

    A doc is "found" if any chunk from it appears in the first k results.
    """
    doc_rel, chunk_rel = _build_grade_maps(query_item)
    relevant_docs = {d for d, g in doc_rel.items() if g > 0}
    if not relevant_docs:
        return 0.0

    found: set = set()
    for r in results[:k]:
        did = r.get("doc_id", "")
        cid = _result_id(r)
        # A chunk counts toward its doc if the doc is relevant
        # (chunk_rel can override grade but the doc is still "relevant")
        if did in relevant_docs or cid in chunk_rel:
            found.add(did if did in relevant_docs else cid)

    # Normalise against relevant docs (not chunks)
    denom = len(relevant_docs) + len({c for c in chunk_rel if chunk_rel[c] > 0
                                       and results and
                                       c not in {_result_id(r) for r in results}})
    denom = len(relevant_docs)  # keep denominator at doc level
    return len(found.intersection(relevant_docs)) / denom


def mrr(results: List[dict], query_item: dict) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result (0 if not found)."""
    doc_rel, chunk_rel = _build_grade_maps(query_item)
    for i, r in enumerate(results):
        cid = _result_id(r)
        did = r.get("doc_id", "")
        grade = chunk_rel.get(cid, doc_rel.get(did, 0))
        if grade > 0:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(results: List[dict], query_item: dict, k: int = 10) -> float:
    """nDCG@k with graded relevance.

    Each annotated doc contributes at most once (first appearance).
    IDCG uses the annotated grades sorted descending.
    """
    grades = grade_results(results, query_item)[:k]
    # Pad to k
    grades = grades + [0] * (k - len(grades))

    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(grades))

    # Ideal: all annotated grades sorted descending
    doc_rel, chunk_rel = _build_grade_maps(query_item)
    all_annotated = sorted(
        list(chunk_rel.values()) + [g for d, g in doc_rel.items()],
        reverse=True,
    )[:k]
    all_annotated = all_annotated + [0] * (k - len(all_annotated))

    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(all_annotated))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------

def latency_percentiles(
    times_ms: Sequence[float],
    percentiles: Tuple[int, ...] = (50, 95, 99),
) -> Dict[str, float]:
    """Return P50/P95/P99 (and mean) over a list of latencies in ms."""
    if not times_ms:
        return {f"p{p}": 0.0 for p in percentiles} | {"mean": 0.0}

    import statistics

    sorted_t = sorted(times_ms)
    n = len(sorted_t)
    result: Dict[str, float] = {}
    for p in percentiles:
        idx = max(0, int(math.ceil(n * p / 100)) - 1)
        result[f"p{p}"] = round(sorted_t[idx], 2)
    result["mean"] = round(statistics.mean(times_ms), 2)
    return result


# ---------------------------------------------------------------------------
# Signal / coverage stats
# ---------------------------------------------------------------------------

def signal_stats(results: List[dict]) -> Dict[str, float]:
    """Fraction of results carrying each retrieval signal."""
    if not results:
        return {"sparse_coverage": 0.0, "dense_coverage": 0.0, "bm25_coverage": 0.0}
    n = len(results)
    return {
        "dense_coverage": sum(1 for r in results if r.get("in_dense_topk")) / n,
        "bm25_coverage": sum(1 for r in results if r.get("in_bm25_topk")) / n,
        "sparse_coverage": sum(1 for r in results if r.get("in_sparse_topk")) / n,
    }


# ---------------------------------------------------------------------------
# Aggregate over multiple queries
# ---------------------------------------------------------------------------

def aggregate_metrics(per_query: List[dict]) -> Dict[str, float]:
    """Mean all numeric per-query metrics."""
    if not per_query:
        return {}

    keys = [k for k in per_query[0] if isinstance(per_query[0][k], (int, float))]
    import statistics
    agg: Dict[str, float] = {}
    for k in keys:
        vals = [q[k] for q in per_query if isinstance(q.get(k), (int, float))]
        agg[k] = round(statistics.mean(vals), 4) if vals else 0.0
    return agg
