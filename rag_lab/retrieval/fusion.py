"""Reciprocal Rank Fusion across three rankings.

Combines dense (cosine), BM25, and sparse (BGE-M3 lexical) rankings into a
single fused score.

Primary function: weighted_rrf — per-signal weights with dense_w fixed at 1.0
as reference so bm25_w and sparse_w express relative contribution.

    score(d) = dense_w/(k + rank_dense(d))
             + bm25_w/(k + rank_bm25(d))
             + sparse_w/(k + rank_sparse(d))

Architecture note: BGE-M3 sparse acts as a SECONDARY refinement signal.
At sparse_w=1.0 it over-represents large high-density documents (e.g. a
197-chunk user guide monopolises SDMX terminology results). Calibrated default
sparse_w=0.25 eliminates this bias while preserving lexical coverage.

Result shape per chunk (five-score + three rank fields):
    {
        "id":             str,
        "rrf_score":      float,        # fused rank score
        "dense_score":    float,        # RRF contribution from dense ranking
        "bm25_score":     float,        # raw BM25 score from FTS5 (not RRF unit)
        "sparse_score":   float,        # raw dot-product score (not RRF unit)
        "in_dense_topk":  bool,
        "in_bm25_topk":   bool,
        "in_sparse_topk": bool,
        "dense_rank":     Optional[int],  # 1-based rank in dense input list, or None
        "bm25_rank":      Optional[int],  # 1-based rank in bm25 input list, or None
        "sparse_rank":    Optional[int],  # 1-based rank in sparse input list, or None
    }
"""

import logging
from typing import List, Optional

logger = logging.getLogger("rag_lab")


def weighted_rrf(
    dense_ids: List[str],
    bm25_ranking: List[dict],
    sparse_ranking: List[dict],
    dense_w: float = 1.0,
    bm25_w: float = 1.0,
    sparse_w: float = 1.0,
    k: int = 60,
) -> List[dict]:
    """Weighted three-way Reciprocal Rank Fusion.

    Args:
        dense_ids: Chunk IDs ordered by dense similarity (best first).
        bm25_ranking: [{id, bm25_score}, ...] ordered by BM25 score (best first).
        sparse_ranking: [{id, sparse_score}, ...] ordered by sparse score (best first).
        dense_w: Weight for dense RRF contribution (reference = 1.0).
        bm25_w: Weight for BM25 RRF contribution.
        sparse_w: Weight for sparse RRF contribution (default 0.25 = secondary signal).
        k: RRF smoothing constant. Lower k → more discriminative ranking.

    Returns:
        List of result dicts sorted by rrf_score descending.
    """
    rrf_scores: dict = {}
    meta: dict = {}

    # Dense contribution — record 1-based rank for explain mode
    for rank, chunk_id in enumerate(dense_ids):
        contrib = dense_w / (k + rank + 1)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contrib
        meta.setdefault(chunk_id, {})["in_dense_topk"] = True
        meta[chunk_id]["dense_rrf_contrib"] = contrib
        meta[chunk_id]["dense_rank"] = rank + 1

    # BM25 contribution — record 1-based rank for explain mode
    for rank, item in enumerate(bm25_ranking):
        cid = item["id"]
        contrib = bm25_w / (k + rank + 1)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contrib
        meta.setdefault(cid, {})
        meta[cid]["in_bm25_topk"] = True
        meta[cid]["bm25_score"] = item.get("bm25_score", 0.0)
        meta[cid]["bm25_rank"] = rank + 1

    # Sparse contribution — record 1-based rank for explain mode
    for rank, item in enumerate(sparse_ranking):
        cid = item["id"]
        contrib = sparse_w / (k + rank + 1)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contrib
        meta.setdefault(cid, {})
        meta[cid]["in_sparse_topk"] = True
        meta[cid]["sparse_score"] = item.get("sparse_score", 0.0)
        meta[cid]["sparse_rank"] = rank + 1

    # Build result list — include rank fields (None when not in that signal's list)
    results = []
    for cid, rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        m = meta.get(cid, {})
        results.append({
            "id": cid,
            "rrf_score": rrf,
            "dense_score": m.get("dense_rrf_contrib", 0.0),
            "bm25_score": m.get("bm25_score", 0.0),
            "sparse_score": m.get("sparse_score", 0.0),
            "in_dense_topk": m.get("in_dense_topk", False),
            "in_bm25_topk": m.get("in_bm25_topk", False),
            "in_sparse_topk": m.get("in_sparse_topk", False),
            "dense_rank": m.get("dense_rank", None),
            "bm25_rank": m.get("bm25_rank", None),
            "sparse_rank": m.get("sparse_rank", None),
        })

    logger.debug(
        f"weighted_rrf fused {len(dense_ids)} dense + {len(bm25_ranking)} BM25 + "
        f"{len(sparse_ranking)} sparse (w={dense_w}/{bm25_w}/{sparse_w}, k={k}) "
        f"→ {len(results)} unique candidates"
    )
    return results


def rrf_three(
    dense_ids: List[str],
    bm25_ranking: List[dict],
    sparse_ranking: List[dict],
    k: int = 60,
) -> List[dict]:
    """Backward-compatible wrapper around weighted_rrf with equal weights."""
    return weighted_rrf(dense_ids, bm25_ranking, sparse_ranking, k=k)
