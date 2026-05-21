"""Two-stage hybrid search: dense + BM25 → candidates → sparse rescore → RRF.

Stage 1 (candidate generation):
  - Dense: ChromaDB HNSW cosine similarity
  - BM25:  FTS5 full-text search

Stage 2 (sparse rescore):
  - BGE-M3 sparse dot-product on the candidate pool only (O(|C|), not O(N))

Stage 3 (fusion):
  - Weighted three-way Reciprocal Rank Fusion → rrf_score
  - Calibrated default: sparse_w=0.25 (secondary signal, avoids large-doc bias)

Result shape per chunk (five-score):
  dense_score, bm25_score, sparse_score, rrf_score, rerank_score (added by reranker)
  in_dense_topk, in_bm25_topk, in_sparse_topk (provenance flags)
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from rag_lab.config import (
    BM25_RRF_WEIGHT,
    DENSE_RRF_WEIGHT,
    DOC_CAP_ENABLED,
    DOC_CAP_N,
    MMR_ENABLED,
    MMR_LAMBDA,
    RRF_K,
    RETRIEVAL_TOP_K,
    SPARSE_COVERAGE_THRESHOLD,
    SPARSE_RRF_WEIGHT,
)
from rag_lab.retrieval.diversity import apply_document_cap, apply_mmr
from rag_lab.retrieval.fusion import weighted_rrf
from rag_lab.retrieval.sparse_scorer import load_sparse_for_chunks, rank_candidates_by_sparse
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")

# Multiplier over top_k used to build the candidate pool before fusion
_CANDIDATE_MULTIPLIER = 3


def _get_sparse_coverage(conn) -> float:
    """Return fraction of chunks that have sparse BLOBs (0.0–1.0)."""
    try:
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN sparse_tokens IS NOT NULL THEN 1 ELSE 0 END) FROM chunks"
        ).fetchone()
        total, with_sparse = row[0], row[1] or 0
        return with_sparse / total if total else 0.0
    except Exception:
        return 0.0


def hybrid_search(
    query: str,
    vector_store: VectorStore,
    doc_store: DocStore,
    fts_store: FTSStore,
    query_dense: np.ndarray = None,
    query_sparse: Dict[int, float] = None,
    top_k: int = None,
    rrf_k: int = None,
    doc_ids: Optional[List[str]] = None,
    dense_weight: float = None,
    bm25_weight: float = None,
    sparse_weight: float = None,
    diversity_mode: Optional[str] = None,
    doc_cap: Optional[int] = None,
    mmr_lambda: Optional[float] = None,
    _return_stats: bool = False,
) -> List[dict]:
    """Perform two-stage hybrid search with weighted RRF fusion.

    Args:
        query: User query text (used for BM25).
        vector_store: ChromaDB wrapper for dense vectors.
        doc_store: SQLite docstore (source of truth + sparse BLOBs).
        fts_store: FTS5 wrapper for BM25 search.
        query_dense: Dense embedding for the query.
        query_sparse: Sparse embedding {token_id: weight} for the query.
        top_k: Number of final results after fusion.
        rrf_k: RRF smoothing constant (lower = more discriminative).
        doc_ids: Optional filter by document IDs.
        dense_weight: RRF weight for dense signal (default from config).
        bm25_weight: RRF weight for BM25 signal (default from config).
        sparse_weight: RRF weight for sparse signal (default from config, 0.25 = secondary).
        diversity_mode: Optional post-processing: "cap" or "mmr". None = no diversity.
          Overrides config flags DOC_CAP_ENABLED / MMR_ENABLED when explicitly set.
        doc_cap: Max chunks per doc_id for "cap" mode (default from config DOC_CAP_N).
        mmr_lambda: Lambda for "mmr" mode (default from config MMR_LAMBDA).

    Returns:
        List of chunk dicts sorted by rrf_score, each with five-score fields.
    """
    top_k = top_k or RETRIEVAL_TOP_K
    rrf_k = rrf_k or RRF_K
    dense_weight = dense_weight if dense_weight is not None else DENSE_RRF_WEIGHT
    bm25_weight = bm25_weight if bm25_weight is not None else BM25_RRF_WEIGHT
    sparse_weight = sparse_weight if sparse_weight is not None else SPARSE_RRF_WEIGHT
    candidate_k = top_k * _CANDIDATE_MULTIPLIER

    # ------------------------------------------------------------------
    # Stage 1a: Dense search
    # ------------------------------------------------------------------
    if query_dense is not None:
        dense_results = vector_store.query(query_dense, candidate_k, doc_ids=doc_ids)
        dense_ids: List[str] = dense_results["ids"]
    else:
        dense_ids = []

    # ------------------------------------------------------------------
    # Stage 1b: BM25 search
    # ------------------------------------------------------------------
    bm25_results = fts_store.query(query, candidate_k, doc_ids=doc_ids)

    # ------------------------------------------------------------------
    # Candidate pool = ordered union (dense order preserved first)
    # ------------------------------------------------------------------
    seen: set = set()
    candidate_ids: List[str] = []
    for cid in dense_ids + [r["id"] for r in bm25_results]:
        if cid not in seen:
            seen.add(cid)
            candidate_ids.append(cid)

    if not candidate_ids:
        logger.warning("Hybrid search: empty candidate pool")
        return []

    # ------------------------------------------------------------------
    # Stage 2: Sparse rescore on candidates only
    # Guard: skip if coverage < threshold to avoid ranking bias on partial data
    # ------------------------------------------------------------------
    sparse_ranking = []
    if query_sparse and doc_store._conn is not None:
        coverage = _get_sparse_coverage(doc_store._conn)
        if coverage < SPARSE_COVERAGE_THRESHOLD:
            logger.warning(
                f"Sparse scoring disabled: coverage {coverage:.1%} < "
                f"threshold {SPARSE_COVERAGE_THRESHOLD:.0%}. "
                "Run: python -m rag_lab.maintenance.backfill_sparse"
            )
        else:
            sparse_data = load_sparse_for_chunks(doc_store._conn, candidate_ids)
            sparse_ranking = rank_candidates_by_sparse(query_sparse, candidate_ids, sparse_data)

    # ------------------------------------------------------------------
    # Stage 3: Weighted RRF fusion
    # ------------------------------------------------------------------
    fused = weighted_rrf(
        dense_ids, bm25_results, sparse_ranking,
        dense_w=dense_weight, bm25_w=bm25_weight, sparse_w=sparse_weight,
        k=rrf_k,
    )
    top_fused = fused[:top_k]
    score_map = {item["id"]: item for item in top_fused}
    top_ids = list(score_map.keys())

    # ------------------------------------------------------------------
    # Stage 4: Retrieve full chunks from docstore
    # ------------------------------------------------------------------
    chunks = doc_store.get_by_ids(top_ids)

    # Attach five-score fields to each chunk
    for chunk in chunks:
        cid = chunk["chunk_id"]
        info = score_map.get(cid, {})
        chunk["rrf_score"] = info.get("rrf_score", 0.0)
        chunk["dense_score"] = info.get("dense_score", 0.0)
        chunk["bm25_score"] = info.get("bm25_score", 0.0)
        chunk["sparse_score"] = info.get("sparse_score", 0.0)
        chunk["in_dense_topk"] = info.get("in_dense_topk", False)
        chunk["in_bm25_topk"] = info.get("in_bm25_topk", False)
        chunk["in_sparse_topk"] = info.get("in_sparse_topk", False)

    # Sort by rrf_score to match fusion order
    chunks.sort(key=lambda c: c.get("rrf_score", 0.0), reverse=True)

    # ------------------------------------------------------------------
    # Optional diversity post-processing (experimental)
    # ------------------------------------------------------------------
    effective_mode = diversity_mode
    if effective_mode is None:
        if DOC_CAP_ENABLED:
            effective_mode = "cap"
        elif MMR_ENABLED:
            effective_mode = "mmr"

    if effective_mode == "cap":
        cap = doc_cap if doc_cap is not None else DOC_CAP_N
        chunks = apply_document_cap(chunks, cap)
        logger.debug(f"document_cap(N={cap}): {len(chunks)} chunks after capping")
    elif effective_mode == "mmr":
        lam = mmr_lambda if mmr_lambda is not None else MMR_LAMBDA
        chunks = apply_mmr(chunks, lambda_=lam, k=top_k)
        logger.debug(f"MMR(lambda={lam}): {len(chunks)} chunks after reranking")

    logger.info(
        f"Hybrid search: {len(dense_ids)} dense + {len(bm25_results)} BM25 → "
        f"{len(candidate_ids)} candidates → {len(chunks)} returned"
    )

    if _return_stats:
        stats = {
            "candidate_pool_size": len(candidate_ids),
            "n_dense": len(dense_ids),
            "n_bm25": len(bm25_results),
            "n_sparse": len(sparse_ranking),
            "sparse_used": len(sparse_ranking) > 0,
        }
        return chunks, stats

    return chunks
