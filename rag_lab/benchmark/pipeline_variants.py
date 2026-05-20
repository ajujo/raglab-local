"""Five retrieval pipeline variants for comparative benchmarking.

Each variant function:
  - Accepts pre-computed query embeddings (no re-encoding)
  - Returns (chunks: List[dict], stats: dict)
  - stats contains: candidate_pool_size, n_dense, n_bm25, n_sparse, latency_ms

Variants
--------
  dense       Dense cosine (ChromaDB only)
  bm25        BM25 / FTS5 (no dense, no sparse)
  dense_bm25  RRF2: dense ∪ BM25
  hybrid      RRF3: dense ∪ BM25 ∪ BGE-M3 sparse rescore
  full        hybrid + BGE cross-encoder reranker
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from rag_lab.retrieval.hybrid_search import hybrid_search
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore

VARIANT_NAMES = ["dense", "bm25", "dense_bm25", "hybrid", "full"]


# ---------------------------------------------------------------------------
# Dense only
# ---------------------------------------------------------------------------

def run_dense(
    query: str,
    query_dense: np.ndarray,
    vector_store: VectorStore,
    doc_store: DocStore,
    top_k: int,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[List[dict], dict]:
    t0 = time.perf_counter()

    raw = vector_store.query(query_dense, top_k, doc_ids=doc_ids)
    ids = raw["ids"]
    distances = raw["distances"]

    chunks = doc_store.get_by_ids(ids)
    dist_map = {cid: d for cid, d in zip(ids, distances)}
    for c in chunks:
        d = dist_map.get(c["chunk_id"], 1.0)
        c["dense_score"] = max(0.0, round(1.0 - d, 6))   # cosine sim = 1 - dist
        c["bm25_score"] = 0.0
        c["sparse_score"] = 0.0
        c["rrf_score"] = 0.0
        c["in_dense_topk"] = True
        c["in_bm25_topk"] = False
        c["in_sparse_topk"] = False
    chunks.sort(key=lambda c: c["dense_score"], reverse=True)

    latency_ms = (time.perf_counter() - t0) * 1000
    stats = {
        "latency_ms": latency_ms,
        "candidate_pool_size": len(ids),
        "n_dense": len(ids),
        "n_bm25": 0,
        "n_sparse": 0,
        "sparse_used": False,
    }
    return chunks, stats


# ---------------------------------------------------------------------------
# BM25 only
# ---------------------------------------------------------------------------

def run_bm25(
    query: str,
    fts_store: FTSStore,
    doc_store: DocStore,
    top_k: int,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[List[dict], dict]:
    t0 = time.perf_counter()

    fts_results = fts_store.query(query, top_k, doc_ids=doc_ids)
    ids = [r["id"] for r in fts_results]
    score_map = {r["id"]: r["bm25_score"] for r in fts_results}

    chunks = doc_store.get_by_ids(ids)
    for c in chunks:
        c["dense_score"] = 0.0
        c["bm25_score"] = score_map.get(c["chunk_id"], 0.0)
        c["sparse_score"] = 0.0
        c["rrf_score"] = 0.0
        c["in_dense_topk"] = False
        c["in_bm25_topk"] = True
        c["in_sparse_topk"] = False
    chunks.sort(key=lambda c: c["bm25_score"], reverse=True)

    latency_ms = (time.perf_counter() - t0) * 1000
    stats = {
        "latency_ms": latency_ms,
        "candidate_pool_size": len(ids),
        "n_dense": 0,
        "n_bm25": len(ids),
        "n_sparse": 0,
        "sparse_used": False,
    }
    return chunks, stats


# ---------------------------------------------------------------------------
# Dense + BM25 (RRF2, sparse disabled)
# ---------------------------------------------------------------------------

def run_dense_bm25(
    query: str,
    query_dense: np.ndarray,
    vector_store: VectorStore,
    doc_store: DocStore,
    fts_store: FTSStore,
    top_k: int,
    rrf_k: int,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[List[dict], dict]:
    t0 = time.perf_counter()

    # Pass query_sparse=None to disable sparse stage in hybrid_search
    chunks, hs_stats = hybrid_search(
        query,
        vector_store,
        doc_store,
        fts_store,
        query_dense=query_dense,
        query_sparse=None,
        top_k=top_k,
        rrf_k=rrf_k,
        doc_ids=doc_ids,
        _return_stats=True,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    stats = {"latency_ms": latency_ms} | hs_stats
    return chunks, stats


# ---------------------------------------------------------------------------
# Dense + BM25 + Sparse rescore (RRF3)
# ---------------------------------------------------------------------------

def run_hybrid(
    query: str,
    query_dense: np.ndarray,
    query_sparse: Dict[int, float],
    vector_store: VectorStore,
    doc_store: DocStore,
    fts_store: FTSStore,
    top_k: int,
    rrf_k: int,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[List[dict], dict]:
    t0 = time.perf_counter()

    chunks, hs_stats = hybrid_search(
        query,
        vector_store,
        doc_store,
        fts_store,
        query_dense=query_dense,
        query_sparse=query_sparse,
        top_k=top_k,
        rrf_k=rrf_k,
        doc_ids=doc_ids,
        _return_stats=True,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    stats = {"latency_ms": latency_ms} | hs_stats
    return chunks, stats


# ---------------------------------------------------------------------------
# Full pipeline: hybrid + cross-encoder reranker
# ---------------------------------------------------------------------------

def run_full(
    query: str,
    query_dense: np.ndarray,
    query_sparse: Dict[int, float],
    vector_store: VectorStore,
    doc_store: DocStore,
    fts_store: FTSStore,
    top_k: int,
    rrf_k: int,
    rerank_device: str,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[List[dict], dict]:
    from rag_lab.retrieval.reranker import rerank

    t0 = time.perf_counter()

    chunks, hs_stats = hybrid_search(
        query,
        vector_store,
        doc_store,
        fts_store,
        query_dense=query_dense,
        query_sparse=query_sparse,
        top_k=top_k,
        rrf_k=rrf_k,
        doc_ids=doc_ids,
        _return_stats=True,
    )

    # Rerank ALL candidates (not just top-8) to get a fair full-list ranking
    if chunks:
        chunks = rerank(query, chunks, top_k=len(chunks), device=rerank_device)

    latency_ms = (time.perf_counter() - t0) * 1000
    stats = {"latency_ms": latency_ms} | hs_stats
    return chunks, stats


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_variant(
    name: str,
    query: str,
    query_dense: np.ndarray,
    query_sparse: Dict[int, float],
    vector_store: VectorStore,
    doc_store: DocStore,
    fts_store: FTSStore,
    top_k: int,
    rrf_k: int,
    rerank_device: str,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[List[dict], dict]:
    """Run one named variant. Returns (chunks, stats)."""
    if name == "dense":
        return run_dense(query, query_dense, vector_store, doc_store, top_k, doc_ids)
    if name == "bm25":
        return run_bm25(query, fts_store, doc_store, top_k, doc_ids)
    if name == "dense_bm25":
        return run_dense_bm25(query, query_dense, vector_store, doc_store, fts_store,
                              top_k, rrf_k, doc_ids)
    if name == "hybrid":
        return run_hybrid(query, query_dense, query_sparse, vector_store, doc_store,
                          fts_store, top_k, rrf_k, doc_ids)
    if name == "full":
        return run_full(query, query_dense, query_sparse, vector_store, doc_store,
                        fts_store, top_k, rrf_k, rerank_device, doc_ids)
    raise ValueError(f"Unknown variant: {name!r}. Choose from: {VARIANT_NAMES}")
