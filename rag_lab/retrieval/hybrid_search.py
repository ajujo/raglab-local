"""Hybrid search: dense + sparse + RRF fusion.

Combines results from ChromaDB (dense) and sparse index using
Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from rag_lab.config import RRF_K, RETRIEVAL_TOP_K, STORAGE_DIR
from rag_lab.embedding.encoder import load_embedding_model, encode_chunks
from rag_lab.exceptions import RetrievalError
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.sparse_store import SparseStore
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")


def hybrid_search(
    query: str,
    vector_store: VectorStore,
    sparse_store: SparseStore,
    doc_store: DocStore,
    query_dense: np.ndarray = None,
    query_sparse: dict = None,
    top_k: int = None,
    rrf_k: int = None,
    doc_ids: Optional[List[str]] = None,
) -> List[dict]:
    """Perform hybrid search with RRF fusion.

    Args:
        query: The user's question.
        vector_store: ChromaDB wrapper for dense vectors.
        sparse_store: Sparse index wrapper.
        doc_store: SQLite docstore.
        query_dense: Dense embedding for the query.
        query_sparse: Sparse embedding for the query.
        top_k: Number of final results.
        rrf_k: RRF constant.
        doc_ids: Optional list of document IDs to filter by.

    Returns:
        List of chunk dicts sorted by relevance.
    """
    top_k = top_k or RETRIEVAL_TOP_K
    rrf_k = rrf_k or RRF_K

    # Step 1: Dense search
    if query_dense is not None:
        dense_results = vector_store.query(query_dense, top_k, doc_ids=doc_ids)
    else:
        dense_results = {"ids": [], "distances": [], "documents": [], "metadatas": []}

    # Step 2: Sparse search
    if query_sparse is not None:
        sparse_results = sparse_store.query(query_sparse, top_k)
    else:
        sparse_results = []

    # Step 3: RRF fusion
    fused = _reciprocal_rank_fusion(
        dense_results["ids"],
        sparse_results,
        k=rrf_k,
    )

    # Step 4: Get top-k results
    top_ids = [item["id"] for item in fused[:top_k]]

    # Step 5: Retrieve full chunks from docstore
    chunks = doc_store.get_by_ids(top_ids)

    logger.info(f"Hybrid search returned {len(chunks)} chunks")
    return chunks


def _reciprocal_rank_fusion(
    dense_ids: List[str],
    sparse_results: List[dict],
    k: int = 60,
) -> List[dict]:
    """Compute Reciprocal Rank Fusion scores.

    Args:
        dense_ids: List of IDs from dense search (ranked).
        sparse_results: List of dicts with 'id' and 'score' from sparse search.
        k: RRF constant.

    Returns:
        List of dicts with 'id', 'rrf_score', 'dense_score', 'sparse_score'.
    """
    scores = {}
    ranks = {}

    # Dense results get ranks 0, 1, 2, ...
    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = 1.0 / (k + rank + 1)
        ranks[chunk_id] = rank

    # Sparse results get ranks based on score
    for i, item in enumerate(sparse_results):
        chunk_id = item["id"]
        if chunk_id in scores:
            scores[chunk_id] += 1.0 / (k + i + 1)
        else:
            scores[chunk_id] = 1.0 / (k + i + 1)
        ranks[chunk_id] = i

    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Add scores to results
    results = []
    for chunk_id, rrf_score in sorted_scores:
        # Find original scores
        dense_score = 1.0 / (k + ranks.get(chunk_id, len(dense_ids)))
        sparse_score = 0.0
        for item in sparse_results:
            if item["id"] == chunk_id:
                sparse_score = item["score"]
                break

        results.append({
            "id": chunk_id,
            "rrf_score": rrf_score,
            "dense_score": dense_score,
            "sparse_score": sparse_score,
        })

    return results