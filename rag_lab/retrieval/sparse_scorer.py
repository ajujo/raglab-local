"""Sparse vector scoring from SQLite BLOBs.

Loads BGE-M3 sparse vectors stored as parallel binary arrays in docstore.sqlite
and computes dot-product similarity against a query sparse vector.

Storage format (sparse_format_version = 1):
  sparse_tokens  BLOB  — np.int32 array of token IDs
  sparse_weights BLOB  — np.float32 array of corresponding weights
"""

import logging
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("rag_lab")


def load_sparse_for_chunks(
    conn: sqlite3.Connection,
    chunk_ids: List[str],
) -> Dict[str, Tuple[Optional[bytes], Optional[bytes]]]:
    """Load sparse BLOBs for a set of chunk IDs.

    Args:
        conn: Open SQLite connection to docstore.sqlite.
        chunk_ids: List of chunk IDs to load.

    Returns:
        Dict mapping chunk_id -> (tokens_blob, weights_blob).
        Either blob may be None if the chunk has no sparse data.
    """
    if not chunk_ids:
        return {}

    placeholders = ",".join(["?"] * len(chunk_ids))
    try:
        rows = conn.execute(
            f"SELECT chunk_id, sparse_tokens, sparse_weights "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
    except sqlite3.OperationalError:
        # sparse_tokens/sparse_weights columns don't exist yet (pre-migration)
        return {}

    return {row[0]: (row[1], row[2]) for row in rows}


def sparse_score(
    query_sparse: Dict[int, float],
    tokens_blob: Optional[bytes],
    weights_blob: Optional[bytes],
) -> float:
    """Compute dot-product similarity between a query sparse vector and stored BLOBs.

    Args:
        query_sparse: Query sparse vector as {token_id: weight}.
        tokens_blob: np.int32 binary array of document token IDs.
        weights_blob: np.float32 binary array of document token weights.

    Returns:
        Dot-product similarity score (0.0 if no data).
    """
    if not tokens_blob or not weights_blob:
        return 0.0
    if not query_sparse:
        return 0.0

    tokens = np.frombuffer(tokens_blob, dtype=np.int32)
    weights = np.frombuffer(weights_blob, dtype=np.float32)

    # Vectorized dot product: sum(q[t] * w) for each (t, w) in doc
    q_weights = np.array([query_sparse.get(int(t), 0.0) for t in tokens], dtype=np.float32)
    return float(np.dot(q_weights, weights))


def rank_candidates_by_sparse(
    query_sparse: Dict[int, float],
    candidate_ids: List[str],
    sparse_data: Dict[str, Tuple[Optional[bytes], Optional[bytes]]],
) -> List[dict]:
    """Score and rank a candidate pool by sparse similarity.

    Args:
        query_sparse: Query sparse vector as {token_id: weight}.
        candidate_ids: Ordered list of candidate chunk IDs.
        sparse_data: Pre-loaded BLOB data from load_sparse_for_chunks().

    Returns:
        List of {"id": chunk_id, "sparse_score": float}, sorted descending.
    """
    results = []
    for cid in candidate_ids:
        blobs = sparse_data.get(cid, (None, None))
        score = sparse_score(query_sparse, blobs[0], blobs[1])
        results.append({"id": cid, "sparse_score": score})
    results.sort(key=lambda x: x["sparse_score"], reverse=True)
    return results
