"""Cross-encoder reranking with BGE-reranker-v2-m3.

Reranks candidate chunks using a cross-encoder model for higher precision.
"""

import logging
from typing import Dict, List, Optional, Tuple

from rag_lab.config import RERANK_TOP_K, RERANKER_DEVICE
from rag_lab.exceptions import EmbeddingError

logger = logging.getLogger("rag_lab")

# Global reranker cache
_reranker_cache: Optional[object] = None


def reset_reranker_cache() -> None:
    """Reset the global reranker cache.

    Useful for tests that need to load the model on a different device.

    Returns:
        None
    """
    global _reranker_cache
    _reranker_cache = None


def rerank(
    query: str,
    chunks: List[dict],
    top_k: int = None,
    device: str = None,
) -> List[dict]:
    """Rerank chunks using cross-encoder scoring.

    Args:
        query: The user's question.
        chunks: List of chunk dicts with 'text' key.
        top_k: Number of results to return.
        device: Device to run reranker on. If None, uses RERANKER_DEVICE from config.

    Returns:
        List of chunk dicts sorted by relevance score.
    """
    if not chunks:
        return []

    top_k = top_k or RERANK_TOP_K
    device = device or RERANKER_DEVICE

    try:
        reranker = load_reranker(device)
        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = reranker.compute_score(pairs)

        # Attach scores to chunks
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = float(scores[i])
            scored_chunks.append(chunk_copy)

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(f"Reranked {len(scored_chunks)} chunks")
        return scored_chunks[:top_k]

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Fallback: return chunks sorted by position
        return chunks[:top_k]


def load_reranker(device: str = None) -> object:
    """Load the BGE reranker model.

    Args:
        device: Device to run model on. If None, uses RERANKER_DEVICE from config.

    Returns:
        The reranker model.
    """
    global _reranker_cache

    if _reranker_cache is not None:
        return _reranker_cache

    device = device or RERANKER_DEVICE

    try:
        from FlagEmbedding import FlagReranker
        _reranker_cache = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=True,
            device=device,
        )
        logger.info(f"Loaded BGE reranker on {device}")
        return _reranker_cache
    except Exception as e:
        raise EmbeddingError(f"Failed to load reranker: {e}")