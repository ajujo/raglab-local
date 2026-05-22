"""Cross-encoder reranking with BGE-reranker-v2-m3.

Reranks candidate chunks using a cross-encoder model for higher precision.
"""

import logging
from typing import List, Optional

from rag_lab.config import RERANK_TOP_K, RERANKER_DEVICE, RERANKER_USE_HEADING_CONTEXT
from rag_lab.exceptions import EmbeddingError

logger = logging.getLogger("rag_lab")

# Global reranker cache — keyed by device so switching devices forces a reload.
_reranker_cache: Optional[object] = None
_reranker_cache_device: Optional[str] = None


def reset_reranker_cache() -> None:
    """Reset the global reranker cache.

    Clears both the cached model and the device it was loaded on, so the
    next call to load_reranker() will load a fresh instance regardless of device.
    """
    global _reranker_cache, _reranker_cache_device
    _reranker_cache = None
    _reranker_cache_device = None


_HEADING_PATH_MAX_LEN = 200  # chars; truncate anomalously long paths


def build_reranker_text(chunk: dict, use_heading_context: bool = True) -> str:
    """Build the text passed to the cross-encoder for a single chunk.

    When *use_heading_context* is True, prepends structural metadata so the
    model can use section context when scoring relevance:

        Document: <doc_id>
        Section: <heading_path>

        <text>

    Falls back gracefully when fields are absent or empty.
    """
    if not use_heading_context:
        return chunk.get("text", "")

    doc_id = (chunk.get("doc_id") or "").strip()
    heading_path = (chunk.get("heading_path") or "").strip()
    text = chunk.get("text", "")

    if heading_path and len(heading_path) > _HEADING_PATH_MAX_LEN:
        heading_path = heading_path[:_HEADING_PATH_MAX_LEN]

    if doc_id and heading_path:
        return f"Document: {doc_id}\nSection: {heading_path}\n\n{text}"
    if doc_id:
        return f"Document: {doc_id}\n\n{text}"
    if heading_path:
        return f"Section: {heading_path}\n\n{text}"
    return text


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

    use_heading = RERANKER_USE_HEADING_CONTEXT

    try:
        reranker = load_reranker(device)
        pairs = [[query, build_reranker_text(chunk, use_heading)] for chunk in chunks]
        scores = reranker.compute_score(pairs)

        scored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_copy = dict(chunk)
            chunk_copy["rerank_score"] = float(scores[i])
            chunk_copy["heading_path_used"] = use_heading and bool(
                (chunk.get("heading_path") or "").strip()
            )
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        n_with_heading = sum(1 for c in scored_chunks if c.get("heading_path_used"))
        logger.info(
            "Reranked %d chunks (heading_context=%s, %d/%d had heading_path)",
            len(scored_chunks), use_heading, n_with_heading, len(scored_chunks),
        )
        return scored_chunks[:top_k]

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return chunks[:top_k]


def load_reranker(device: str = None) -> object:
    """Load the BGE reranker model.

    Returns the cached model when the requested device matches the cached device.
    Reloads when a different device is requested (e.g. switching from cuda to cpu).

    Args:
        device: Device to run model on. If None, uses RERANKER_DEVICE from config.

    Returns:
        The reranker model.
    """
    global _reranker_cache, _reranker_cache_device

    device = device or RERANKER_DEVICE

    if _reranker_cache is not None and _reranker_cache_device == device:
        return _reranker_cache

    # Cache miss or device mismatch — discard stale cache and reload.
    _reranker_cache = None
    _reranker_cache_device = None

    try:
        from FlagEmbedding import FlagReranker
        _reranker_cache = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=True,
            device=device,
        )
        _reranker_cache_device = device
        logger.info(f"Loaded BGE reranker on {device}")
        return _reranker_cache
    except Exception as e:
        raise EmbeddingError(f"Failed to load reranker: {e}")