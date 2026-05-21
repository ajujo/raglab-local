"""Document diversity post-processing for hybrid search results.

Two strategies, both applied after weighted_rrf fusion:

  document_cap  — hard limit: at most N chunks per doc_id
  mmr           — soft penalisation via doc-level Maximal Marginal Relevance

Both preserve rrf_score ordering as the relevance signal and are designed
to prevent large documents from monopolising the top-k result slots.

Neither is active by default (see config.py: DOC_CAP_ENABLED, MMR_ENABLED).
"""

from typing import List, Optional


def apply_document_cap(chunks: List[dict], cap: int) -> List[dict]:
    """Limit each doc_id to at most `cap` chunks, preserving rrf_score order.

    Args:
        chunks: Result list sorted by rrf_score descending.
        cap:    Maximum chunks allowed per doc_id (must be >= 1).

    Returns:
        Filtered list; length <= len(chunks).
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    counts: dict = {}
    result = []
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "")
        n = counts.get(doc_id, 0)
        if n < cap:
            result.append(chunk)
            counts[doc_id] = n + 1
    return result


def apply_mmr(
    chunks: List[dict],
    lambda_: float = 0.7,
    k: Optional[int] = None,
) -> List[dict]:
    """Doc-diversity MMR reranking using doc_id as the diversity signal.

    Greedily selects the next chunk that maximises:

        score(d) = lambda_ * rel_norm(d) - (1 - lambda_) * already_seen(d)

    where:
      rel_norm     — rrf_score normalised to [0, 1] within the candidate list
      already_seen — 1.0 if any chunk from d.doc_id is already selected, 0 otherwise

    At lambda_=1.0 the result equals the original rrf_score order (pure relevance).
    At lambda_=0.0 it maximises doc diversity with no regard for score (not useful
    in practice; calibrated default is 0.7).

    The selected chunk dicts gain an extra "mmr_score" field.

    Args:
        chunks:  Result list sorted by rrf_score descending (input unchanged).
        lambda_: Relevance/diversity trade-off in [0, 1].
        k:       Output length. Defaults to len(chunks).

    Returns:
        Re-ordered list of length min(k, len(chunks)).
    """
    if not chunks:
        return chunks
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")

    target = k if k is not None else len(chunks)

    raw_scores = [c.get("rrf_score", 0.0) for c in chunks]
    max_score = max(raw_scores) if raw_scores else 0.0
    if max_score == 0.0:
        return [dict(c) for c in chunks[:target]]

    norm = [s / max_score for s in raw_scores]

    selected: List[dict] = []
    selected_docs: set = set()
    remaining = list(range(len(chunks)))

    while remaining and len(selected) < target:
        best_pos: Optional[int] = None
        best_mmr = float("-inf")

        for pos in remaining:
            rel = norm[pos]
            penalty = 1.0 if chunks[pos].get("doc_id", "") in selected_docs else 0.0
            score = lambda_ * rel - (1.0 - lambda_) * penalty
            if score > best_mmr:
                best_mmr = score
                best_pos = pos

        if best_pos is None:
            break

        chunk = dict(chunks[best_pos])
        chunk["mmr_score"] = round(best_mmr, 6)
        selected.append(chunk)
        selected_docs.add(chunk.get("doc_id", ""))
        remaining.remove(best_pos)

    return selected
