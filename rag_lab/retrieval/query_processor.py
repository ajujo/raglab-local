"""Query processing: expansion and optional HyDE.

Transforms the user's question into representations optimal for retrieval.
"""

import logging
from typing import List, Tuple

from rag_lab.config import EMBEDDING_MODEL, VARIANTS_COUNT, HYDE_ENABLED
from rag_lab.embedding.encoder import load_embedding_model

logger = logging.getLogger("rag_lab")


def process_query(
    query: str,
    use_hyde: bool = False,
    top_k: int = 20,
) -> List[dict]:
    """Process a user query for retrieval.

    Args:
        query: The user's question.
        use_hyde: If True, generate hypothetical document for embedding.
        top_k: Number of results to retrieve.

    Returns:
        List of query dicts with 'text', 'dense', 'sparse' keys.
    """
    queries = [{"text": query, "type": "original"}]

    if use_hyde:
        hypothetical = _generate_hypothetical_answer(query)
        if hypothetical:
            queries.append({
                "text": hypothetical,
                "type": "hyde",
            })

    # Query expansion: generate variants
    for i in range(VARIANTS_COUNT):
        variant = _generate_query_variant(query, i)
        if variant and variant != query:
            queries.append({
                "text": variant,
                "type": "expanded",
            })

    logger.info(f"Processed query into {len(queries)} query variants")
    return queries


def _generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer for HyDE.

    In a full implementation, this would call an LLM to generate
    a plausible answer. For now, returns a simple template.

    Args:
        query: The user's question.

    Returns:
        Hypothetical answer text.
    """
    # Simple template - in production, call an LLM
    return (
        f"This question is about {query.lower()}. "
        f"The answer involves technical specifications and "
        f"implementation details related to the topic."
    )


def _generate_query_variant(query: str, variant_idx: int) -> str:
    """Generate a query variant for expansion.

    Args:
        query: The original query.
        variant_idx: Index of the variant.

    Returns:
        A variant of the query.
    """
    stop_words = {
        # English
        "what", "is", "the", "of", "in", "on", "about", "how", "which",
        "are", "was", "were", "do", "does", "did", "a", "an", "and", "or",
        "to", "for", "with", "by", "from", "at", "that", "this", "it",
        # Spanish
        "qué", "cuál", "cuáles", "cómo", "dónde", "cuando", "cuándo",
        "quién", "quiénes", "por", "para", "una", "uno", "un", "los",
        "las", "del", "con", "que", "se", "en", "el", "la", "es", "son",
        "de", "al", "como", "este", "esta", "estos", "estas",
        # Punctuation-like
        "¿", "?",
    }

    words = query.lower().replace("¿", "").replace("?", "").split()
    filtered = [w for w in words if w.strip("'\"(),.:;") not in stop_words]

    if not filtered:
        return query

    if variant_idx == 0:
        # Key terms only
        return " ".join(filtered)
    elif variant_idx == 1:
        # Tail terms (often contain the specific topic)
        return " ".join(filtered[max(0, len(filtered) - 5):])