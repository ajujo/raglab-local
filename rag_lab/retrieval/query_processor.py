"""Query processing: expansion and optional HyDE.

Transforms the user's question into representations optimal for retrieval.
"""

import logging
from typing import List

from rag_lab.config import (
    HYDE_ENABLED,
    HYDE_MAX_TOKENS,
    HYDE_TEMPERATURE,
    HYDE_FORCE_NO_THINKING,
    HYDE_TIMEOUT_SECONDS,
    HYDE_USE_FOR_DENSE,
    HYDE_USE_FOR_BM25,
    HYDE_USE_FOR_SPARSE,
    QUERY_REWRITING_ENABLED,
    QUERY_REWRITING_MAX_TOKENS,
    QUERY_REWRITING_TEMPERATURE,
    QUERY_REWRITING_TIMEOUT_SECONDS,
    QUERY_VARIANT_STOPWORD_ENABLED,
    QUERY_VARIANT_LAST_TERMS_ENABLED,
)
from rag_lab.utils.tokenizer import count_tokens as _count_tokens
from rag_lab.generation.llm_client import generate_response
from rag_lab.retrieval.query_rewriter import rewrite_query

logger = logging.getLogger("rag_lab")


def process_query(
    query: str,
    use_hyde: bool = False,
    use_rewriting: bool = False,
    top_k: int = 20,
) -> List[dict]:
    """Process a user query for retrieval.

    Args:
        query: The user's question.
        use_hyde: If True, generate hypothetical document for embedding.
        use_rewriting: If True, rewrite the query before processing.
        top_k: Number of results to retrieve.

    Returns:
        List of query dicts with 'text', 'dense', 'sparse' keys.
    """
    # Step 1: Query rewriting (if enabled)
    if use_rewriting:
        _rw_timeout = QUERY_REWRITING_TIMEOUT_SECONDS or None
        query = rewrite_query(
            query,
            llm_call=lambda prompt: generate_response(
                "",
                prompt,
                max_tokens=QUERY_REWRITING_MAX_TOKENS,
                temperature=QUERY_REWRITING_TEMPERATURE,
                timeout=_rw_timeout,
                force_no_thinking=True,
            ),
        )

    queries = [{"text": query, "type": "original"}]

    # Step 2: HyDE (if enabled, operates on the potentially rewritten query)
    if use_hyde:
        hypothetical = _generate_hypothetical_answer(query)
        if hypothetical and hypothetical != query:
            queries.append({
                "text": hypothetical,
                "type": "hyde",
                "use_for_dense": HYDE_USE_FOR_DENSE,
                "use_for_bm25": HYDE_USE_FOR_BM25,
                "use_for_sparse": HYDE_USE_FOR_SPARSE,
            })

    # Step 3: Query expansion variants (disabled by default — see A/B results in v1.11)
    if QUERY_VARIANT_STOPWORD_ENABLED:
        variant = _generate_stopword_variant(query)
        if variant and variant != query:
            queries.append({"text": variant, "type": "variant_stopword"})

    if QUERY_VARIANT_LAST_TERMS_ENABLED:
        variant = _generate_last_terms_variant(query)
        if variant and variant != query and not any(q["text"] == variant for q in queries):
            queries.append({"text": variant, "type": "variant_last_terms"})

    logger.info(f"Processed query into {len(queries)} query variants")
    return queries


HYDE_SYSTEM_PROMPT = """\
Eres un experto técnico que genera respuestas hipotéticas para mejorar la recuperación de documentos.
"""

HYDE_USER_PROMPT_TEMPLATE = """\
Genera un párrafo técnico de 3-5 oraciones que respondería directamente
a la siguiente pregunta, usando el vocabulario especializado del dominio.
No cites fuentes. No uses expresiones como "según los documentos".
No rarones en voz alta. No expliques pasos intermedios.
Escribe como si fueras un experto respondiendo desde su conocimiento.
Sé conciso: máximo 3-5 oraciones.

Pregunta: {question}
"""


def _generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer for HyDE using the real LLM.

    Calls the LLM to generate a plausible technical answer to the query,
    which is then used as a hypothetical document for embedding-based retrieval.

    Uses HYDE_MAX_TOKENS and HYDE_TEMPERATURE from config. When
    HYDE_FORCE_NO_THINKING=True, skips the thinking token multiplier and
    passes enable_thinking=False — the token budget is used entirely for
    the hypothetical answer, not for reasoning chains.

    Args:
        query: The user's question.

    Returns:
        Hypothetical answer text, or original query on LLM failure.
    """
    user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(question=query)
    _timeout = HYDE_TIMEOUT_SECONDS if HYDE_TIMEOUT_SECONDS > 0 else None

    try:
        hypothetical = generate_response(
            HYDE_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=HYDE_MAX_TOKENS,
            temperature=HYDE_TEMPERATURE,
            timeout=_timeout,
            force_no_thinking=HYDE_FORCE_NO_THINKING,
        )
        if hypothetical:
            n_tokens = _count_tokens(hypothetical)
            logger.info(
                "HyDE: hypothesis generated (%d tokens) for query: \"%s...\"",
                n_tokens, query[:60],
            )
            return hypothetical
        else:
            logger.warning("HyDE: LLM returned empty text, falling back to original query")
            return query

    except Exception as e:
        logger.warning("HyDE: LLM call failed (%s). Falling back to original query.", e)
        return query


_STOP_WORDS = frozenset({
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
})


def _filtered_terms(query: str) -> list:
    """Return query words that are not stop words, lowercased, stripped of punctuation."""
    words = query.lower().replace("¿", "").replace("?", "").split()
    stripped = [w.strip("'\"(),.:;") for w in words]
    return [w for w in stripped if w and w not in _STOP_WORDS]


def _generate_stopword_variant(query: str) -> str:
    """Return query with stop words removed (key terms only).

    Example: "What is the role of SDMX?" → "role sdmx"
    Falls back to original query when no terms remain after filtering.
    """
    filtered = _filtered_terms(query)
    return " ".join(filtered) if filtered else query


def _generate_last_terms_variant(query: str) -> str:
    """Return the last 5 key terms of the query (tail focus).

    Useful for long queries where the specific topic is mentioned last.
    Example: "What are the rules for DSD key families in SDMX?" → "rules dsd key families sdmx"
    Falls back to original query when no terms remain after filtering.
    """
    filtered = _filtered_terms(query)
    if not filtered:
        return query
    return " ".join(filtered[max(0, len(filtered) - 5):])


def _generate_query_variant(query: str, variant_idx: int) -> str:
    """Legacy dispatcher kept for backward compatibility.

    Prefer _generate_stopword_variant / _generate_last_terms_variant directly.
    """
    if variant_idx == 0:
        return _generate_stopword_variant(query)
    elif variant_idx == 1:
        return _generate_last_terms_variant(query)
    return query