"""Query processing: expansion and optional HyDE.

Transforms the user's question into representations optimal for retrieval.
"""

import logging
from typing import List

from rag_lab.config import (
    HYDE_ENABLED,
    QUERY_REWRITING_ENABLED,
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
        query = rewrite_query(
            query,
            llm_call=lambda prompt: generate_response("", prompt),
        )

    queries = [{"text": query, "type": "original"}]

    # Step 2: HyDE (if enabled, operates on the potentially rewritten query)
    if use_hyde:
        hypothetical = _generate_hypothetical_answer(query)
        if hypothetical:
            queries.append({
                "text": hypothetical,
                "type": "hyde",
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
Escribe como si fueras un experto respondiendo desde su conocimiento.

Pregunta: {question}
"""

# HyDE only needs a short hypothetical paragraph — bounded to prevent thinking
# mode or verbose generation from consuming large token budgets.
HYDE_MAX_TOKENS = 300
HYDE_TEMPERATURE = 0.1


def _generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer for HyDE using the real LLM.

    Calls the LLM to generate a plausible technical answer to the query,
    which is then used as a hypothetical document for embedding-based retrieval.

    Uses HYDE_MAX_TOKENS=300 and HYDE_TEMPERATURE=0.1 to keep generation short
    and focused. generate_response already passes enable_thinking=False via
    chat_template_kwargs, so thinking mode is suppressed on supporting servers.

    Args:
        query: The user's question.

    Returns:
        Hypothetical answer text, or original query on LLM failure.
    """
    user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(question=query)

    try:
        hypothetical = generate_response(
            HYDE_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=HYDE_MAX_TOKENS,
            temperature=HYDE_TEMPERATURE,
        )
        if hypothetical:
            n_tokens = _count_tokens(hypothetical)
            logger.info(
                f"HyDE: hipótesis generada ({n_tokens} tokens) para query: \"{query[:60]}...\""
            )
            return hypothetical
        else:
            logger.warning("HyDE: LLM devolvió texto vacío, usando query original como fallback")
            return query

    except Exception as e:
        logger.warning(f"HyDE: error al llamar al LLM: {e}. Usando query original como fallback.")
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