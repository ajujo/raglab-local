"""Build prompts for the LLM.

Constructs system and user prompts for the RAG generation phase.
"""

import logging
from typing import List

from rag_lab.config import LLM_MODEL, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger("rag_lab")


def build_prompt(
    query: str,
    chunks: List[dict],
) -> tuple[str, str]:
    """Build system and user prompts for the LLM.

    Args:
        query: The user's question.
        chunks: List of retrieved chunk dicts.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        heading = chunk.get("heading_path", "Sin encabezado")
        text = chunk.get("text", "")
        line_start = chunk.get("line_start", "N/A")
        line_end = chunk.get("line_end", "N/A")
        context_parts.append(f"[{i}] Fuente: {chunk.get('doc_id', 'N/A')} | Sección: {heading} | Líneas: {line_start}-{line_end}\n---\n{text}\n")
    context = "".join(context_parts)

    # Build user prompt from template
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=query)

    logger.info(f"Built prompt with {len(chunks)} chunks")
    return (SYSTEM_PROMPT, user_prompt)