"""Build prompts for the LLM.

Constructs system and user prompts for the RAG generation phase.
"""

import logging
from typing import List

from rag_lab.config import LLM_MODEL

logger = logging.getLogger("rag_lab")

SYSTEM_PROMPT = """\
Eres un asistente especializado en estándares SDMX (Standard for the Exchange of
Statistical Metadata). Responde ÚNICAMENTE basándote en los fragmentos de documentos
proporcionados. Si la información no está en los fragmentos, indícalo explícitamente:
"No encuentro esta información en los documentos proporcionados".

Reglas:
- No inventes datos, cifras ni referencias
- Cita siempre el documento y sección de origen usando [DOC: nombre, Sección: path]
- Si un fragmento citado no contiene la respuesta, indícalo
- Responde en el mismo idioma que la pregunta del usuario
"""


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
    # Build user prompt with chunks
    user_prompt = "## Fragmentos de referencia\n"

    for i, chunk in enumerate(chunks, 1):
        heading = chunk.get("heading_path", "Sin encabezado")
        text = chunk.get("text", "")
        user_prompt += f"\n[{i}] Fuente: {chunk.get('doc_id', 'N/A')} | Sección: {heading}\n---\n{text}\n"

    user_prompt += f"\n## Pregunta\n{query}"

    logger.info(f"Built prompt with {len(chunks)} chunks")
    return (
        "Eres un asistente especializado en estándares SDMX. Responde ÚNICAMENTE "
        "basándote en los fragmentos de documentos proporcionados. Si la información "
        "no está en los fragmentos, indícalo explícitamente: 'No encuentro esta "
        "información en los documentos proporcionados'.\n\n"
        "Reglas:\n"
        "- No inventes datos, cifras ni referencias que no aparezcan textualmente en los fragmentos\n"
        "- Sé EXHAUSTIVO: incluye TODOS los datos, listas, enumeraciones y detalles relevantes "
        "que encuentres en los fragmentos. No omitas elementos de una lista o enumeración\n"
        "- Cita siempre el documento y sección de origen usando [DOC: nombre, Sección: path]\n"
        "- Si un fragmento citado no contiene la respuesta, indícalo\n"
        "- Responde en el mismo idioma que la pregunta del usuario",
        user_prompt,
    )