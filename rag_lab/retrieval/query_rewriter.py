"""Query rewriting module.

Reformulates user questions to maximize semantic retrieval effectiveness
by expanding acronyms, adding domain terminology, and clarifying intent.
"""

import logging
from typing import Callable

from rag_lab.generation.llm_client import generate_response, LLMConnectionError

logger = logging.getLogger("rag_lab")

QUERY_REWRITER_SYSTEM_PROMPT = """\
Eres un experto en optimización de consultas para sistemas de búsqueda semántica.
Tu tarea es reescribir preguntas para que sean más efectivas al buscar
en documentación técnica especializada.
"""

QUERY_REWRITER_USER_PROMPT_TEMPLATE = """\
Reescribe la siguiente pregunta para que sea más efectiva en una búsqueda
semántica sobre documentación técnica especializada.

Reglas:
- Mantén el significado original exacto
- Expande siglas si las reconoces (ej. DSD → Data Structure Definition)
- Usa terminología técnica del dominio si es relevante
- Devuelve ÚNICAMENTE la pregunta reescrita, sin explicaciones ni comillas

Pregunta original: {question}
"""


def rewrite_query(question: str, llm_call: Callable[[str], str]) -> str:
    """Reformulate the user's question to maximize semantic retrieval.

    Calls the LLM to rewrite the query with domain-specific terminology
    and expanded acronyms. If the LLM call fails or returns empty text,
    falls back to the original question.

    Args:
        question: The user's original question.
        llm_call: Callable that takes a prompt string and returns LLM output.

    Returns:
        The rewritten question, or the original if rewriting fails.
    """
    user_prompt = QUERY_REWRITER_USER_PROMPT_TEMPLATE.format(question=question)

    try:
        rewritten = llm_call(user_prompt)

        if rewritten:
            logger.info(f"QueryRewriter: \"{question}\" → \"{rewritten}\"")
            return rewritten
        else:
            logger.warning("QueryRewriter: LLM devolvió texto vacío, usando pregunta original")
            return question

    except Exception as e:
        logger.warning(f"QueryRewriter: error al llamar al LLM: {e}. Usando pregunta original.")
        return question
