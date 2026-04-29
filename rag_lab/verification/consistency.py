"""Componente 2: Self-consistency check (Faithfulness check).

Detecta si la respuesta del LLM es coherente con los chunks que recibió,
sin inventar información ni contradecir las fuentes.
"""

import json
import logging
from typing import Dict, List, Optional

from rag_lab.generation.llm_client import generate_response
from rag_lab.config import LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger("rag_lab")

CONSISTENCY_SYSTEM_PROMPT = """\
Eres un evaluador de coherencia de respuestas RAG. Tu tarea es verificar si la respuesta generada es fiel a los fragmentos proporcionados.
"""

CONSISTENCY_USER_PROMPT_TEMPLATE = """\
Dado este conjunto de fragmentos de documento:
<chunks>
{chunks_text}
</chunks>

Y esta respuesta generada:
<respuesta>
{response}
</respuesta>

Evalúa si la respuesta:
a) Contiene alguna afirmación que NO está respaldada por los fragmentos
b) Contradice algún fragmento
c) Inventa datos, cifras o definiciones que no aparecen en los fragmentos

Responde SOLO en JSON con este esquema:
{
  "has_unsupported_claims": true/false,
  "has_contradictions": true/false,
  "has_hallucinations": true/false,
  "details": "<explicación breve si alguno es true, vacío si todo es false>"
}
"""


def check_consistency(
    response: str,
    retrieved_chunks: List[dict],
    enable_consistency_check: bool = True,
) -> Optional[Dict[str, object]]:
    """Ejecutar el self-consistency check.

    Args:
        response: La respuesta generada por el LLM.
        retrieved_chunks: Lista de chunks recuperados.
        enable_consistency_check: Si es False, se omite el check.

    Returns:
        Dict con los resultados de la evaluación, o None si está desactivado.
    """
    if not enable_consistency_check:
        logger.debug("Consistency check desactivado")
        return None

    # Construir el texto de los chunks
    chunks_text = "\n".join([
        f"[{i+1}] {c.get('text', '')}"
        for i, c in enumerate(retrieved_chunks)
    ])

    # Construir el prompt del usuario
    user_prompt = CONSISTENCY_USER_PROMPT_TEMPLATE.format(
        chunks_text=chunks_text,
        response=response
    )

    try:
        # Llamar al LLM para la evaluación
        result_text = generate_response(
            CONSISTENCY_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,  # Determinista para evaluación
        )

        if not result_text:
            logger.warning("El LLM no devolvió respuesta de consistencia")
            return None

        # Limpiar la respuesta para extraer el JSON
        json_str = result_text.strip()
        # Intentar extraer JSON si hay texto adicional
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = json_str[start:end]

        result = json.loads(json_str)
        logger.info(f"Consistency check completado: {result}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Error al parsear JSON de consistencia: {e}")
        return None
    except Exception as e:
        logger.error(f"Error en el consistency check: {e}")
        return None