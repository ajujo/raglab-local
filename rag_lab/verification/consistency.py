"""Componente 2: Self-consistency check (Faithfulness check).

Detecta si la respuesta del LLM es coherente con los chunks que recibió,
sin inventar información ni contradecir las fuentes.
"""

import logging
import re
from dataclasses import dataclass
from typing import Callable, List

from rag_lab.generation.llm_client import generate_response

logger = logging.getLogger("rag_lab")

CONSISTENCY_PROMPT = """\
Analiza si la siguiente respuesta está respaldada por los fragmentos de documento proporcionados.

## Fragmentos de referencia
{chunks}

## Respuesta a evaluar
{response}

## Instrucciones
Responde ÚNICAMENTE con estas cuatro líneas, sin texto adicional antes ni después:
UNSUPPORTED: SÍ o NO
CONTRADICTIONS: SÍ o NO
HALLUCINATIONS: SÍ o NO
DETAILS: <una sola frase explicando el problema, o vacío si todo es NO>

Ejemplo de respuesta correcta:
UNSUPPORTED: NO
CONTRADICTIONS: NO
HALLUCINATIONS: NO
DETAILS:
"""


@dataclass
class ConsistencyResult:
    has_unsupported_claims: bool
    has_contradictions: bool
    has_hallucinations: bool
    details: str
    score: float          # 1.0 OK | 0.5 problemas menores | 0.0 alucinaciones
    parse_success: bool   # False si se agotaron los reintentos


def _parse_response(raw: str) -> dict | None:
    """
    Parsea el formato clave:valor simple.
    Devuelve None si algún campo obligatorio falta o no es SÍ/NO.
    """
    result = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip().upper()] = value.strip()

    required = {"UNSUPPORTED", "CONTRADICTIONS", "HALLUCINATIONS"}
    if not required.issubset(result.keys()):
        return None

    for key in required:
        if result[key].upper() not in {"SÍ", "SI", "NO"}:
            return None

    return result


def _normalize(value: str) -> bool:
    return value.upper() in {"SÍ", "SI"}


def _compute_score(parsed: dict) -> float:
    if _normalize(parsed["HALLUCINATIONS"]):
        return 0.0
    if _normalize(parsed["UNSUPPORTED"]) or _normalize(parsed["CONTRADICTIONS"]):
        return 0.5
    return 1.0


def run_consistency_check(
    response: str,
    retrieved_chunks: List[dict],
    llm_call: Callable[[str], str],
    max_retries: int = 2,
) -> ConsistencyResult:
    """
    Ejecuta el consistency check con reintentos automáticos.

    Args:
        response:         Texto de la respuesta generada por el LLM principal.
        retrieved_chunks: Lista de chunks con al menos el campo 'text'.
        llm_call:         Función que acepta un prompt (str) y devuelve un str.
                          Ejemplo: lambda prompt: ollama.generate(prompt)
        max_retries:      Número máximo de reintentos si el parseo falla.

    Returns:
        ConsistencyResult con score y parse_success.
    """
    # Construir el bloque de chunks para el prompt
    chunks_text = "\n\n".join(
        f"[{i+1}] {chunk.get('text', '')}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = CONSISTENCY_PROMPT.format(
        chunks=chunks_text,
        response=response,
    )

    # Intentos con reintentos
    for attempt in range(1, max_retries + 2):  # +2: intento inicial + reintentos
        try:
            raw = llm_call(prompt)
            parsed = _parse_response(raw)

            if parsed is not None:
                logger.debug(f"Consistency check OK en intento {attempt}")
                return ConsistencyResult(
                    has_unsupported_claims=_normalize(parsed["UNSUPPORTED"]),
                    has_contradictions=_normalize(parsed["CONTRADICTIONS"]),
                    has_hallucinations=_normalize(parsed["HALLUCINATIONS"]),
                    details=parsed.get("DETAILS", ""),
                    score=_compute_score(parsed),
                    parse_success=True,
                )

            logger.warning(f"Intento {attempt}: parseo fallido. Raw: {repr(raw[:200])}")

        except Exception as e:
            logger.warning(f"Intento {attempt}: error en LLM call: {e}")

    # Se agotaron los reintentos → score neutro, no penaliza
    logger.error("Consistency check: reintentos agotados. Usando score neutro 0.5.")
    return ConsistencyResult(
        has_unsupported_claims=False,
        has_contradictions=False,
        has_hallucinations=False,
        details="Consistency check no pudo ejecutarse correctamente.",
        score=0.5,
        parse_success=False,
    )