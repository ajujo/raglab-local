"""Componente 3: Scoring (Puntuación de confianza).

Calcula una puntuación de 0 a 1 que resume la fiabilidad global
de la respuesta del LLM.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

from rag_lab.verification.verifier import CitationResult, CitationStatus
from rag_lab.verification.consistency import ConsistencyResult

logger = logging.getLogger("rag_lab")


class ConfidenceLevel(str, Enum):
    """Nivel de confianza basado en el score final."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ScoreResult:
    """Resultado del scoring."""
    citation_score: float
    retrieval_score: float
    consistency_score: float
    coverage_score: float
    final_score: float
    confidence_level: ConfidenceLevel


def calculate_score(
    citation_results: List[CitationResult],
    retrieval_scores: List[float],
    consistency_result: ConsistencyResult,
    total_retrieved: int,
) -> ScoreResult:
    """Calcular la puntuación de confianza de la respuesta.

    Args:
        citation_results: Resultados de la verificación de citas.
        retrieval_scores: Scores de similitud de los chunks recuperados.
        consistency_result: Resultado del consistency check.
        total_retrieved: Número total de chunks recuperados.

    Returns:
        ScoreResult con todos los sub-scores y el nivel de confianza.
    """
    # 1. citation_score: proporción de citas VALID sobre el total
    if not citation_results:
        citation_score = 0.5  # Neutro si no hay citas
    else:
        valid_count = sum(1 for r in citation_results if r.status == CitationStatus.VALID)
        citation_score = valid_count / len(citation_results)

    # 2. retrieval_score: promedio de los scores de similitud (clipped a [0, 1])
    if retrieval_scores:
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        retrieval_score = min(max(avg_score, 0.0), 1.0)
    else:
        retrieval_score = 0.5

    # 3. consistency_score: basado en el consistency check
    consistency_score = consistency_result.score

    # 4. coverage_score: proporción de chunks citados sobre los recuperados
    cited_chunk_ids = {r.matched_chunk.get("chunk_id") for r in citation_results if r.matched_chunk}
    coverage_score = len(cited_chunk_ids) / max(total_retrieved, 1)

    # Score final ponderado (clipped a [0, 1])
    raw_final = (
        citation_score * 0.35 +
        retrieval_score * 0.30 +
        consistency_score * 0.25 +
        coverage_score * 0.10
    )
    final_score = min(max(raw_final, 0.0), 1.0)

    # Nivel de confianza
    if final_score >= 0.75:
        confidence_level = ConfidenceLevel.HIGH
    elif final_score >= 0.50:
        confidence_level = ConfidenceLevel.MEDIUM
    else:
        confidence_level = ConfidenceLevel.LOW

    return ScoreResult(
        citation_score=citation_score,
        retrieval_score=retrieval_score,
        consistency_score=consistency_score,
        coverage_score=coverage_score,
        final_score=final_score,
        confidence_level=confidence_level,
    )