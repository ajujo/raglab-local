"""Pipeline de verificación: orquesta los tres componentes.

Combina la verificación de citas, el consistency check y el scoring
en una única función que se ejecuta después de la generación del LLM.
"""

import logging
from typing import List

from rag_lab.generation.llm_client import generate_response
from rag_lab.verification.verifier import CitationResult, verify_citations_layer, CitationStatus
from rag_lab.verification.consistency import run_consistency_check, ConsistencyResult
from rag_lab.verification.scoring import calculate_score, ScoreResult, ConfidenceLevel

logger = logging.getLogger("rag_lab")


class VerificationResult:
    """Resultado completo de la capa de verificación."""

    def __init__(
        self,
        response: str,
        citation_results: List[CitationResult],
        consistency_result: ConsistencyResult,
        score_result: ScoreResult,
    ):
        self.response = response
        self.citation_results = citation_results
        self.consistency_result = consistency_result
        self.score_result = score_result

    def get_warnings(self) -> List[str]:
        """Obtener advertencias de citas inválidas."""
        warnings = []
        for result in self.citation_results:
            if result.status == CitationStatus.INVALID:
                warnings.append(f"Cita inválida: {result.citation_text}")

        # Advertencias del consistency check
        cr = self.consistency_result
        if cr.parse_success:
            if cr.has_hallucinations:
                warnings.append("⚠ Se detectaron posibles alucinaciones en la respuesta.")
            elif cr.has_unsupported_claims or cr.has_contradictions:
                warnings.append("⚠ Algunas afirmaciones pueden no estar respaldadas por los fragmentos.")
        else:
            warnings.append("⚠ Consistency check no pudo ejecutarse correctamente.")

        return warnings

    def format_verification_block(self) -> str:
        """Formatear el bloque de metadatos de verificación."""
        valid_citations = sum(1 for r in self.citation_results if r.status == CitationStatus.VALID)
        total_citations = len(self.citation_results)

        cr = self.consistency_result
        if cr.parse_success:
            if cr.has_hallucinations:
                consistency_status = "ALUCINACIONES ⚠"
            elif cr.has_unsupported_claims or cr.has_contradictions:
                consistency_status = "WARN ⚠"
            else:
                consistency_status = "OK ✓"
        else:
            consistency_status = "N/A"

        score = self.score_result
        confidence_emoji = {"HIGH": "✓", "MEDIUM": "⚠", "LOW": "✗"}

        block = "─" * 45
        block += "\nVerificación de respuesta\n"
        block += f"Citas verificadas : {valid_citations}/{total_citations} {('✓' if total_citations == valid_citations else '⚠')}\n"
        block += f"Consistencia      : {consistency_status}\n"
        block += f"Score de confianza: {score.final_score:.2f} — {score.confidence_level.value} {confidence_emoji.get(score.confidence_level.value, '')}\n"
        block += "─" * 45

        return block


def verify_and_score(
    response: str,
    retrieved_chunks: List[dict],
    retrieval_scores: List[float],
    enable_consistency_check: bool = True,
) -> VerificationResult:
    """Orquesta los tres componentes de la capa de verificación.

    Args:
        response: La respuesta generada por el LLM.
        retrieved_chunks: Lista de chunks recuperados con sus metadatos.
        retrieval_scores: Scores de similitud de cada chunk recuperado.
        enable_consistency_check: Si es True, se ejecuta el consistency check.

    Returns:
        VerificationResult con todos los resultados de la verificación.
    """
    # Componente 1: Verificación de citas
    citation_results = verify_citations_layer(response, retrieved_chunks)
    logger.info(f"Verificación de citas: {len(citation_results)} citas encontradas")

    # Componente 2: Consistency check
    if enable_consistency_check:
        consistency_result = run_consistency_check(
            response=response,
            retrieved_chunks=retrieved_chunks,
            llm_call=lambda prompt: generate_response(prompt),
            max_retries=2,
        )
        logger.info(f"Consistency check: parse_success={consistency_result.parse_success}, score={consistency_result.score}")
    else:
        # Si está desactivado, usar score neutro
        consistency_result = ConsistencyResult(
            has_unsupported_claims=False,
            has_contradictions=False,
            has_hallucinations=False,
            details="",
            score=1.0,
            parse_success=True,
        )
        logger.info("Consistency check desactivado")

    # Componente 3: Scoring
    score_result = calculate_score(
        citation_results=citation_results,
        retrieval_scores=retrieval_scores,
        consistency_result=consistency_result,
        total_retrieved=len(retrieved_chunks),
    )
    logger.info(f"Score final: {score_result.final_score:.2f} ({score_result.confidence_level.value})")

    return VerificationResult(
        response=response,
        citation_results=citation_results,
        consistency_result=consistency_result,
        score_result=score_result,
    )