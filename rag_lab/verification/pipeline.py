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

_SCORE_BAR_WIDTH = 10


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize raw reranker logits to [0, 1] via min-max scaling.

    Best score → 1.0, worst → 0.0. Works with any logit range.
    """
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    min_s = min(scores)
    max_s = max(scores)
    score_range = max_s - min_s
    if score_range == 0:
        return [1.0] * len(scores)
    return [(s - min_s) / score_range for s in scores]


def _score_bar(score: float, width: int = _SCORE_BAR_WIDTH) -> str:
    """Generate a visual bar for a score between 0 and 1.

    Args:
        score: Value clamped to [0, 1].
        width: Total bar width in characters.
    """
    clamped = max(0.0, min(1.0, score))
    filled = round(clamped * width)
    empty = width - filled
    return "█" * filled + "░" * empty


class VerificationResult:
    """Resultado completo de la capa de verificación."""

    def __init__(
        self,
        response: str,
        citation_results: List[CitationResult],
        consistency_result: ConsistencyResult,
        score_result: ScoreResult,
        retrieved_chunks: List[dict],
        retrieval_scores: List[float],
    ):
        self.response = response
        self.citation_results = citation_results
        self.consistency_result = consistency_result
        self.score_result = score_result
        self.retrieved_chunks = retrieved_chunks
        self.retrieval_scores = retrieval_scores

    def get_warnings(self) -> List[str]:
        """Obtener advertencias de citas inválidas y scores bajos."""
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

        # Advertencia por scores bajos — basada en scores normalizados.
        # Solo se activa cuando el MEJOR chunk normalizado está por debajo
        # de 0.3, indicando que genuinamente no se encontró nada relevante.
        # Con min-max, el top-1 siempre es 1.0 si hay spread entre scores,
        # así que esto solo se activa cuando todos los scores son casi iguales
        # (spread < 0.5) — es decir, el reranker no pudo distinguir relevancia.
        if len(self.retrieval_scores) >= 2:
            max_s = max(self.retrieval_scores)
            min_s = min(self.retrieval_scores)
            spread = max_s - min_s
            # If the spread is very small, the reranker couldn't differentiate
            # relevance — all chunks look equally (ir)relevant
            if spread < 0.5:
                warnings.append(
                    "⚠ Algunos fragmentos tienen relevancia baja. Considera reformular la pregunta "
                    "o activar HyDE con --hyde para mejorar la recuperación."
                )

        return warnings

    def format_verification_block(self) -> str:
        """Formatear el bloque de metadatos de verificación con scores por chunk."""
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

        # Normalizar scores para visualización (logits crudos → escala 0-10)
        norm_scores = _normalize_scores(self.retrieval_scores)

        # Construir sección de fragmentos recuperados
        chunks_lines = []
        for i, (chunk, raw_sc, norm_sc) in enumerate(
            zip(self.retrieved_chunks, self.retrieval_scores, norm_scores)
        ):
            doc_id = chunk.get("doc_id", "desconocido")
            line_start = chunk.get("line_start", "?")
            line_end = chunk.get("line_end", "?")
            display_score = norm_sc * 10  # Scale to 0-10
            bar = _score_bar(norm_sc)
            chunks_lines.append(
                f"  [{i+1}] {doc_id} | Líneas {line_start}-{line_end}  {display_score:4.1f}/10 {bar}"
            )

        chunks_section = "\n".join(chunks_lines)

        block = "─" * 45
        block += "\nVerificación de respuesta\n"
        block += "  Fragmentos recuperados:\n"
        block += chunks_section + "\n"
        block += f"  Citas verificadas : {valid_citations}/{total_citations} {('✓' if total_citations == valid_citations else '⚠')}\n"
        block += f"  Consistencia      : {consistency_status}\n"
        block += f"  Score de confianza: {score.final_score:.2f} — {score.confidence_level.value} {confidence_emoji.get(score.confidence_level.value, '')}\n"
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
            llm_call=lambda prompt: generate_response("", prompt),
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
        retrieved_chunks=retrieved_chunks,
        retrieval_scores=retrieval_scores,
    )
