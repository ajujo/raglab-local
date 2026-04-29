"""Tests para la Verification Layer.

Verifica los tres componentes: verifier, consistency, y scoring.
"""

import pytest
from rag_lab.verification.verifier import (
    verify_citations_layer,
    CitationResult,
    CitationStatus,
)
from rag_lab.verification.scoring import calculate_score, ScoreResult, ConfidenceLevel
from rag_lab.verification.pipeline import verify_and_score, VerificationResult


class TestVerifier:
    """Tests para el componente de verificación de citas."""

    def test_verify_valid_citation(self):
        """Verificar que una cita válida se clasifique correctamente."""
        response = "SDMX es un estándar [[1] Fuente: doc1 | Sección: Section 1 | Líneas: 10-20]"
        chunks = [
            {"chunk_id": "c1", "doc_id": "doc1", "heading_path": "Section 1", "line_start": 10, "line_end": 20}
        ]
        results = verify_citations_layer(response, chunks)
        assert len(results) == 1
        assert results[0].status == CitationStatus.VALID
        assert results[0].matched_chunk is not None

    def test_verify_invalid_citation(self):
        """Verificar que una cita inválida se detecte."""
        response = "Algo [[1] Fuente: doc_missing | Sección: Section X | Líneas: 99-100]"
        chunks = [
            {"chunk_id": "c1", "doc_id": "doc1", "heading_path": "Section 1", "line_start": 10, "line_end": 20}
        ]
        results = verify_citations_layer(response, chunks)
        assert len(results) == 1
        assert results[0].status == CitationStatus.INVALID
        assert results[0].matched_chunk is None

    def test_verify_partial_citation(self):
        """Verificar que una cita parcial se clasifique como PARTIAL."""
        response = "Dato [[1] Fuente: doc1 | Sección: Section 1 | Líneas: 10-30]"
        chunks = [
            {"chunk_id": "c1", "doc_id": "doc1", "heading_path": "Section 1", "line_start": 10, "line_end": 20}
        ]
        results = verify_citations_layer(response, chunks)
        assert len(results) == 1
        assert results[0].status == CitationStatus.PARTIAL

    def test_no_citations(self):
        """Verificar que una respuesta sin citas devuelva lista vacía."""
        response = "Esta respuesta no tiene citas."
        chunks = [
            {"chunk_id": "c1", "doc_id": "doc1", "heading_path": "Section 1", "line_start": 10, "line_end": 20}
        ]
        results = verify_citations_layer(response, chunks)
        assert len(results) == 0


class TestScoring:
    """Tests para el componente de scoring."""

    def test_score_all_valid(self):
        """Verificar scoring con todas las citas válidas."""
        citation_results = [
            CitationResult(citation_text="[[1] ...", status=CitationStatus.VALID, matched_chunk={"chunk_id": "c1"}),
            CitationResult(citation_text="[[2] ...", status=CitationStatus.VALID, matched_chunk={"chunk_id": "c2"}),
        ]
        retrieval_scores = [0.9, 0.85, 0.8]
        score = calculate_score(citation_results, retrieval_scores, None, 3)
        assert score.citation_score == 1.0
        assert score.consistency_score == 1.0
        assert score.coverage_score == pytest.approx(2/3)
        assert score.final_score > 0.75
        assert score.confidence_level == ConfidenceLevel.HIGH

    def test_score_with_invalid(self):
        """Verificar scoring con citas inválidas."""
        citation_results = [
            CitationResult(citation_text="[[1] ...", status=CitationStatus.VALID, matched_chunk={"chunk_id": "c1"}),
            CitationResult(citation_text="[[2] ...", status=CitationStatus.INVALID, matched_chunk=None),
        ]
        retrieval_scores = [0.5, 0.4]
        score = calculate_score(citation_results, retrieval_scores, None, 2)
        assert score.citation_score == 0.5
        assert score.confidence_level == ConfidenceLevel.MEDIUM

    def test_score_low_confidence(self):
        """Verificar scoring con baja confianza."""
        citation_results = [
            CitationResult(citation_text="[[1] ...", status=CitationStatus.INVALID, matched_chunk=None),
        ]
        retrieval_scores = [0.2]
        score = calculate_score(citation_results, retrieval_scores, None, 5)
        assert score.confidence_level == ConfidenceLevel.LOW


class TestPipeline:
    """Tests para el pipeline de verificación."""

    def test_pipeline_basic(self):
        """Verificar el pipeline completo."""
        response = "Respuesta [[1] Fuente: doc1 | Sección: Sec1 | Líneas: 10-20]"
        chunks = [
            {"chunk_id": "c1", "doc_id": "doc1", "heading_path": "Sec1", "line_start": 10, "line_end": 20},
            {"chunk_id": "c2", "doc_id": "doc1", "heading_path": "Sec2", "line_start": 30, "line_end": 40},
        ]
        retrieval_scores = [0.9, 0.8]

        result = verify_and_score(response, chunks, retrieval_scores, enable_consistency_check=False)

        assert isinstance(result, VerificationResult)
        assert len(result.citation_results) == 1
        assert result.score_result.final_score > 0
        assert result.format_verification_block() is not None

    def test_pipeline_warnings(self):
        """Verificar que las advertencias se generen correctamente."""
        response = "Respuesta [[1] Fuente: doc_missing | Sección: SecX | Líneas: 99-100]"
        chunks = [
            {"chunk_id": "c1", "doc_id": "doc1", "heading_path": "Sec1", "line_start": 10, "line_end": 20},
        ]
        retrieval_scores = [0.9]

        result = verify_and_score(response, chunks, retrieval_scores, enable_consistency_check=False)
        warnings = result.get_warnings()
        assert len(warnings) == 1
        assert "inválida" in warnings[0].lower() or "invalid" in warnings[0].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])