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
from rag_lab.verification.consistency import ConsistencyResult, run_consistency_check
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


class TestConsistency:
    """Tests para el componente de consistency check."""

    def test_parse_response_valid(self):
        """Verificar parseo de respuesta válida."""
        from rag_lab.verification.consistency import _parse_response
        raw = "UNSUPPORTED: NO\nCONTRADICTIONS: NO\nHALLUCINATIONS: NO\nDETAILS:"
        result = _parse_response(raw)
        assert result is not None
        assert result["UNSUPPORTED"] == "NO"
        assert result["CONTRADICTIONS"] == "NO"
        assert result["HALLUCINATIONS"] == "NO"

    def test_parse_response_invalid(self):
        """Verificar que un parseo inválido devuelva None."""
        from rag_lab.verification.consistency import _parse_response
        raw = "ALGO: SI\nOTRA: NO"
        result = _parse_response(raw)
        assert result is None

    def test_compute_score_hallucinations(self):
        """Verificar score con alucinaciones."""
        from rag_lab.verification.consistency import _compute_score
        parsed = {"UNSUPPORTED": "NO", "CONTRADICTIONS": "NO", "HALLUCINATIONS": "SI", "DETAILS": "Hallucination detected"}
        assert _compute_score(parsed) == 0.0

    def test_compute_score_unsupported(self):
        """Verificar score con afirmaciones sin respaldo."""
        from rag_lab.verification.consistency import _compute_score
        parsed = {"UNSUPPORTED": "SI", "CONTRADICTIONS": "NO", "HALLUCINATIONS": "NO", "DETAILS": "Unsupported claim"}
        assert _compute_score(parsed) == 0.5

    def test_compute_score_ok(self):
        """Verificar score cuando todo está bien."""
        from rag_lab.verification.consistency import _compute_score
        parsed = {"UNSUPPORTED": "NO", "CONTRADICTIONS": "NO", "HALLUCINATIONS": "NO", "DETAILS": ""}
        assert _compute_score(parsed) == 1.0

    def test_run_consistency_check_mock(self):
        """Verificar run_consistency_check con LLM mock."""
        def mock_llm_call(prompt):
            return "UNSUPPORTED: NO\nCONTRADICTIONS: NO\nHALLUCINATIONS: NO\nDETAILS:"

        result = run_consistency_check(
            response="Respuesta de prueba",
            retrieved_chunks=[{"text": "Chunk 1"}, {"text": "Chunk 2"}],
            llm_call=mock_llm_call,
            max_retries=2,
        )
        assert result.parse_success == True
        assert result.score == 1.0
        assert result.has_hallucinations == False


class TestScoring:
    """Tests para el componente de scoring."""

    def test_score_all_valid(self):
        """Verificar scoring con todas las citas válidas."""
        consistency = ConsistencyResult(
            has_unsupported_claims=False,
            has_contradictions=False,
            has_hallucinations=False,
            details="",
            score=1.0,
            parse_success=True,
        )
        citation_results = [
            CitationResult(citation_text="[[1] ...", status=CitationStatus.VALID, matched_chunk={"chunk_id": "c1"}),
            CitationResult(citation_text="[[2] ...", status=CitationStatus.VALID, matched_chunk={"chunk_id": "c2"}),
        ]
        retrieval_scores = [0.9, 0.85, 0.8]
        score = calculate_score(citation_results, retrieval_scores, consistency, 3)
        assert score.citation_score == 1.0
        assert score.consistency_score == 1.0
        assert score.coverage_score == pytest.approx(2/3)
        assert score.final_score > 0.75
        assert score.confidence_level == ConfidenceLevel.HIGH

    def test_score_with_invalid(self):
        """Verificar scoring con citas inválidas."""
        consistency = ConsistencyResult(
            has_unsupported_claims=False,
            has_contradictions=False,
            has_hallucinations=False,
            details="",
            score=1.0,
            parse_success=True,
        )
        citation_results = [
            CitationResult(citation_text="[[1] ...", status=CitationStatus.VALID, matched_chunk={"chunk_id": "c1"}),
            CitationResult(citation_text="[[2] ...", status=CitationStatus.INVALID, matched_chunk=None),
        ]
        retrieval_scores = [0.5, 0.4]
        score = calculate_score(citation_results, retrieval_scores, consistency, 2)
        assert score.citation_score == 0.5
        assert score.confidence_level == ConfidenceLevel.MEDIUM

    def test_score_low_confidence(self):
        """Verificar scoring con baja confianza."""
        consistency = ConsistencyResult(
            has_unsupported_claims=False,
            has_contradictions=False,
            has_hallucinations=False,
            details="",
            score=1.0,
            parse_success=True,
        )
        citation_results = [
            CitationResult(citation_text="[[1] ...", status=CitationStatus.INVALID, matched_chunk=None),
        ]
        retrieval_scores = [0.2]
        score = calculate_score(citation_results, retrieval_scores, consistency, 5)
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