"""Tests para la normalización de scores crudos del reranker."""

import pytest
from rag_lab.verification.pipeline import VerificationResult, _score_bar
from rag_lab.verification.scoring import calculate_score, ConfidenceLevel, ScoreResult
from rag_lab.verification.verifier import CitationResult, CitationStatus
from rag_lab.verification.consistency import ConsistencyResult


class TestScoreBarNormalization:
    """Tests para la normalización de barras en el bloque de verificación."""

    def test_bar_with_normalized_ratio(self):
        """Verificar que _score_bar funciona con ratios en [0, 1]."""
        assert _score_bar(1.0) == "█" * 10
        assert _score_bar(0.0) == "░" * 10
        assert _score_bar(0.5) == "█" * 5 + "░" * 5
        # Negative values are clamped to 0
        assert _score_bar(-0.5) == "░" * 10
        # Values > 1 are clamped to 1
        assert _score_bar(1.5) == "█" * 10

    def test_format_verification_block_normalizes_scores(self):
        """Verificar que format_verification_block normaliza scores crudos."""
        chunks = [
            {"doc_id": "doc1", "line_start": 10, "line_end": 20, "chunk_id": "c1"},
            {"doc_id": "doc2", "line_start": 50, "line_end": 60, "chunk_id": "c2"},
            {"doc_id": "doc1", "line_start": 30, "line_end": 40, "chunk_id": "c3"},
        ]
        scores = [6.06, 2.5, -0.3]  # logits crudos
        citation_results = []
        consistency_result = ConsistencyResult(
            has_unsupported_claims=False,
            has_contradictions=False,
            has_hallucinations=False,
            details="",
            score=1.0,
            parse_success=True,
        )
        score_result = ScoreResult(
            citation_score=0.5,
            retrieval_score=0.8,
            consistency_score=1.0,
            coverage_score=0.5,
            final_score=0.82,
            confidence_level=ConfidenceLevel.HIGH,
        )
        vr = VerificationResult(
            response="Test response",
            citation_results=citation_results,
            consistency_result=consistency_result,
            score_result=score_result,
            retrieved_chunks=chunks,
            retrieval_scores=scores,
        )
        block = vr.format_verification_block()

        # Verificar que las barras están presentes y no se desbordan
        assert "█" in block
        # With min-max normalization:
        # 6.06 normalizes to 1.0 (10.0/10), 2.5 → ~0.44, -0.3 → 0.0
        assert "/10" in block  # Scores shown in 0-10 scale
        assert "doc1" in block
        assert "doc2" in block

    def test_empty_scores_fallback(self):
        """Verificar fallback cuando no hay scores."""
        vr = VerificationResult(
            response="Test",
            citation_results=[],
            consistency_result=ConsistencyResult(
                has_unsupported_claims=False,
                has_contradictions=False,
                has_hallucinations=False,
                details="",
                score=1.0,
                parse_success=True,
            ),
            score_result=ScoreResult(
                citation_score=0.5,
                retrieval_score=0.5,
                consistency_score=1.0,
                coverage_score=0.0,
                final_score=0.55,
                confidence_level=ConfidenceLevel.MEDIUM,
            ),
            retrieved_chunks=[],
            retrieval_scores=[],
        )
        block = vr.format_verification_block()
        # Con scores vacíos, max_score = 1.0 (fallback)
        assert "Verificación de respuesta" in block


class TestScoringClipping:
    """Tests para el clipping de scores en calculate_score."""

    def test_retrieval_score_clipped(self):
        """Verificar que retrieval_score se clippea a [0, 1]."""
        scores = [6.06, 2.5, -0.3]
        result = calculate_score(
            citation_results=[],
            retrieval_scores=scores,
            consistency_result=ConsistencyResult(
                has_unsupported_claims=False,
                has_contradictions=False,
                has_hallucinations=False,
                details="",
                score=1.0,
                parse_success=True,
            ),
            total_retrieved=3,
        )
        # min-max: min=-0.3, max=6.06, range=6.36
        # normalized = [1.0, 0.44, 0.0] -> top-3 avg ≈ 0.48
        assert 0.4 < result.retrieval_score < 0.55

    def test_final_score_clipped(self):
        """Verificar que final_score se clippea a [0, 1]."""
        # Con retrieval_score = 1.0 y citation_score = 0.5 (sin citas)
        # final = 0.5*0.35 + 1.0*0.30 + 1.0*0.25 + 0.0*0.10 = 0.725
        result = calculate_score(
            citation_results=[],
            retrieval_scores=[6.06, 2.5, -0.3],
            consistency_result=ConsistencyResult(
                has_unsupported_claims=False,
                has_contradictions=False,
                has_hallucinations=False,
                details="",
                score=1.0,
                parse_success=True,
            ),
            total_retrieved=3,
        )
        assert 0.0 <= result.final_score <= 1.0

    def test_negative_scores(self):
        """Verificar que scores negativos se manejan correctamente."""
        result = calculate_score(
            citation_results=[],
            retrieval_scores=[-2.0, -1.5, -0.5],
            consistency_result=ConsistencyResult(
                has_unsupported_claims=False,
                has_contradictions=False,
                has_hallucinations=False,
                details="",
                score=0.5,
                parse_success=True,
            ),
            total_retrieved=3,
        )
        # min-max: min=-2.0, max=-0.5, range=1.5
        # normalized = [0.0, 0.33, 1.0] -> top-3 avg ≈ 0.44
        assert 0.35 < result.retrieval_score < 0.55
        assert 0.0 <= result.final_score <= 1.0

    def test_all_valid_citations(self):
        """Verificar que con todas las citas válidas, el score es alto."""
        citation_results = [
            CitationResult(
                citation_text="SDMX es un estándar",
                matched_chunk={"chunk_id": "c1", "text": "SDMX es un estándar de intercambio"},
                status=CitationStatus.VALID,
            )
        ]
        result = calculate_score(
            citation_results=citation_results,
            retrieval_scores=[0.9, 0.8, 0.7],
            consistency_result=ConsistencyResult(
                has_unsupported_claims=False,
                has_contradictions=False,
                has_hallucinations=False,
                details="",
                score=1.0,
                parse_success=True,
            ),
            total_retrieved=3,
        )
        # citation_score = 1.0, retrieval_score = 0.833, consistency = 1.0, coverage = 0.333
        assert result.citation_score == 1.0
        assert result.final_score > 0.75
        assert result.confidence_level == ConfidenceLevel.HIGH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
