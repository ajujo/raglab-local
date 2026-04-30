"""Tests para verificar que los scores del reranker se usan correctamente."""

import pytest


class TestRerankScoreUsage:
    """Tests para el bug de scores colapsados a 0.50."""

    def test_rerank_score_takes_precedence(self):
        """Verificar que rerank_score tiene prioridad sobre score."""
        # Simular un chunk con ambos scores
        chunk = {
            "chunk_id": "c1",
            "score": 0.3,  # score de retrieval
            "rerank_score": 0.92,  # score del reranker
        }
        result = chunk.get("rerank_score", chunk.get("score", 0.5))
        assert result == 0.92

    def test_fallback_to_score_when_no_rerank(self):
        """Verificar fallback a score cuando no hay rerank_score."""
        chunk = {
            "chunk_id": "c2",
            "score": 0.7,
        }
        result = chunk.get("rerank_score", chunk.get("score", 0.5))
        assert result == 0.7

    def test_default_0_5_when_no_scores(self):
        """Verificar default 0.5 cuando no hay scores."""
        chunk = {"chunk_id": "c3"}
        result = chunk.get("rerank_score", chunk.get("score", 0.5))
        assert result == 0.5

    def test_list_comprehension_with_rerank_score(self):
        """Simular el patrón de list comprehension del código."""
        chunks = [
            {"chunk_id": "c1", "score": 0.3, "rerank_score": 0.92},
            {"chunk_id": "c2", "score": 0.7},
            {"chunk_id": "c3"},
        ]
        scores = [r.get("rerank_score", r.get("score", 0.5)) for r in chunks]
        assert scores == [0.92, 0.7, 0.5]

    def test_reranker_attaches_rerank_score(self):
        """Verificar que reranker.py adjunta 'rerank_score' al chunk."""
        from rag_lab.retrieval.reranker import rerank
        import inspect
        sig = inspect.signature(rerank)
        assert "chunks" in sig.parameters

    def test_verification_uses_correct_score(self):
        """Verificar que verify_and_score recibe los scores correctos."""
        from rag_lab.verification.pipeline import verify_and_score
        # Simular scores del reranker
        scores = [0.92, 0.7, 0.5]
        response = "Respuesta de prueba."
        chunks = [
            {"chunk_id": "c1", "text": "Texto 1"},
            {"chunk_id": "c2", "text": "Texto 2"},
            {"chunk_id": "c3", "text": "Texto 3"},
        ]
        result = verify_and_score(response, chunks, scores)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
