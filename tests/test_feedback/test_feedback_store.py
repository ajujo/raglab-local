"""Tests para el módulo de feedback."""

import os
import tempfile
import pytest
from rag_lab.feedback.feedback_store import (
    FeedbackEntry,
    init_db,
    save_feedback,
    load_feedback,
)


class TestFeedbackStore:
    """Tests para el almacenamiento de feedback."""

    def test_init_db(self, tmp_path):
        init_db(tmp_path)
        assert os.path.exists(tmp_path)

    def test_save_and_load(self, tmp_path):
        init_db(tmp_path)
        entry = FeedbackEntry(
            question="¿Qué es SDMX?",
            rewritten_query=None,
            hyde_used=False,
            chunks_retrieved='[{"doc_id": "doc1", "line_start": 10, "line_end": 20, "retrieval_score": 0.9}]',
            final_score=0.78,
            score_level="HIGH",
            useful=True,
            timestamp="2026-04-30T12:00:00",
        )
        save_feedback(entry, tmp_path)
        entries = load_feedback(tmp_path)
        assert len(entries) == 1
        assert entries[0].question == "¿Qué es SDMX?"
        assert entries[0].useful == True
        assert entries[0].score_level == "HIGH"

    def test_save_multiple(self, tmp_path):
        init_db(tmp_path)
        for i in range(5):
            entry = FeedbackEntry(
                question=f"Pregunta {i}",
                rewritten_query=None,
                hyde_used=False,
                chunks_retrieved="[]",
                final_score=0.75 + i * 0.05,
                score_level="HIGH" if i < 3 else "MEDIUM",
                useful=i % 2 == 0,
                timestamp="2026-04-30T12:00:00",
            )
            save_feedback(entry, tmp_path)
        entries = load_feedback(tmp_path)
        assert len(entries) == 5

    def test_empty_load(self, tmp_path):
        init_db(tmp_path)
        entries = load_feedback(tmp_path)
        assert len(entries) == 0

    def test_feedback_entry_to_dict(self):
        entry = FeedbackEntry(
            question="Test",
            rewritten_query=None,
            hyde_used=True,
            chunks_retrieved="[]",
            final_score=0.8,
            score_level="HIGH",
            useful=True,
            timestamp="2026-04-30T12:00:00",
        )
        d = entry.to_dict()
        assert d["question"] == "Test"
        assert d["useful"] == True
        assert d["hyde_used"] == True


class TestAnalyzeFeedback:
    """Tests para el análisis de feedback."""

    def test_analyze_empty(self, tmp_path, capsys):
        init_db(tmp_path)
        from rag_lab.feedback.analyze_feedback import analyze
        analyze(tmp_path)
        captured = capsys.readouterr()
        assert "No hay entradas de feedback registradas." in captured.out

    def test_analyze_with_data(self, tmp_path, capsys):
        init_db(tmp_path)
        # Save some entries
        for i, (useful, score, level) in enumerate([
            (True, 0.85, "HIGH"),
            (True, 0.80, "HIGH"),
            (False, 0.50, "MEDIUM"),
        ]):
            save_feedback(FeedbackEntry(
                question=f"Q{i}",
                rewritten_query=None,
                hyde_used=False,
                chunks_retrieved='[{"doc_id": "doc1", "heading_path": "Sec1", "line_start": 10, "line_end": 20, "retrieval_score": 0.9}]',
                final_score=score,
                score_level=level,
                useful=useful,
                timestamp="2026-04-30T12:00:00",
            ), tmp_path)

        from rag_lab.feedback.analyze_feedback import analyze
        analyze(tmp_path)
        captured = capsys.readouterr()
        assert "Total de respuestas evaluadas: 3" in captured.out
        assert "Útiles  : 2" in captured.out
        assert "No útiles: 1" in captured.out


@pytest.fixture
def tmp_path(tmp_path):
    """Provide a temp directory path for the SQLite DB."""
    return str(tmp_path / "feedback_test.db")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
