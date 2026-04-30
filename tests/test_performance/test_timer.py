"""Tests para el módulo de rendimiento."""

import json
import os
import pytest
from rag_lab.performance.timer import PhaseTimer
from rag_lab.performance.report import (
    generate_report,
    save_report_json,
    analyze_percentiles,
    _score_bar,
)


class TestPhaseTimer:
    """Tests para PhaseTimer."""

    def test_start_stop(self):
        timer = PhaseTimer()
        timer.start("test_phase")
        import time
        time.sleep(0.01)
        duration = timer.stop()
        assert duration > 0
        assert timer.get_duration("test_phase") > 0

    def test_multiple_phases(self):
        timer = PhaseTimer()
        timer.start("phase1")
        timer.stop()
        timer.start("phase2")
        timer.stop()
        durations = timer.get_all_durations()
        assert len(durations) == 2
        assert "phase1" in durations
        assert "phase2" in durations

    def test_total_duration(self):
        timer = PhaseTimer()
        timer.start("a")
        timer.stop()
        timer.start("b")
        timer.stop()
        total = timer.total_duration()
        assert total > 0

    def test_reset(self):
        timer = PhaseTimer()
        timer.start("x")
        timer.stop()
        assert timer.get_duration("x") > 0
        timer.reset()
        assert timer.get_all_durations() == {}

    def test_unknown_phase(self):
        timer = PhaseTimer()
        assert timer.get_duration("nonexistent") == 0.0


class TestScoreBar:
    """Tests para la función de barra visual."""

    def test_full_bar(self):
        bar = _score_bar(1.0)
        assert bar == "█" * 20

    def test_empty_bar(self):
        bar = _score_bar(0.0)
        assert bar == "░" * 20

    def test_half_bar(self):
        bar = _score_bar(0.5)
        assert bar == "█" * 10 + "░" * 10


class TestGenerateReport:
    """Tests para generate_report."""

    def test_empty_durations(self):
        result = generate_report({})
        assert "No hay métricas" in result

    def test_single_phase(self):
        durations = {"embedding": 1.5}
        result = generate_report(durations)
        assert "embedding" in result
        assert "1.50s" in result
        assert "█" in result

    def test_multiple_phases(self):
        durations = {
            "query_processing": 0.3,
            "embedding": 1.5,
            "hybrid_search": 0.5,
            "reranking": 2.0,
            "llm_generation": 12.0,
            "verification": 1.8,
        }
        result = generate_report(durations)
        assert "Métricas de rendimiento" in result
        assert "Total" in result
        assert "█" in result

    def test_total_calculation(self):
        durations = {"a": 1.0, "b": 2.0}
        result = generate_report(durations)
        assert "3.00s" in result


class TestSaveReportJson:
    """Tests para save_report_json."""

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "metrics.json")
        durations = {"test": 1.5}
        save_report_json(durations, path)
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["phases"] == durations
        assert data[0]["total"] == pytest.approx(1.5)

    def test_append(self, tmp_path):
        path = str(tmp_path / "metrics.json")
        save_report_json({"a": 1.0}, path)
        save_report_json({"b": 2.0}, path)

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2


class TestAnalyzePercentiles:
    """Tests para analyze_percentiles."""

    def test_no_file(self, tmp_path):
        result = analyze_percentiles(str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_percentiles(self, tmp_path):
        path = str(tmp_path / "metrics.json")
        # Create test data
        data = [
            {"timestamp": "2026-01-01T00:00:00", "phases": {"a": 1.0}, "total": 1.0},
            {"timestamp": "2026-01-02T00:00:00", "phases": {"b": 2.0}, "total": 2.0},
            {"timestamp": "2026-01-03T00:00:00", "phases": {"c": 3.0}, "total": 3.0},
            {"timestamp": "2026-01-04T00:00:00", "phases": {"d": 4.0}, "total": 4.0},
            {"timestamp": "2026-01-05T00:00:00", "phases": {"e": 5.0}, "total": 5.0},
        ]
        with open(path, "w") as f:
            json.dump(data, f)

        result = analyze_percentiles(path)
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["count"] == 5
        assert "p50" in result
        assert "p95" in result
        assert "p99" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
