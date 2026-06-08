"""Tests for EvalRunner and run_eval.

Mocks the full pipeline stack so no LLM, no GPU, no live stores needed.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_lab.evaluation.e2e_runner import EvalRunner, run_eval
from rag_lab.evaluation.types import EvalResult, EvalSample


def _make_sample(qid: str = "q001", question: str = "What is SDMX?") -> EvalSample:
    return EvalSample.from_yaml_entry(
        {
            "id": qid,
            "text": question,
            "language": "en",
            "category": "glossary_definition",
            "doc_relevance": {"SDMX_Glossary": 3},
        }
    )


def _make_verification_mock():
    from rag_lab.verification.verifier import CitationStatus

    cit = MagicMock()
    cit.chunk_id = "chunk_abc"
    cit.status = CitationStatus.VALID
    cit.matched_chunk = {"doc_id": "SDMX_Glossary", "line_start": 1, "line_end": 10}

    score = MagicMock()
    score.final_score = 0.85
    score.confidence_level.value = "HIGH"

    ver = MagicMock()
    ver.citation_results = [cit]
    ver.score_result = score
    return ver


def _mock_chunk():
    return {
        "chunk_id": "chunk_abc",
        "doc_id": "SDMX_Glossary",
        "heading_path": "## Glossary",
        "text": "SDMX is a standard for data exchange.",
        "rerank_score": 0.9,
        "rrf_score": 0.8,
    }


@pytest.fixture(autouse=True)
def mock_stores():
    with (
        patch("rag_lab.evaluation.e2e_runner.VectorStore") as vs,
        patch("rag_lab.evaluation.e2e_runner.FTSStore") as fs,
        patch("rag_lab.evaluation.e2e_runner.DocStore") as ds,
    ):
        vs.return_value.initialize.return_value = None
        fs.return_value.initialize.return_value = None
        ds.return_value.initialize.return_value = None
        ds.return_value.close.return_value = None
        fs.return_value.close.return_value = None
        yield


@pytest.fixture(autouse=True)
def mock_pipeline():
    import numpy as np

    with (
        patch("rag_lab.evaluation.e2e_runner.process_query") as pq,
        patch("rag_lab.evaluation.e2e_runner.encode_chunks") as ec,
        patch("rag_lab.evaluation.e2e_runner.hybrid_search") as hs,
        patch("rag_lab.evaluation.e2e_runner.rerank") as rr,
        patch("rag_lab.evaluation.e2e_runner.build_prompt") as bp,
        patch("rag_lab.evaluation.e2e_runner.generate_response") as gr,
        patch("rag_lab.evaluation.e2e_runner.verify_and_score") as vs,
    ):
        pq.return_value = [{"text": "What is SDMX?", "use_for_sparse": True}]
        ec.return_value = (np.array([[0.1] * 128]), {"chunk_abc": {"1": 0.5}})
        hs.return_value = [_mock_chunk()]
        rr.return_value = [_mock_chunk()]
        bp.return_value = ("sys prompt", "user prompt")
        gr.return_value = "SDMX is a standard."
        vs.return_value = _make_verification_mock()
        yield


class TestEvalRunnerSingle:
    def test_run_single_returns_eval_result(self):
        runner = EvalRunner()
        result = runner.run_single(_make_sample())
        assert isinstance(result, EvalResult)

    def test_run_single_captures_answer(self):
        runner = EvalRunner()
        result = runner.run_single(_make_sample())
        assert result.answer == "SDMX is a standard."

    def test_run_single_captures_contexts(self):
        runner = EvalRunner()
        result = runner.run_single(_make_sample())
        assert len(result.contexts) == 1
        assert "SDMX" in result.contexts[0]

    def test_run_single_captures_trust_score(self):
        runner = EvalRunner()
        result = runner.run_single(_make_sample())
        assert result.trust_score == pytest.approx(0.85)
        assert result.trust_level == "HIGH"

    def test_run_single_latency_positive(self):
        runner = EvalRunner()
        result = runner.run_single(_make_sample())
        assert result.latency_ms >= 0

    def test_run_single_no_error_on_success(self):
        runner = EvalRunner()
        result = runner.run_single(_make_sample())
        assert result.error is None

    def test_run_single_captures_error_on_llm_failure(self):
        with patch("rag_lab.evaluation.e2e_runner.generate_response") as gr:
            gr.side_effect = Exception("LLM timeout")
            runner = EvalRunner()
            result = runner.run_single(_make_sample())
        assert result.error is not None
        assert "LLM timeout" in result.error
        assert result.answer == ""


class TestRunEval:
    def test_run_eval_writes_jsonl(self, tmp_path):
        out = tmp_path / "test_run.jsonl"
        samples = [_make_sample("q001"), _make_sample("q002", "Second question")]
        run_eval(samples, output_path=out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_run_eval_output_schema(self, tmp_path):
        out = tmp_path / "test_run.jsonl"
        run_eval([_make_sample()], output_path=out)

        row = json.loads(out.read_text().strip())
        required = {
            "query_id", "question", "language", "category",
            "answer", "contexts", "context_metadata", "citations",
            "trust_score", "trust_level", "latency_ms",
            "expected_answer", "expected_doc_ids", "doc_relevance", "error",
        }
        assert required == set(row.keys())

    def test_run_eval_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "run.jsonl"
        run_eval([_make_sample()], output_path=out)
        assert out.exists()

    def test_run_eval_partial_run_preserved(self, tmp_path):
        """If a query fails mid-run, previous lines are not lost."""
        out = tmp_path / "partial.jsonl"
        samples = [_make_sample("q001"), _make_sample("q002")]

        call_count = 0

        def failing_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("simulated failure")
            return "ok"

        with patch("rag_lab.evaluation.e2e_runner.generate_response", side_effect=failing_generate):
            run_eval(samples, output_path=out)

        lines = [json.loads(l) for l in out.read_text().strip().split("\n")]
        assert len(lines) == 2
        assert lines[0]["error"] is None
        assert lines[1]["error"] is not None
