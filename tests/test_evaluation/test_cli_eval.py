"""Smoke tests for the rag-lab eval CLI commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from rag_lab.cli import app
from rag_lab.evaluation.types import EvalResult, EvalSample


runner = CliRunner()


def _make_sample(qid: str) -> EvalSample:
    return EvalSample.from_yaml_entry(
        {"id": qid, "text": "What is SDMX?", "suite": "official",
         "validated": True, "doc_relevance": {"SDMX_Glossary": 3}}
    )


def _make_result(sample: EvalSample) -> EvalResult:
    return EvalResult(
        sample=sample,
        answer="SDMX is a standard.",
        contexts=["chunk text"],
        context_metadata=[{"chunk_id": "c1", "doc_id": "SDMX_Glossary",
                           "heading_path": "/h1", "rerank_score": 0.9}],
        citations=[{"chunk_id": "c1", "doc_id": "SDMX_Glossary",
                   "lines": "1-10", "status": "valid"}],
        trust_score=0.85,
        trust_level="HIGH",
        latency_ms=200,
    )


class TestCliEvalRun:
    def test_eval_run_smoke(self, tmp_path):
        out = tmp_path / "smoke.jsonl"
        samples = [_make_sample("q001"), _make_sample("q002")]

        def write_fake_output(samples, output_path, **kwargs):
            with open(output_path, "w") as f:
                for s in samples:
                    f.write(_make_result(s).to_jsonl_line() + "\n")
            return output_path

        with (
            patch("rag_lab.evaluation.dataset.load_eval_samples", return_value=samples),
            patch("rag_lab.evaluation.e2e_runner.run_eval", side_effect=write_fake_output),
        ):
            result = runner.invoke(
                app,
                ["eval", "run", "--suite", "official", "--output", str(out)],
            )

        assert result.exit_code == 0, result.output
        lines = [json.loads(l) for l in out.read_text().strip().split("\n")]
        assert len(lines) == 2

    def test_eval_run_with_limit(self, tmp_path):
        out = tmp_path / "limited.jsonl"
        samples = [_make_sample(f"q{i:03d}") for i in range(10)]

        captured = {}

        def write_fake(samples, output_path, **kwargs):
            captured["n"] = len(samples)
            with open(output_path, "w") as f:
                for s in samples:
                    f.write(_make_result(s).to_jsonl_line() + "\n")

        with (
            patch("rag_lab.evaluation.dataset.load_eval_samples", return_value=samples),
            patch("rag_lab.evaluation.e2e_runner.run_eval", side_effect=write_fake),
        ):
            runner.invoke(
                app,
                ["eval", "run", "--limit", "3", "--output", str(out)],
            )

        assert captured["n"] == 3

    def test_eval_list_empty(self, tmp_path):
        with patch("rag_lab.cli_eval.EVAL_OUTPUT_DIR", tmp_path):
            result = runner.invoke(app, ["eval", "list"])
        assert result.exit_code == 0

    def test_eval_show_not_found(self, tmp_path):
        with patch("rag_lab.cli_eval.EVAL_OUTPUT_DIR", tmp_path):
            result = runner.invoke(app, ["eval", "show", "nonexistent_run"])
        assert result.exit_code != 0
