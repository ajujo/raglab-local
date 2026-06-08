"""Tests for the dataset loader."""

import json

import pytest

from rag_lab.evaluation.dataset import EvaluationError, load_eval_samples
from rag_lab.evaluation.types import EvalSample


@pytest.fixture
def minimal_yaml(tmp_path):
    f = tmp_path / "queries.yaml"
    f.write_text(
        "queries:\n"
        "  - id: q001\n"
        "    text: 'What is SDMX?'\n"
        "    suite: official\n"
        "    validated: true\n"
        "    language: en\n"
        "    category: glossary_definition\n"
        "    doc_relevance:\n"
        "      SDMX_Glossary: 3\n"
        "  - id: q002\n"
        "    text: 'Not validated'\n"
        "    suite: official\n"
        "    validated: false\n"
        "    doc_relevance: {}\n"
        "  - id: q003\n"
        "    text: 'Candidate query'\n"
        "    suite: candidate\n"
        "    validated: true\n"
        "    doc_relevance: {}\n"
    )
    return f


@pytest.fixture
def minimal_json(tmp_path):
    f = tmp_path / "queries.json"
    data = [
        {"id": "q001", "text": "X?", "suite": "official", "validated": True, "doc_relevance": {}},
    ]
    f.write_text(json.dumps(data))
    return f


class TestLoadEvalSamples:
    def test_loads_official_validated(self, minimal_yaml):
        samples = load_eval_samples(suite="official", path=minimal_yaml)
        assert len(samples) == 1
        assert samples[0].query_id == "q001"

    def test_excludes_not_validated(self, minimal_yaml):
        samples = load_eval_samples(suite="official", path=minimal_yaml, validated_only=True)
        ids = {s.query_id for s in samples}
        assert "q002" not in ids

    def test_includes_not_validated_when_flag_off(self, minimal_yaml):
        samples = load_eval_samples(suite="official", path=minimal_yaml, validated_only=False)
        ids = {s.query_id for s in samples}
        assert "q002" in ids

    def test_filters_by_suite(self, minimal_yaml):
        samples = load_eval_samples(suite="candidate", path=minimal_yaml, validated_only=False)
        assert len(samples) == 1
        assert samples[0].query_id == "q003"

    def test_returns_eval_samples(self, minimal_yaml):
        samples = load_eval_samples(suite="official", path=minimal_yaml)
        assert all(isinstance(s, EvalSample) for s in samples)

    def test_loads_json(self, minimal_json):
        samples = load_eval_samples(suite="official", path=minimal_json, validated_only=False)
        assert len(samples) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_eval_samples(path=tmp_path / "nonexistent.yaml")

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "q.txt"
        f.write_text("hello")
        with pytest.raises(EvaluationError, match="Unsupported"):
            load_eval_samples(path=f)

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text("queries: []\n")
        with pytest.raises(EvaluationError, match="No queries"):
            load_eval_samples(path=f)
