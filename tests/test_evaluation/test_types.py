"""Tests for EvalSample and EvalResult types."""

import json

import pytest

from rag_lab.evaluation.types import EvalResult, EvalSample


class TestEvalSample:
    def test_from_yaml_entry_full(self):
        entry = {
            "id": "q001",
            "text": "What is SDMX?",
            "language": "en",
            "category": "glossary_definition",
            "doc_relevance": {"SDMX_Glossary": 3, "SDMX-Training": 2, "noise": 1},
        }
        s = EvalSample.from_yaml_entry(entry)
        assert s.query_id == "q001"
        assert s.question == "What is SDMX?"
        assert s.language == "en"
        assert s.category == "glossary_definition"
        assert s.expected_answer is None
        assert set(s.expected_doc_ids) == {"SDMX_Glossary", "SDMX-Training"}
        assert s.doc_relevance["SDMX_Glossary"] == 3

    def test_from_yaml_entry_minimal(self):
        entry = {"id": "q002", "text": "hello"}
        s = EvalSample.from_yaml_entry(entry)
        assert s.query_id == "q002"
        assert s.question == "hello"
        assert s.language is None
        assert s.category is None
        assert s.expected_doc_ids == []

    def test_from_yaml_entry_with_expected_answer(self):
        entry = {
            "id": "q003",
            "text": "Define SDMX",
            "expected_answer": "SDMX stands for...",
            "doc_relevance": {},
        }
        s = EvalSample.from_yaml_entry(entry)
        assert s.expected_answer == "SDMX stands for..."

    def test_expected_doc_ids_threshold(self):
        entry = {
            "id": "q004",
            "text": "test",
            "doc_relevance": {"A": 3, "B": 2, "C": 1, "D": 0},
        }
        s = EvalSample.from_yaml_entry(entry)
        assert set(s.expected_doc_ids) == {"A", "B"}


class TestEvalResult:
    def _make_sample(self) -> EvalSample:
        return EvalSample.from_yaml_entry(
            {"id": "q001", "text": "What is SDMX?", "doc_relevance": {"DocA": 3}}
        )

    def test_to_jsonl_dict_schema(self):
        result = EvalResult(
            sample=self._make_sample(),
            answer="SDMX is a standard.",
            contexts=["chunk1", "chunk2"],
            context_metadata=[{"chunk_id": "c1", "doc_id": "DocA", "heading_path": "/h1", "rerank_score": 0.9}],
            citations=[{"chunk_id": "c1", "doc_id": "DocA", "lines": "10-20", "status": "valid"}],
            trust_score=0.85,
            trust_level="HIGH",
            latency_ms=350,
        )
        d = result.to_jsonl_dict()

        required_keys = {
            "query_id", "question", "language", "category",
            "answer", "answer_for_eval", "contexts", "context_metadata", "citations",
            "trust_score", "trust_level", "latency_ms",
            "expected_answer", "expected_doc_ids", "doc_relevance", "error",
        }
        assert required_keys == set(d.keys())

        assert d["query_id"] == "q001"
        assert d["answer"] == "SDMX is a standard."
        assert d["trust_score"] == 0.85
        assert d["trust_level"] == "HIGH"
        assert d["latency_ms"] == 350
        assert d["error"] is None

    def test_to_jsonl_line_is_valid_json(self):
        result = EvalResult(
            sample=self._make_sample(),
            answer="ans",
            contexts=[],
            context_metadata=[],
            citations=[],
            trust_score=0.5,
            trust_level="MEDIUM",
            latency_ms=100,
        )
        line = result.to_jsonl_line()
        parsed = json.loads(line)
        assert parsed["query_id"] == "q001"

    def test_answer_for_eval_populated_when_set(self):
        result = EvalResult(
            sample=self._make_sample(),
            answer="SDMX is a standard [[1] Fuente: X | Sección: S | Líneas: 1-2].",
            answer_for_eval="SDMX is a standard.",
            contexts=[],
            context_metadata=[],
            citations=[],
            trust_score=0.9,
            trust_level="HIGH",
            latency_ms=100,
        )
        d = result.to_jsonl_dict()
        assert d["answer_for_eval"] == "SDMX is a standard."
        assert d["answer"] != d["answer_for_eval"]

    def test_answer_for_eval_falls_back_to_answer_when_empty(self):
        result = EvalResult(
            sample=self._make_sample(),
            answer="SDMX is a standard.",
            contexts=[],
            context_metadata=[],
            citations=[],
            trust_score=0.9,
            trust_level="HIGH",
            latency_ms=100,
        )
        d = result.to_jsonl_dict()
        # answer_for_eval defaults to "" → falls back to answer
        assert d["answer_for_eval"] == "SDMX is a standard."

    def test_error_field_preserved(self):
        result = EvalResult(
            sample=self._make_sample(),
            answer="",
            contexts=[],
            context_metadata=[],
            citations=[],
            trust_score=0.0,
            trust_level="LOW",
            latency_ms=50,
            error="LLM connection refused",
        )
        d = result.to_jsonl_dict()
        assert d["error"] == "LLM connection refused"
