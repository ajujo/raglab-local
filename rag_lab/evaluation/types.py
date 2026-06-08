"""Data types for E2E evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """One evaluation query, derived from benchmark_queries.yaml."""

    query_id: str
    question: str
    language: str | None
    category: str | None
    expected_answer: str | None
    expected_doc_ids: list[str]
    doc_relevance: dict[str, int]

    @classmethod
    def from_yaml_entry(cls, entry: dict) -> "EvalSample":
        doc_relevance: dict[str, int] = entry.get("doc_relevance") or {}
        expected_doc_ids = [
            doc_id for doc_id, grade in doc_relevance.items() if grade >= 2
        ]
        return cls(
            query_id=entry["id"],
            question=entry.get("text", entry.get("query", "")),
            language=entry.get("language"),
            category=entry.get("category"),
            expected_answer=entry.get("expected_answer"),
            expected_doc_ids=expected_doc_ids,
            doc_relevance=doc_relevance,
        )


@dataclass
class EvalResult:
    """Full pipeline output for a single evaluation query."""

    sample: EvalSample
    answer: str
    contexts: list[str]
    context_metadata: list[dict]
    citations: list[dict]
    trust_score: float
    trust_level: str
    latency_ms: int
    error: str | None = None

    def to_jsonl_dict(self) -> dict:
        return {
            "query_id": self.sample.query_id,
            "question": self.sample.question,
            "language": self.sample.language,
            "category": self.sample.category,
            "answer": self.answer,
            "contexts": self.contexts,
            "context_metadata": self.context_metadata,
            "citations": self.citations,
            "trust_score": self.trust_score,
            "trust_level": self.trust_level,
            "latency_ms": self.latency_ms,
            "expected_answer": self.sample.expected_answer,
            "expected_doc_ids": self.sample.expected_doc_ids,
            "doc_relevance": self.sample.doc_relevance,
            "error": self.error,
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_jsonl_dict(), ensure_ascii=False)
