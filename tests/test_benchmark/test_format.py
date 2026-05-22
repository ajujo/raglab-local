"""Tests for v1.8 benchmark query format: new fields, filter_queries, backward compat."""

import pytest

from rag_lab.benchmark.runner import BenchmarkRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query(
    id="q001",
    text="What is SDMX?",
    category="glossary_definition",
    language="en",
    suite="official",
    validated=True,
    doc_relevance=None,
    **extra,
):
    q = {
        "id": id,
        "text": text,
        "category": category,
        "language": language,
        "suite": suite,
        "validated": validated,
        "doc_relevance": doc_relevance or {"SDMX_Glossary": 3},
        "notes": "test note",
    }
    q.update(extra)
    return q


def _v17_query():
    """Minimal v1.7-style query with no new fields."""
    return {
        "id": "q001",
        "text": "What is SDMX?",
        "doc_relevance": {"SDMX_Glossary": 3},
        "notes": "Basic definition.",
    }


# ---------------------------------------------------------------------------
# load_queries — backward compatibility
# ---------------------------------------------------------------------------

class TestLoadQueriesBackwardCompat:
    def test_loads_v17_yaml(self, tmp_path):
        """v1.7 YAML without new fields loads without error."""
        f = tmp_path / "queries.yaml"
        f.write_text(
            "queries:\n"
            "  - id: q001\n"
            "    text: 'What is SDMX?'\n"
            "    doc_relevance:\n"
            "      SDMX_Glossary: 3\n"
            "    notes: 'test'\n"
        )
        queries = BenchmarkRunner.load_queries(f)
        assert len(queries) == 1
        assert queries[0]["id"] == "q001"

    def test_loads_v18_yaml(self, tmp_path):
        """v1.8 YAML with all new fields loads correctly."""
        f = tmp_path / "queries.yaml"
        f.write_text(
            "queries:\n"
            "  - id: q001\n"
            "    text: 'What is SDMX?'\n"
            "    category: glossary_definition\n"
            "    language: en\n"
            "    suite: official\n"
            "    validated: true\n"
            "    expected_behavior: 'Return definition chunks'\n"
            "    source_of_truth: 'SDMX_Glossary'\n"
            "    doc_relevance:\n"
            "      SDMX_Glossary: 3\n"
            "    notes: 'test'\n"
        )
        queries = BenchmarkRunner.load_queries(f)
        assert queries[0].get("category") == "glossary_definition"
        assert queries[0].get("validated") is True
        assert queries[0].get("suite") == "official"

    def test_v17_query_has_no_suite_field(self):
        """v1.7 query dict does not have 'suite' field."""
        q = _v17_query()
        assert "suite" not in q

    def test_filter_treats_missing_suite_as_official(self):
        """filter_queries with suite='official' includes v1.7 queries (no suite key)."""
        queries = [_v17_query()]
        filtered = BenchmarkRunner.filter_queries(queries, suite="official")
        assert len(filtered) == 1

    def test_filter_treats_missing_validated_as_true(self):
        """filter_queries with validated_only=True includes queries with no 'validated' key."""
        queries = [_v17_query()]
        filtered = BenchmarkRunner.filter_queries(queries, validated_only=True)
        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# filter_queries
# ---------------------------------------------------------------------------

class TestFilterQueries:
    def _make_dataset(self):
        return [
            _make_query("q001", suite="official", validated=True),
            _make_query("q002", suite="official", validated=False),
            _make_query("q003", suite="candidate", validated=False),
            _make_query("q004", suite="candidate", validated=False),
        ]

    def test_filter_official_validated_only(self):
        qs = self._make_dataset()
        result = BenchmarkRunner.filter_queries(qs, suite="official", validated_only=True)
        assert len(result) == 1
        assert result[0]["id"] == "q001"

    def test_filter_official_all(self):
        qs = self._make_dataset()
        result = BenchmarkRunner.filter_queries(qs, suite="official")
        ids = [q["id"] for q in result]
        assert "q001" in ids
        assert "q002" in ids
        assert "q003" not in ids

    def test_filter_candidates(self):
        qs = self._make_dataset()
        result = BenchmarkRunner.filter_queries(qs, suite="candidate")
        ids = [q["id"] for q in result]
        assert "q003" in ids
        assert "q004" in ids
        assert "q001" not in ids

    def test_filter_none_suite_returns_all(self):
        qs = self._make_dataset()
        result = BenchmarkRunner.filter_queries(qs, suite=None)
        assert len(result) == 4

    def test_filter_validated_only_excludes_false(self):
        qs = self._make_dataset()
        result = BenchmarkRunner.filter_queries(qs, validated_only=True)
        ids = [q["id"] for q in result]
        assert "q001" in ids
        assert "q002" not in ids
        assert "q003" not in ids

    def test_filter_empty_list(self):
        assert BenchmarkRunner.filter_queries([], suite="official") == []

    def test_candidates_not_in_official_validated(self):
        """Candidate queries do not appear in official+validated filter."""
        qs = self._make_dataset()
        official = BenchmarkRunner.filter_queries(qs, suite="official", validated_only=True)
        ids = [q["id"] for q in official]
        assert "q003" not in ids
        assert "q004" not in ids


# ---------------------------------------------------------------------------
# New fields in query dict
# ---------------------------------------------------------------------------

class TestQueryNewFields:
    def test_category_field_present(self):
        q = _make_query(category="technical_standard")
        assert q["category"] == "technical_standard"

    def test_language_field_en(self):
        q = _make_query(language="en")
        assert q["language"] == "en"

    def test_language_field_es(self):
        q = _make_query(language="es")
        assert q["language"] == "es"

    def test_suite_official(self):
        q = _make_query(suite="official")
        assert q["suite"] == "official"

    def test_suite_candidate(self):
        q = _make_query(suite="candidate")
        assert q["suite"] == "candidate"

    def test_validated_true(self):
        q = _make_query(validated=True)
        assert q["validated"] is True

    def test_validated_false(self):
        q = _make_query(validated=False)
        assert q["validated"] is False


# ---------------------------------------------------------------------------
# Real benchmark_queries.yaml validation
# ---------------------------------------------------------------------------

class TestBenchmarkQueriesYaml:
    YAML_PATH = "data/benchmark_queries.yaml"

    def test_file_loadable(self):
        queries = BenchmarkRunner.load_queries(self.YAML_PATH)
        assert len(queries) > 0

    def test_total_count(self):
        queries = BenchmarkRunner.load_queries(self.YAML_PATH)
        assert len(queries) >= 28

    def test_official_queries_have_required_fields(self):
        queries = BenchmarkRunner.load_queries(self.YAML_PATH)
        official = BenchmarkRunner.filter_queries(queries, suite="official", validated_only=True)
        required = ["id", "text", "category", "language", "validated", "doc_relevance"]
        for q in official:
            for field in required:
                assert field in q, f"Query {q.get('id')} missing field '{field}'"

    def test_original_28_queries_present(self):
        queries = BenchmarkRunner.load_queries(self.YAML_PATH)
        ids = {q["id"] for q in queries}
        for i in range(1, 29):
            qid = f"q{i:03d}"
            assert qid in ids, f"Original query {qid} missing from YAML"

    def test_official_all_validated_true(self):
        queries = BenchmarkRunner.load_queries(self.YAML_PATH)
        official = BenchmarkRunner.filter_queries(queries, suite="official")
        for q in official:
            assert q.get("validated") is True, f"{q['id']} is official but validated=False"

    def test_candidates_all_validated_false(self):
        queries = BenchmarkRunner.load_queries(self.YAML_PATH)
        candidates = BenchmarkRunner.filter_queries(queries, suite="candidate")
        for q in candidates:
            assert q.get("validated") is False, f"{q['id']} is candidate but validated=True"

    def test_all_10_categories_covered(self):
        queries = BenchmarkRunner.load_queries(self.YAML_PATH)
        cats = {q.get("category") for q in queries}
        required_cats = {
            "glossary_definition", "technical_standard", "cross_lingual_es_en",
            "multi_chunk_same_doc", "multi_doc_synthesis", "acronym_or_exact_term",
            "table_or_structured_reference", "negative_no_answer",
            "ambiguity_test", "regression_known_hard",
        }
        missing = required_cats - cats
        assert not missing, f"Missing categories: {missing}"

    def test_languages_are_en_or_es(self):
        queries = BenchmarkRunner.load_queries(self.YAML_PATH)
        for q in queries:
            lang = q.get("language")
            if lang is not None:
                assert lang in ("en", "es"), f"{q['id']} has unknown language {lang!r}"
