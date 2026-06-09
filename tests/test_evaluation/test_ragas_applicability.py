"""Tests for rag_lab.evaluation.ragas_applicability."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from rag_lab.evaluation.ragas_applicability import (
    ApplicabilityEntry,
    build_applicability_report,
    compute_no_answer_correctness,
    load_applicability_map,
    mean_score,
    split_by_applicability,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _yaml_file(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "queries.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def _query_block(qid: str, ragas_block: str = "") -> str:
    lines = [
        f"  - id: {qid}",
        f'    text: "Test query {qid}"',
        "    category: glossary_definition",
        "    language: en",
        "    suite: official",
        "    validated: true",
    ]
    if ragas_block:
        for line in textwrap.dedent(ragas_block).splitlines():
            lines.append("    " + line if line.strip() else "")
    return "\n".join(lines) + "\n"


# ── load_applicability_map ────────────────────────────────────────────────────

def test_default_applicable_when_no_ragas_block(tmp_path):
    p = _yaml_file(tmp_path, "queries:\n" + _query_block("q001"))
    m = load_applicability_map(p)
    assert m["q001"].applicable is True
    assert m["q001"].reason == "normal_in_corpus"


def test_not_applicable_query(tmp_path):
    block = textwrap.dedent("""\
        ragas:
          answer_relevancy_applicable: false
          applicability_reason: ambiguity_test
          decision: keep_as_stress_test
    """)
    p = _yaml_file(tmp_path, "queries:\n" + _query_block("q048", block))
    m = load_applicability_map(p)
    assert m["q048"].applicable is False
    assert m["q048"].reason == "ambiguity_test"
    assert m["q048"].decision == "keep_as_stress_test"


def test_invalid_reason_raises(tmp_path):
    block = textwrap.dedent("""\
        ragas:
          answer_relevancy_applicable: false
          applicability_reason: unknown_reason
    """)
    p = _yaml_file(tmp_path, "queries:\n" + _query_block("q001", block))
    with pytest.raises(ValueError, match="invalid applicability_reason"):
        load_applicability_map(p)


def test_invalid_decision_raises(tmp_path):
    block = textwrap.dedent("""\
        ragas:
          answer_relevancy_applicable: false
          applicability_reason: ambiguity_test
          decision: bad_decision
    """)
    p = _yaml_file(tmp_path, "queries:\n" + _query_block("q001", block))
    with pytest.raises(ValueError, match="invalid decision"):
        load_applicability_map(p)


def test_empty_decision_allowed(tmp_path):
    block = textwrap.dedent("""\
        ragas:
          answer_relevancy_applicable: false
          applicability_reason: meta_synthesis
    """)
    p = _yaml_file(tmp_path, "queries:\n" + _query_block("q001", block))
    m = load_applicability_map(p)
    assert m["q001"].decision == ""


def test_missing_yaml_returns_empty_dict():
    m = load_applicability_map(Path("/nonexistent/path.yaml"))
    assert m == {}


def test_multiple_queries_mixed(tmp_path):
    not_applicable_block = textwrap.dedent("""\
        ragas:
          answer_relevancy_applicable: false
          applicability_reason: out_of_corpus
          decision: needs_corpus_expansion
    """)
    content = "queries:\n" + _query_block("q001") + _query_block("q002", not_applicable_block)
    p = _yaml_file(tmp_path, content)
    m = load_applicability_map(p)
    assert m["q001"].applicable is True
    assert m["q002"].applicable is False
    assert m["q002"].reason == "out_of_corpus"


# ── split_by_applicability ────────────────────────────────────────────────────

def _make_rows(query_ids: list[str]) -> list[dict]:
    return [
        {
            "query_id": qid,
            "question": f"Question {qid}",
            "category": "test",
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "answer_for_eval": f"Clean answer {qid}",
            "answer": f"Raw answer with [[1] source] {qid}",
        }
        for qid in query_ids
    ]


def test_split_all_applicable(tmp_path):
    amap = {"q001": ApplicabilityEntry(True, "normal_in_corpus"), "q002": ApplicabilityEntry(True, "normal_in_corpus")}
    rows = _make_rows(["q001", "q002"])
    applicable, not_applicable = split_by_applicability(rows, amap)
    assert len(applicable) == 2
    assert len(not_applicable) == 0


def test_split_mixed(tmp_path):
    amap = {
        "q001": ApplicabilityEntry(True, "normal_in_corpus"),
        "q048": ApplicabilityEntry(False, "ambiguity_test"),
    }
    rows = _make_rows(["q001", "q048"])
    applicable, not_applicable = split_by_applicability(rows, amap)
    assert len(applicable) == 1
    assert applicable[0]["query_id"] == "q001"
    assert len(not_applicable) == 1
    assert not_applicable[0]["query_id"] == "q048"


def test_split_preserves_all_rows():
    amap = {
        "q001": ApplicabilityEntry(True, "normal_in_corpus"),
        "q002": ApplicabilityEntry(False, "meta_synthesis"),
        "q003": ApplicabilityEntry(False, "ambiguity_test"),
    }
    rows = _make_rows(["q001", "q002", "q003"])
    applicable, not_applicable = split_by_applicability(rows, amap)
    assert len(applicable) + len(not_applicable) == len(rows)


def test_split_scores_unchanged():
    amap = {"q048": ApplicabilityEntry(False, "ambiguity_test")}
    rows = _make_rows(["q048"])
    rows[0]["answer_relevancy"] = 0.0
    _, not_applicable = split_by_applicability(rows, amap)
    assert not_applicable[0]["answer_relevancy"] == 0.0


def test_split_answer_fields_unchanged():
    amap = {"q001": ApplicabilityEntry(True, "normal_in_corpus")}
    rows = _make_rows(["q001"])
    original_answer = rows[0]["answer"]
    original_eval = rows[0]["answer_for_eval"]
    applicable, _ = split_by_applicability(rows, amap)
    assert applicable[0]["answer"] == original_answer
    assert applicable[0]["answer_for_eval"] == original_eval


def test_split_unknown_query_id_defaults_to_applicable():
    rows = _make_rows(["q999"])
    applicable, not_applicable = split_by_applicability(rows, {})
    assert len(applicable) == 1
    assert len(not_applicable) == 0


# ── mean_score ────────────────────────────────────────────────────────────────

def test_mean_score_correct():
    rows = [{"ar": 0.9}, {"ar": 0.7}, {"ar": 0.8}]
    assert abs(mean_score(rows, "ar") - 0.8) < 1e-9


def test_mean_score_empty_returns_none():
    assert mean_score([], "ar") is None


def test_mean_score_skips_none_values():
    rows = [{"ar": 0.9}, {"ar": None}, {"ar": 0.7}]
    assert abs(mean_score(rows, "ar") - 0.8) < 1e-9


# ── build_applicability_report ────────────────────────────────────────────────

def test_report_structure():
    amap = {
        "q001": ApplicabilityEntry(True, "normal_in_corpus"),
        "q048": ApplicabilityEntry(False, "ambiguity_test", "keep_as_stress_test"),
    }
    rows = _make_rows(["q001", "q048"])
    rows[0]["answer_relevancy"] = 0.9
    rows[1]["answer_relevancy"] = 0.0

    report = build_applicability_report(rows, amap, ["answer_relevancy"])

    assert report["n_all"] == 2
    assert report["n_applicable"] == 1
    assert report["n_not_applicable"] == 1
    assert abs(report["scores_all"]["answer_relevancy"] - 0.45) < 1e-9
    assert abs(report["scores_applicable"]["answer_relevancy"] - 0.9) < 1e-9
    assert abs(report["scores_not_applicable"]["answer_relevancy"] - 0.0) < 1e-9


def test_report_not_applicable_visible():
    amap = {"q048": ApplicabilityEntry(False, "ambiguity_test", "keep_as_stress_test")}
    rows = _make_rows(["q048"])
    rows[0]["answer_relevancy"] = 0.0

    report = build_applicability_report(rows, amap, ["answer_relevancy"])

    assert len(report["not_applicable_queries"]) == 1
    entry = report["not_applicable_queries"][0]
    assert entry["query_id"] == "q048"
    assert entry["reason"] == "ambiguity_test"
    assert entry["decision"] == "keep_as_stress_test"
    assert entry["answer_relevancy"] == 0.0


def test_report_scores_not_altered():
    """Applicability splitting must never modify scores."""
    amap = {
        "q001": ApplicabilityEntry(True, "normal_in_corpus"),
        "q002": ApplicabilityEntry(False, "out_of_corpus", "needs_corpus_expansion"),
    }
    rows = _make_rows(["q001", "q002"])
    rows[0]["faithfulness"] = 0.95
    rows[1]["faithfulness"] = 0.40

    report = build_applicability_report(rows, amap, ["faithfulness"])
    # Applicable mean must be exactly 0.95, not averaged with the not-applicable row
    assert abs(report["scores_applicable"]["faithfulness"] - 0.95) < 1e-9
    # Not-applicable score preserved (not zeroed or dropped)
    assert abs(report["not_applicable_queries"][0]["faithfulness"] - 0.40) < 1e-9


def test_report_applicable_scores_exclude_not_applicable():
    amap = {
        "q001": ApplicabilityEntry(True, "normal_in_corpus"),
        "q002": ApplicabilityEntry(True, "normal_in_corpus"),
        "q048": ApplicabilityEntry(False, "ambiguity_test"),
    }
    rows = _make_rows(["q001", "q002", "q048"])
    rows[0]["answer_relevancy"] = 0.9
    rows[1]["answer_relevancy"] = 0.8
    rows[2]["answer_relevancy"] = 0.0

    report = build_applicability_report(rows, amap, ["answer_relevancy"])

    # Applicable mean: (0.9 + 0.8) / 2 = 0.85
    assert abs(report["scores_applicable"]["answer_relevancy"] - 0.85) < 1e-9
    # All mean: (0.9 + 0.8 + 0.0) / 3 ≈ 0.567
    assert abs(report["scores_all"]["answer_relevancy"] - (0.9 + 0.8 + 0.0) / 3) < 1e-9


# ── integration: load real benchmark_queries.yaml ────────────────────────────

def test_real_yaml_loads_without_error():
    """The live benchmark_queries.yaml must parse cleanly."""
    from rag_lab.evaluation.ragas_applicability import DEFAULT_BENCHMARK_YAML
    if not DEFAULT_BENCHMARK_YAML.exists():
        pytest.skip("benchmark_queries.yaml not found")
    m = load_applicability_map()
    assert len(m) > 0


def test_real_yaml_has_ten_not_applicable():
    from rag_lab.evaluation.ragas_applicability import DEFAULT_BENCHMARK_YAML
    if not DEFAULT_BENCHMARK_YAML.exists():
        pytest.skip("benchmark_queries.yaml not found")
    m = load_applicability_map()
    not_applicable = {qid for qid, e in m.items() if not e.applicable}
    expected = {"q013", "q032", "q038", "q039", "q041", "q042", "q048", "q050", "q054", "q065"}
    assert not_applicable == expected


def test_real_yaml_official_suite_applicable_count():
    """55 of the 62 official-suite queries must be applicable."""
    from rag_lab.evaluation.ragas_applicability import DEFAULT_BENCHMARK_YAML
    if not DEFAULT_BENCHMARK_YAML.exists():
        pytest.skip("benchmark_queries.yaml not found")
    import yaml as _yaml
    with open(DEFAULT_BENCHMARK_YAML) as f:
        data = _yaml.safe_load(f)
    official = [q for q in data["queries"] if q.get("suite") == "official" and q.get("validated")]
    m = load_applicability_map()
    applicable_official = [q for q in official if m.get(q["id"], ApplicabilityEntry(True, "normal_in_corpus")).applicable]
    assert len(official) == 62
    assert len(applicable_official) == 55


def test_real_yaml_negative_suite_count():
    """Exactly 3 validated negative-suite queries (q039, q041, q042)."""
    from rag_lab.evaluation.ragas_applicability import DEFAULT_BENCHMARK_YAML
    if not DEFAULT_BENCHMARK_YAML.exists():
        pytest.skip("benchmark_queries.yaml not found")
    import yaml as _yaml
    with open(DEFAULT_BENCHMARK_YAML) as f:
        data = _yaml.safe_load(f)
    negative = [q for q in data["queries"] if q.get("suite") == "negative" and q.get("validated")]
    assert len(negative) == 3
    negative_ids = {q["id"] for q in negative}
    assert negative_ids == {"q039", "q041", "q042"}


def test_real_yaml_negative_queries_not_applicable():
    """q039/q041/q042 must be in the applicability map with applicable=False."""
    from rag_lab.evaluation.ragas_applicability import DEFAULT_BENCHMARK_YAML
    if not DEFAULT_BENCHMARK_YAML.exists():
        pytest.skip("benchmark_queries.yaml not found")
    m = load_applicability_map()
    for qid in ("q039", "q041", "q042"):
        assert qid in m, f"{qid} missing from applicability map"
        assert m[qid].applicable is False
        assert m[qid].reason == "answer_not_present_in_current_corpus"
        assert m[qid].decision == "keep_as_no_answer_test"


# ── new reason/decision constants ────────────────────────────────────────────

def test_answer_not_present_in_current_corpus_is_valid_reason(tmp_path):
    block = textwrap.dedent("""\
        ragas:
          answer_relevancy_applicable: false
          applicability_reason: answer_not_present_in_current_corpus
          decision: keep_as_no_answer_test
    """)
    p = _yaml_file(tmp_path, "queries:\n" + _query_block("q039", block))
    m = load_applicability_map(p)
    assert m["q039"].applicable is False
    assert m["q039"].reason == "answer_not_present_in_current_corpus"
    assert m["q039"].decision == "keep_as_no_answer_test"


def test_keep_as_no_answer_test_is_valid_decision(tmp_path):
    block = textwrap.dedent("""\
        ragas:
          answer_relevancy_applicable: false
          applicability_reason: out_of_corpus
          decision: keep_as_no_answer_test
    """)
    p = _yaml_file(tmp_path, "queries:\n" + _query_block("q001", block))
    m = load_applicability_map(p)
    assert m["q001"].decision == "keep_as_no_answer_test"


# ── compute_no_answer_correctness ─────────────────────────────────────────────

def _make_negative_amap(*qids: str) -> dict:
    return {
        qid: ApplicabilityEntry(
            applicable=False,
            reason="answer_not_present_in_current_corpus",
            decision="keep_as_no_answer_test",
        )
        for qid in qids
    }


def _row(qid: str, answer: str, trust: float | None = None) -> dict:
    return {
        "query_id": qid,
        "question": f"Q {qid}",
        "answer": answer,
        "trust_score": trust,
    }


def test_no_answer_correctness_no_negative_queries():
    amap = {"q001": ApplicabilityEntry(True, "normal_in_corpus")}
    rows = [_row("q001", "Some answer about SDMX", 0.8)]
    result = compute_no_answer_correctness(rows, amap)
    assert result["n_negative"] == 0
    assert result["n_correct_abstentions"] == 0
    assert result["no_answer_correctness"] is None
    assert result["per_query"] == []


def test_no_answer_correctness_correct_abstention():
    amap = _make_negative_amap("q039")
    rows = [_row("q039", "No encuentro información sobre eso en el corpus.", 0.1)]
    result = compute_no_answer_correctness(rows, amap)
    assert result["n_negative"] == 1
    assert result["n_correct_abstentions"] == 1
    assert result["no_answer_correctness"] == 1.0
    assert result["per_query"][0]["abstained_correctly"] is True


def test_no_answer_correctness_wrong_answer_counted_as_failure():
    amap = _make_negative_amap("q039")
    rows = [_row("q039", "The allowed values are A, B, and C according to the SDMX standard.", 0.9)]
    result = compute_no_answer_correctness(rows, amap)
    assert result["n_negative"] == 1
    assert result["n_correct_abstentions"] == 0
    assert result["no_answer_correctness"] == 0.0
    assert result["per_query"][0]["abstained_correctly"] is False


def test_no_answer_correctness_partial_score():
    amap = _make_negative_amap("q039", "q041", "q042")
    rows = [
        _row("q039", "No encuentro información sobre los valores permitidos.", 0.1),
        _row("q041", "The context does not contain information about mandatory header elements.", 0.15),
        _row("q042", "The XML namespace prefixes are urn:sdmx, xsi, and message prefix.", 0.85),
    ]
    result = compute_no_answer_correctness(rows, amap)
    assert result["n_negative"] == 3
    assert result["n_correct_abstentions"] == 2
    assert abs(result["no_answer_correctness"] - 2 / 3) < 1e-9


def test_no_answer_correctness_per_query_has_required_fields():
    amap = _make_negative_amap("q039")
    rows = [_row("q039", "No encuentro información sobre eso.", 0.05)]
    result = compute_no_answer_correctness(rows, amap)
    entry = result["per_query"][0]
    assert "query_id" in entry
    assert "question" in entry
    assert "abstained_correctly" in entry
    assert "trust_score" in entry
    assert "answer_snippet" in entry


def test_no_answer_correctness_empty_answer_is_abstention():
    amap = _make_negative_amap("q039")
    rows = [_row("q039", "", None)]
    result = compute_no_answer_correctness(rows, amap)
    assert result["per_query"][0]["abstained_correctly"] is True


def test_no_answer_correctness_ignores_non_negative_rows():
    """Rows not in the amap as keep_as_no_answer_test must be ignored."""
    amap = {
        "q001": ApplicabilityEntry(True, "normal_in_corpus"),
        "q039": ApplicabilityEntry(False, "answer_not_present_in_current_corpus", "keep_as_no_answer_test"),
    }
    rows = [
        _row("q001", "SDMX is a data standard.", 0.9),
        _row("q039", "No encuentro información sobre eso.", 0.05),
    ]
    result = compute_no_answer_correctness(rows, amap)
    assert result["n_negative"] == 1
    assert result["per_query"][0]["query_id"] == "q039"
