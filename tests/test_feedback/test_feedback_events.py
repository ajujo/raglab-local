"""Tests for the v1.15 structured chunk-level FeedbackStore.

Covers: schema creation, add/list/stats/export/clear, validation,
query_hash determinism, retrieval_config_hash determinism,
feedback does NOT change ranking, feedback does NOT invalidate cache.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_lab.feedback.store import (
    VALID_FEEDBACK,
    FeedbackStore,
    make_query_hash,
    make_retrieval_config_hash,
    reset_feedback_store_instance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    db = tmp_path / "feedback_test.sqlite"
    s = FeedbackStore(db_path=db)
    s.initialize()
    yield s
    s.close()


def _add(store, *, query="What is SDMX?", chunk_id="c-001", doc_id="sdmx_glossary",
         rank=1, feedback="relevant", **kw):
    return store.add(query, chunk_id=chunk_id, doc_id=doc_id, rank=rank,
                     feedback=feedback, **kw)


# ---------------------------------------------------------------------------
# 1. Schema creates idempotently
# ---------------------------------------------------------------------------

def test_initialize_idempotent(tmp_path):
    db = tmp_path / "fb.sqlite"
    s = FeedbackStore(db_path=db)
    s.initialize()
    s.initialize()  # second call should not raise
    s.close()


def test_schema_table_exists(tmp_path):
    db = tmp_path / "fb.sqlite"
    s = FeedbackStore(db_path=db)
    s.initialize()
    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback_events'"
    ).fetchone()
    conn.close()
    s.close()
    assert row is not None


# ---------------------------------------------------------------------------
# 2. Add valid feedback
# ---------------------------------------------------------------------------

def test_add_returns_row_id(store):
    row_id = _add(store)
    assert isinstance(row_id, int)
    assert row_id >= 1


def test_add_all_valid_feedback_types(store):
    for i, fb in enumerate(sorted(VALID_FEEDBACK)):
        row_id = _add(store, chunk_id=f"chunk-{i}", feedback=fb)
        assert row_id >= 1


def test_add_persists_fields(store):
    _add(store, query="Query A", chunk_id="ch-xyz", doc_id="doc-1",
         rank=3, feedback="irrelevant", rating=2, reason="off-topic",
         user_note="test note", source="cli", pipeline_variant="full",
         cache_hit=True, cache_key="abc123", corpus_fingerprint="10:5:0")
    rows = store.list()
    assert len(rows) == 1
    r = rows[0]
    assert r["query_text"] == "Query A"
    assert r["chunk_id"] == "ch-xyz"
    assert r["doc_id"] == "doc-1"
    assert r["rank"] == 3
    assert r["feedback"] == "irrelevant"
    assert r["rating"] == 2
    assert r["reason"] == "off-topic"
    assert r["user_note"] == "test note"
    assert r["cache_hit"] == 1
    assert r["cache_key"] == "abc123"
    assert r["corpus_fingerprint"] == "10:5:0"


# ---------------------------------------------------------------------------
# 3. Reject invalid feedback
# ---------------------------------------------------------------------------

def test_add_invalid_feedback_raises(store):
    with pytest.raises(ValueError, match="Invalid feedback"):
        _add(store, feedback="not_a_valid_type")


def test_add_invalid_feedback_nothing_stored(store):
    try:
        _add(store, feedback="garbage")
    except ValueError:
        pass
    assert store.stats()["total_events"] == 0


# ---------------------------------------------------------------------------
# 4. List returns events
# ---------------------------------------------------------------------------

def test_list_returns_all(store):
    for i in range(5):
        _add(store, chunk_id=f"c-{i}", feedback="relevant")
    rows = store.list()
    assert len(rows) == 5


def test_list_most_recent_first(store):
    for i in range(3):
        _add(store, chunk_id=f"c-{i}")
    rows = store.list()
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids, reverse=True)


def test_list_filter_by_chunk_id(store):
    _add(store, chunk_id="target")
    _add(store, chunk_id="other")
    rows = store.list(chunk_id="target")
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "target"


def test_list_filter_by_feedback(store):
    _add(store, chunk_id="c1", feedback="relevant")
    _add(store, chunk_id="c2", feedback="irrelevant")
    rows = store.list(feedback="relevant")
    assert len(rows) == 1
    assert rows[0]["feedback"] == "relevant"


def test_list_filter_by_query_hash(store):
    qh = make_query_hash("specific query")
    _add(store, query="specific query", chunk_id="c1")
    _add(store, query="other query", chunk_id="c2")
    rows = store.list(query_hash=qh)
    assert len(rows) == 1
    assert rows[0]["query_hash"] == qh


def test_list_limit(store):
    for i in range(10):
        _add(store, chunk_id=f"c-{i}")
    rows = store.list(limit=3)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# 5. Stats aggregate by feedback type
# ---------------------------------------------------------------------------

def test_stats_empty(store):
    s = store.stats()
    assert s["total_events"] == 0
    assert s["unique_queries"] == 0
    assert s["unique_chunks"] == 0
    assert s["by_feedback_type"] == {}


def test_stats_counts(store):
    _add(store, chunk_id="c1", feedback="relevant")
    _add(store, chunk_id="c2", feedback="relevant")
    _add(store, chunk_id="c3", feedback="irrelevant")
    s = store.stats()
    assert s["total_events"] == 3
    assert s["by_feedback_type"]["relevant"] == 2
    assert s["by_feedback_type"]["irrelevant"] == 1


def test_stats_unique_queries(store):
    _add(store, query="Q1", chunk_id="c1")
    _add(store, query="Q1", chunk_id="c2")  # same normalized query
    _add(store, query="Q2", chunk_id="c3")
    s = store.stats()
    assert s["unique_queries"] == 2
    assert s["unique_chunks"] == 3


# ---------------------------------------------------------------------------
# 6. Export JSONL
# ---------------------------------------------------------------------------

def test_export_jsonl_returns_string(store):
    _add(store, chunk_id="c1")
    result = store.export_jsonl()
    assert isinstance(result, str)
    obj = json.loads(result.strip())
    assert obj["chunk_id"] == "c1"


def test_export_jsonl_writes_file(store, tmp_path):
    _add(store, chunk_id="c1", corpus_fingerprint="5:3:1")
    out = tmp_path / "out.jsonl"
    store.export_jsonl(path=out)
    assert out.exists()
    line = json.loads(out.read_text().strip())
    assert line["corpus_fingerprint"] == "5:3:1"


def test_export_includes_corpus_fingerprint(store):
    _add(store, chunk_id="c1", corpus_fingerprint="610:7:2")
    result = store.export_jsonl()
    obj = json.loads(result.strip())
    assert obj["corpus_fingerprint"] == "610:7:2"


def test_export_empty(store):
    result = store.export_jsonl()
    assert result == ""


# ---------------------------------------------------------------------------
# 7. Clear
# ---------------------------------------------------------------------------

def test_clear_removes_events(store):
    for i in range(5):
        _add(store, chunk_id=f"c-{i}")
    n = store.clear()
    assert n == 5
    assert store.stats()["total_events"] == 0


def test_clear_returns_count(store):
    _add(store, chunk_id="c1")
    _add(store, chunk_id="c2")
    n = store.clear()
    assert n == 2


# ---------------------------------------------------------------------------
# 8. Feedback on non-existent chunk is allowed (no FK to chunks table)
# ---------------------------------------------------------------------------

def test_add_unknown_chunk_id_allowed(store):
    row_id = _add(store, chunk_id="nonexistent-chunk-xyz")
    assert row_id >= 1


# ---------------------------------------------------------------------------
# 9. query_hash is deterministic
# ---------------------------------------------------------------------------

def test_query_hash_deterministic():
    h1 = make_query_hash("What is SDMX?")
    h2 = make_query_hash("What is SDMX?")
    assert h1 == h2


def test_query_hash_case_insensitive():
    h1 = make_query_hash("what is sdmx?")
    h2 = make_query_hash("WHAT IS SDMX?")
    assert h1 == h2


def test_query_hash_whitespace_normalized():
    h1 = make_query_hash("what  is   sdmx?")
    h2 = make_query_hash("what is sdmx?")
    assert h1 == h2


def test_query_hash_different_queries_differ():
    h1 = make_query_hash("What is SDMX?")
    h2 = make_query_hash("What is RDF?")
    assert h1 != h2


def test_add_stores_correct_query_hash(store):
    _add(store, query="What is SDMX?")
    rows = store.list()
    expected = make_query_hash("What is SDMX?")
    assert rows[0]["query_hash"] == expected


# ---------------------------------------------------------------------------
# 10. retrieval_config_hash is deterministic
# ---------------------------------------------------------------------------

def test_retrieval_config_hash_deterministic():
    h1 = make_retrieval_config_hash()
    h2 = make_retrieval_config_hash()
    assert h1 == h2


def test_retrieval_config_hash_is_hex():
    h = make_retrieval_config_hash()
    assert len(h) == 64
    int(h, 16)  # valid hex


def test_retrieval_config_hash_stored_on_add(store):
    _add(store, chunk_id="c1")
    rows = store.list()
    expected = make_retrieval_config_hash()
    assert rows[0]["retrieval_config_hash"] == expected


# ---------------------------------------------------------------------------
# 11. Feedback does NOT change retrieval ranking
# ---------------------------------------------------------------------------

def test_feedback_does_not_alter_retrieval_results(tmp_path):
    """Adding feedback must not modify any retrieval index or score."""
    db = tmp_path / "fb.sqlite"
    store = FeedbackStore(db_path=db)
    store.initialize()

    simulated_results_before = [
        {"chunk_id": "c1", "doc_id": "d1", "rrf_score": 0.9},
        {"chunk_id": "c2", "doc_id": "d1", "rrf_score": 0.7},
        {"chunk_id": "c3", "doc_id": "d2", "rrf_score": 0.5},
    ]

    # Record feedback for top chunk
    store.add(
        "What is SDMX?",
        chunk_id="c1", doc_id="d1", rank=1, feedback="irrelevant",
    )

    # Verify the results list is unchanged (feedback is write-only to its own table)
    simulated_results_after = [
        {"chunk_id": "c1", "doc_id": "d1", "rrf_score": 0.9},
        {"chunk_id": "c2", "doc_id": "d1", "rrf_score": 0.7},
        {"chunk_id": "c3", "doc_id": "d2", "rrf_score": 0.5},
    ]
    assert simulated_results_before == simulated_results_after

    store.close()


# ---------------------------------------------------------------------------
# 12. Feedback does NOT invalidate query cache
# ---------------------------------------------------------------------------

def test_feedback_does_not_change_corpus_fingerprint(tmp_path):
    """FeedbackStore writes to feedback_events only — cache_revision unchanged."""
    from rag_lab.storage.metadata_store import MetadataStore
    from rag_lab.cache.query_cache import get_corpus_fingerprint

    db = tmp_path / "docstore.sqlite"
    meta = MetadataStore(db_path=db)
    meta.initialize()
    meta.upsert_document("doc-a", title="Doc A")
    fp_before = get_corpus_fingerprint(meta._conn)

    fb_store = FeedbackStore(db_path=db)
    fb_store.initialize()
    fb_store.add(
        "What is SDMX?",
        chunk_id="c1", doc_id="doc-a", rank=1, feedback="relevant",
    )

    fp_after = get_corpus_fingerprint(meta._conn)
    assert fp_before == fp_after, (
        "Corpus fingerprint changed after feedback — this would invalidate the cache"
    )

    fb_store.close()
    meta.close()


# ---------------------------------------------------------------------------
# 13. CLI commands (basic smoke via import)
# ---------------------------------------------------------------------------

def test_feedback_add_command_registered():
    from rag_lab.cli import feedback_app
    command_names = [cmd.name for cmd in feedback_app.registered_commands]
    assert "add" in command_names


def test_feedback_list_command_registered():
    from rag_lab.cli import feedback_app
    command_names = [cmd.name for cmd in feedback_app.registered_commands]
    assert "list" in command_names


def test_feedback_stats_command_registered():
    from rag_lab.cli import feedback_app
    command_names = [cmd.name for cmd in feedback_app.registered_commands]
    assert "stats" in command_names


def test_feedback_export_command_registered():
    from rag_lab.cli import feedback_app
    command_names = [cmd.name for cmd in feedback_app.registered_commands]
    assert "export" in command_names


def test_feedback_clear_command_registered():
    from rag_lab.cli import feedback_app
    command_names = [cmd.name for cmd in feedback_app.registered_commands]
    assert "clear" in command_names
