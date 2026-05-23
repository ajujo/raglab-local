"""Tests for cache invalidation via MetadataStore.cache_revision.

Covers v1.14.1: bump_revision() is called on assign_tag, unassign_tag,
rename_tag, delete_tag, and delete_document — ensuring that
get_corpus_fingerprint() returns a new value after each operation,
which invalidates any cached retrieval results that depended on the
prior corpus state.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from rag_lab.storage.metadata_store import MetadataStore
from rag_lab.cache.query_cache import get_corpus_fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(db_path: Path) -> MetadataStore:
    store = MetadataStore(db_path=db_path)
    store.initialize()
    return store


def _fp(store: MetadataStore) -> str:
    """Get the corpus fingerprint via the store's connection."""
    return get_corpus_fingerprint(store._conn)


def _revision(store: MetadataStore) -> int:
    return store.get_revision()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    db = tmp_path / "meta.sqlite"
    store = _make_store(db)
    # Insert a document so tag operations have a valid doc_id FK target
    store.upsert_document("doc-a", title="Doc A")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# 1. cache_revision initialises to 0
# ---------------------------------------------------------------------------

def test_initial_revision_is_zero(tmp_store):
    assert _revision(tmp_store) == 0


# ---------------------------------------------------------------------------
# 2. assign_tag bumps revision
# ---------------------------------------------------------------------------

def test_assign_tag_bumps_revision(tmp_store):
    before = _revision(tmp_store)
    tmp_store.assign_tag("doc-a", "mytag")
    assert _revision(tmp_store) == before + 1


# ---------------------------------------------------------------------------
# 3. assign_tag changes fingerprint
# ---------------------------------------------------------------------------

def test_assign_tag_changes_fingerprint(tmp_store):
    fp_before = _fp(tmp_store)
    tmp_store.assign_tag("doc-a", "mytag")
    assert _fp(tmp_store) != fp_before


# ---------------------------------------------------------------------------
# 4. unassign_tag bumps revision
# ---------------------------------------------------------------------------

def test_unassign_tag_bumps_revision(tmp_store):
    tmp_store.assign_tag("doc-a", "mytag")
    before = _revision(tmp_store)
    tmp_store.unassign_tag("doc-a", "mytag")
    assert _revision(tmp_store) == before + 1


# ---------------------------------------------------------------------------
# 5. unassign non-existent tag does NOT bump revision
# ---------------------------------------------------------------------------

def test_unassign_nonexistent_tag_no_bump(tmp_store):
    before = _revision(tmp_store)
    tmp_store.unassign_tag("doc-a", "no-such-tag")
    assert _revision(tmp_store) == before


# ---------------------------------------------------------------------------
# 6. rename_tag bumps revision
# ---------------------------------------------------------------------------

def test_rename_tag_bumps_revision(tmp_store):
    tmp_store.assign_tag("doc-a", "old-name")
    before = _revision(tmp_store)
    result = tmp_store.rename_tag("old-name", "new-name")
    assert result is True
    assert _revision(tmp_store) == before + 1


# ---------------------------------------------------------------------------
# 7. rename non-existent tag does NOT bump revision
# ---------------------------------------------------------------------------

def test_rename_nonexistent_tag_no_bump(tmp_store):
    before = _revision(tmp_store)
    result = tmp_store.rename_tag("ghost", "phantom")
    assert result is False
    assert _revision(tmp_store) == before


# ---------------------------------------------------------------------------
# 8. delete_tag bumps revision
# ---------------------------------------------------------------------------

def test_delete_tag_bumps_revision(tmp_store):
    tmp_store.assign_tag("doc-a", "removeme")
    before = _revision(tmp_store)
    result = tmp_store.delete_tag("removeme")
    assert result is True
    assert _revision(tmp_store) == before + 1


# ---------------------------------------------------------------------------
# 9. delete non-existent tag does NOT bump revision
# ---------------------------------------------------------------------------

def test_delete_nonexistent_tag_no_bump(tmp_store):
    before = _revision(tmp_store)
    result = tmp_store.delete_tag("ghost")
    assert result is False
    assert _revision(tmp_store) == before


# ---------------------------------------------------------------------------
# 10. delete_document bumps revision
# ---------------------------------------------------------------------------

def test_delete_document_bumps_revision(tmp_store):
    before = _revision(tmp_store)
    result = tmp_store.delete_document("doc-a")
    assert result is True
    assert _revision(tmp_store) == before + 1


# ---------------------------------------------------------------------------
# 11. delete non-existent document does NOT bump revision
# ---------------------------------------------------------------------------

def test_delete_nonexistent_document_no_bump(tmp_store):
    before = _revision(tmp_store)
    result = tmp_store.delete_document("no-such-doc")
    assert result is False
    assert _revision(tmp_store) == before


# ---------------------------------------------------------------------------
# 12. Multiple operations accumulate revision monotonically
# ---------------------------------------------------------------------------

def test_multiple_operations_accumulate_revision(tmp_store):
    tmp_store.assign_tag("doc-a", "tag1")   # +1
    tmp_store.assign_tag("doc-a", "tag2")   # +1
    tmp_store.unassign_tag("doc-a", "tag1") # +1
    tmp_store.rename_tag("tag2", "tag2-v2") # +1
    tmp_store.delete_tag("tag2-v2")         # +1
    assert _revision(tmp_store) == 5


# ---------------------------------------------------------------------------
# 13. get_corpus_fingerprint includes revision segment
# ---------------------------------------------------------------------------

def test_fingerprint_has_three_segments(tmp_store):
    fp = _fp(tmp_store)
    parts = fp.split(":")
    assert len(parts) == 3, f"Expected 3 segments, got: {fp!r}"


# ---------------------------------------------------------------------------
# 14. Fingerprint revision segment tracks get_revision()
# ---------------------------------------------------------------------------

def test_fingerprint_revision_segment_matches_get_revision(tmp_store):
    tmp_store.assign_tag("doc-a", "t")
    fp = _fp(tmp_store)
    revision_in_fp = int(fp.split(":")[2])
    assert revision_in_fp == _revision(tmp_store)


# ---------------------------------------------------------------------------
# 15. External-conn store: bump_revision does not auto-commit
#     (revision update is staged but caller must commit)
# ---------------------------------------------------------------------------

def test_external_conn_revision_not_autocommitted(tmp_path):
    db = tmp_path / "ext.sqlite"
    ext_conn = sqlite3.connect(str(db))
    store = MetadataStore(conn=ext_conn)
    store.initialize()
    store.upsert_document("d1", title="D1")

    before = store.get_revision()
    store.assign_tag("d1", "t")
    # Without explicit commit, a second connection should see the old revision
    # (WAL mode: reader sees last committed state)
    reader = sqlite3.connect(str(db))
    row = reader.execute(
        "SELECT value FROM cache_revision WHERE key = 'retrieval'"
    ).fetchone()
    # The update is uncommitted on external conn — reader sees 0 (or before value)
    assert row is None or row[0] == before
    reader.close()
    # Now commit explicitly
    ext_conn.commit()
    reader2 = sqlite3.connect(str(db))
    row2 = reader2.execute(
        "SELECT value FROM cache_revision WHERE key = 'retrieval'"
    ).fetchone()
    assert row2[0] == before + 1
    reader2.close()
    store.close()
    ext_conn.close()
