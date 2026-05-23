"""Tests for FTS5 write idempotency — v1.16.1 regression guard.

Bug fixed: docstore.add() used INSERT OR REPLACE INTO chunks_fts, which silently
inserts duplicate rows on re-ingest because FTS5 virtual tables have no UNIQUE
constraint on chunk_id. Fixed by DELETE-then-INSERT per chunk_id.

Each test that verifies the fixed behaviour is marked with a comment:
  # BUG: would fail before v1.16.1
"""

import pytest
from unittest.mock import MagicMock, patch

from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.maintenance.reconcile import reconcile, _has_issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, doc_id: str, text: str = "") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text or f"Unique content for chunk {chunk_id}.",
        "heading_path": "Section",
        "tipo": "texto",
        "posicion_relativa": 0.1,
        "n_tokens": 10,
        "line_start": 1,
        "line_end": 5,
    }


def _fts_count(ds: DocStore) -> int:
    return ds._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]


def _fts_count_for_doc(ds: DocStore, doc_id: str) -> int:
    return ds._conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE doc_id = ?", (doc_id,)
    ).fetchone()[0]


def _fts_dups(ds: DocStore) -> list:
    """Return chunk_ids that appear more than once in chunks_fts."""
    rows = ds._conn.execute(
        "SELECT chunk_id FROM chunks_fts GROUP BY chunk_id HAVING COUNT(*) > 1"
    ).fetchall()
    return [r[0] for r in rows]


@pytest.fixture
def ds(tmp_path):
    db = DocStore(db_path=tmp_path / "test.sqlite")
    db.initialize()
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Core idempotency
# ---------------------------------------------------------------------------

class TestFTS5WriteIdempotency:
    def test_single_add_creates_exactly_one_fts_row(self, ds):
        ds.add([_chunk("c1", "doc_a")])
        assert _fts_count(ds) == 1

    def test_add_same_chunk_twice_no_duplicate(self, ds):
        """BUG: would fail before v1.16.1 — second add inflated FTS count."""
        ds.add([_chunk("c1", "doc_a")])
        ds.add([_chunk("c1", "doc_a")])  # same chunk_id — re-ingest scenario
        assert _fts_count(ds) == 1
        assert _fts_dups(ds) == []

    def test_force_reingest_multiple_chunks_no_duplicates(self, ds):
        """BUG: would fail before v1.16.1 — each force pass added extra rows."""
        chunks = [_chunk(f"c{i}", "doc_a", text=f"Content {i}.") for i in range(5)]
        ds.add(chunks)
        assert _fts_count(ds) == 5

        ds.add(chunks)  # re-ingest with same chunk_ids
        assert _fts_count(ds) == 5  # BUG: was 10 before the fix
        assert _fts_dups(ds) == []

    def test_force_reingest_three_times_idempotent(self, ds):
        """BUG: N force passes would create N×original FTS rows."""
        chunks = [_chunk(f"x{i}", "doc_x") for i in range(3)]
        for _ in range(4):  # ingest 4 times
            ds.add(chunks)
        assert _fts_count(ds) == 3  # BUG: was 12 before the fix
        assert _fts_dups(ds) == []

    def test_multi_doc_reingest_each_doc_idempotent(self, ds):
        doc_a = [_chunk("a1", "doc_a"), _chunk("a2", "doc_a")]
        doc_b = [_chunk("b1", "doc_b"), _chunk("b2", "doc_b")]
        ds.add(doc_a)
        ds.add(doc_b)
        assert _fts_count(ds) == 4

        # Re-ingest doc_a only
        ds.add(doc_a)
        assert _fts_count(ds) == 4  # BUG: was 6 before the fix
        assert _fts_dups(ds) == []

    def test_updated_text_reflects_in_fts(self, ds):
        """After re-ingest, FTS5 must contain the NEW text, not both versions."""
        original = _chunk("cx", "doc_c", text="original text about SDMX")
        ds.add([original])

        updated = _chunk("cx", "doc_c", text="updated text about ISO")
        ds.add([updated])

        assert _fts_count(ds) == 1
        # Verify the updated text is searchable
        fts = FTSStore(db_path=ds.db_path)
        results = fts.query("ISO", top_k=5)
        ids = [r["id"] for r in results]
        assert "cx" in ids

    def test_old_text_not_searchable_after_update(self, ds):
        """The original text must not remain in FTS5 after re-ingest."""
        original = _chunk("cy", "doc_c", text="old unique phrase xyzzy1234")
        ds.add([original])

        updated = _chunk("cy", "doc_c", text="new different phrase foobar5678")
        ds.add([updated])

        fts = FTSStore(db_path=ds.db_path)
        results = fts.query("xyzzy1234", top_k=5)
        ids = [r["id"] for r in results]
        assert "cy" not in ids  # BUG: would be present before fix


# ---------------------------------------------------------------------------
# BM25 — no duplicate results
# ---------------------------------------------------------------------------

class TestBM25NoDuplicates:
    def test_bm25_query_no_duplicate_chunk_ids(self, ds):
        """Re-ingest must not cause the same chunk to appear twice in BM25 results."""
        chunks = [_chunk(f"q{i}", "doc_q", text=f"SDMX terminology concept {i}") for i in range(5)]
        ds.add(chunks)
        ds.add(chunks)  # force re-ingest

        fts = FTSStore(db_path=ds.db_path)
        results = fts.query("SDMX terminology", top_k=20)
        result_ids = [r["id"] for r in results]
        assert len(result_ids) == len(set(result_ids)), (
            f"BM25 returned duplicate chunk_ids: {result_ids}"
        )

    def test_bm25_scores_not_inflated_by_duplicates(self, ds):
        """After N re-ingests, BM25 score for a chunk must equal single-ingest score."""
        chunk = _chunk("score_c", "doc_s", text="SDMX unique scoring test phrase")
        ds.add([chunk])

        fts = FTSStore(db_path=ds.db_path)
        results_once = fts.query("unique scoring test", top_k=5)
        score_once = next((r["bm25_score"] for r in results_once if r["id"] == "score_c"), None)

        # Re-ingest 3 more times
        for _ in range(3):
            ds.add([chunk])

        results_multi = fts.query("unique scoring test", top_k=5)
        score_multi = next((r["bm25_score"] for r in results_multi if r["id"] == "score_c"), None)

        assert score_once is not None
        assert score_multi is not None
        assert abs(score_once - score_multi) < 1e-6, (
            f"BM25 score changed after re-ingest: {score_once} → {score_multi}"
        )


# ---------------------------------------------------------------------------
# delete_by_doc_id still cleans FTS5
# ---------------------------------------------------------------------------

class TestDeleteByDocIdFTS5:
    def test_delete_removes_fts5_rows(self, ds):
        ds.add([_chunk("d1", "doc_del"), _chunk("d2", "doc_del")])
        assert _fts_count_for_doc(ds, "doc_del") == 2

        ds.delete_by_doc_id("doc_del")
        assert _fts_count_for_doc(ds, "doc_del") == 0

    def test_delete_after_force_reingest_cleans_completely(self, ds):
        """After N re-ingests, delete must still leave FTS5 at 0 for that doc."""
        chunks = [_chunk(f"dr{i}", "doc_dr") for i in range(3)]
        for _ in range(3):
            ds.add(chunks)  # would create 9 rows before fix
        # After fix: still 3 rows

        ds.delete_by_doc_id("doc_dr")
        assert _fts_count_for_doc(ds, "doc_dr") == 0


# ---------------------------------------------------------------------------
# Batch ingest FTS5 idempotency (via _run_batch_ingest)
# ---------------------------------------------------------------------------

_VALID_MD = """\
# Alpha Document

## Section One

This document has enough content to produce valid chunks for the test suite.
Multiple sentences ensure the token count exceeds the minimum threshold value.
The chunking logic creates proper segment boundaries at heading transitions.

## Section Two

Second section with additional text content for validation and testing purposes.
This ensures the document produces multiple chunks with distinct heading paths.
"""

_patch_encode = patch("rag_lab.embedding.encoder.encode_chunks")
_patch_manifest = patch("rag_lab.ingest.manifest.create_manifest")


def _fake_encode(chunk_dicts, batch_size=8, device="cpu"):
    import numpy as np
    n = len(chunk_dicts)
    dense = np.random.rand(n, 1024).astype(np.float32)
    sparse = {c["chunk_id"]: {} for c in chunk_dicts}
    return dense, sparse


class TestBatchIngestFTS5Idempotency:
    @_patch_manifest
    @_patch_encode
    def test_force_reingest_single_doc_no_fts5_duplicates(
        self, mock_enc, mock_manifest, tmp_path
    ):
        """BUG: would fail before v1.16.1 — FTS5 inflated after --force."""
        from unittest.mock import MagicMock
        mock_enc.side_effect = _fake_encode
        p = tmp_path / "alpha.md"
        p.write_text(_VALID_MD, encoding="utf-8")

        db = DocStore(db_path=tmp_path / "db.sqlite")
        db.initialize()
        vs = MagicMock()
        vs._added_ids = []
        vs.add.side_effect = lambda ids, embeddings, documents, metadatas: vs._added_ids.extend(ids)
        vs.delete_by_doc_id.return_value = 0

        from rag_lab.cli_ingest import _run_batch_ingest
        _run_batch_ingest([p], db, vs, force=False, device="cpu", strict=False)
        chunks_first = db.count()
        fts_first = _fts_count(db)

        # Force re-ingest
        _run_batch_ingest([p], db, vs, force=True, device="cpu", strict=False)

        assert db.count() == chunks_first
        assert _fts_count(db) == fts_first  # BUG: was fts_first * 2 before fix
        assert _fts_dups(db) == []
        db.close()

    @_patch_manifest
    @_patch_encode
    def test_force_reingest_three_docs_no_fts5_duplicates(
        self, mock_enc, mock_manifest, tmp_path
    ):
        from unittest.mock import MagicMock
        mock_enc.side_effect = _fake_encode
        paths = []
        for i, suffix in enumerate(["A", "B", "C"]):
            p = tmp_path / f"doc{suffix}.md"
            p.write_text(_VALID_MD + f"\n\n## Extra {suffix}\n\nContent {i}.", encoding="utf-8")
            paths.append(p)

        db = DocStore(db_path=tmp_path / "db3.sqlite")
        db.initialize()
        vs = MagicMock()
        vs.add.side_effect = lambda ids, **kw: None
        vs.delete_by_doc_id.return_value = 0

        from rag_lab.cli_ingest import _run_batch_ingest
        _run_batch_ingest(paths, db, vs, force=False, device="cpu", strict=False)
        fts_first = _fts_count(db)

        _run_batch_ingest(paths, db, vs, force=True, device="cpu", strict=False)

        assert _fts_count(db) == fts_first  # BUG: was 2×fts_first before fix
        assert _fts_dups(db) == []
        db.close()


# ---------------------------------------------------------------------------
# Rollback — no FTS5 residue
# ---------------------------------------------------------------------------

class TestRollbackFTS5:
    def test_rollback_does_not_leave_fts5_rows(self, ds):
        ds.add([_chunk("rb1", "doc_rb"), _chunk("rb2", "doc_rb")])
        assert _fts_count_for_doc(ds, "doc_rb") == 2

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            from unittest.mock import MagicMock
            inner_vs = MagicMock()
            inner_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = inner_vs

            from rag_lab.ingest.transaction import IngestTransaction
            from rag_lab.ingest.transaction import IngestRunStore
            IngestRunStore(ds._conn).create("rb_run", "doc_rb", None)
            txn = IngestTransaction("doc_rb", None, ds)
            txn.run_id = "rb_run"
            txn.rollback()

        assert _fts_count_for_doc(ds, "doc_rb") == 0

    def test_rollback_after_reingest_cleans_all_fts5(self, ds):
        """After a re-ingest, rollback must still leave FTS5 at 0."""
        chunks = [_chunk("rb3", "doc_rr"), _chunk("rb4", "doc_rr")]
        ds.add(chunks)
        ds.add(chunks)  # before fix, this would leave 4 rows; after fix, still 2

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            from unittest.mock import MagicMock
            inner_vs = MagicMock()
            inner_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = inner_vs

            from rag_lab.ingest.transaction import IngestRunStore, IngestTransaction
            IngestRunStore(ds._conn).create("rr_run", "doc_rr", None)
            txn = IngestTransaction("doc_rr", None, ds)
            txn.run_id = "rr_run"
            txn.rollback()

        assert _fts_count_for_doc(ds, "doc_rr") == 0


# ---------------------------------------------------------------------------
# Reconcile — detects and repairs FTS5 duplicates
# ---------------------------------------------------------------------------

def _make_reconcile_ds_vs(ds: DocStore, chunk_ids: list):
    """Return (mock_ds_class, mock_vs_class) for reconcile patching."""
    mock_vs = MagicMock()
    mock_vs._collection.get.return_value = {"ids": chunk_ids}
    mock_vs_cls = MagicMock(return_value=mock_vs)

    # reconcile() calls ds.initialize() then ds.close(); we need the real ds to stay open
    mock_ds_instance = MagicMock()
    mock_ds_instance._conn = ds._conn
    mock_ds_instance.close = lambda: None
    mock_ds_cls = MagicMock(return_value=mock_ds_instance)
    return mock_ds_cls, mock_vs_cls


class TestReconcileFTS5Detection:
    def test_reconcile_detects_no_fts5_duplicates_on_clean_db(self, ds):
        ds.add([_chunk("c1", "doc_clean"), _chunk("c2", "doc_clean")])
        mock_ds_cls, mock_vs_cls = _make_reconcile_ds_vs(ds, ["c1", "c2"])

        with patch("rag_lab.maintenance.reconcile.DocStore", mock_ds_cls), \
             patch("rag_lab.maintenance.reconcile.VectorStore", mock_vs_cls):
            result = reconcile(quiet=True)

        assert result["fts_duplicate_chunk_ids"] == []

    def test_reconcile_detects_fts5_duplicates_when_present(self, ds):
        """Insert FTS5 duplicate manually to simulate a pre-fix database."""
        ds.add([_chunk("dup1", "doc_dup", text="important content")])
        # Manually inject duplicate (simulates pre-v1.16.1 INSERT OR REPLACE bug)
        ds._conn.execute(
            "INSERT INTO chunks_fts(chunk_id, doc_id, text) VALUES (?, ?, ?)",
            ("dup1", "doc_dup", "important content"),
        )
        ds._conn.commit()
        assert ds._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 2

        mock_ds_cls, mock_vs_cls = _make_reconcile_ds_vs(ds, ["dup1"])
        with patch("rag_lab.maintenance.reconcile.DocStore", mock_ds_cls), \
             patch("rag_lab.maintenance.reconcile.VectorStore", mock_vs_cls):
            result = reconcile(quiet=True)

        assert "dup1" in result["fts_duplicate_chunk_ids"]

    def test_reconcile_repair_fts_removes_duplicates(self, ds):
        """--repair-fts must deduplicate FTS5, leaving exactly one row per chunk_id."""
        ds.add([_chunk("r1", "doc_r", text="content one"), _chunk("r2", "doc_r", text="content two")])
        # Inject duplicates
        for cid, txt in [("r1", "content one"), ("r2", "content two")]:
            ds._conn.execute(
                "INSERT INTO chunks_fts(chunk_id, doc_id, text) VALUES (?, ?, ?)",
                (cid, "doc_r", txt),
            )
        ds._conn.commit()
        assert ds._conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 4

        mock_ds_cls, mock_vs_cls = _make_reconcile_ds_vs(ds, ["r1", "r2"])
        with patch("rag_lab.maintenance.reconcile.DocStore", mock_ds_cls), \
             patch("rag_lab.maintenance.reconcile.VectorStore", mock_vs_cls):
            result = reconcile(repair_fts=True, quiet=True)

        assert result["fts_repaired"] is True
        assert result["fts_duplicate_chunk_ids"] == []
        assert result["fts_count"] == 2

    def test_reconcile_has_issues_returns_true_for_fts_duplicates(self):
        result = {
            "chroma_orphans": [], "missing_from_chroma": [],
            "duplicate_chunk_ids": [],
            "fts_duplicate_chunk_ids": ["abc123"],
            "model_version_mismatches": [], "embedding_dim_mismatches": [],
            "sparse_format_version_mismatches": [], "orphaned_documents": [],
            "orphaned_document_tags": [], "stale_ingest_runs": [], "failed_ingest_runs": [],
        }
        assert _has_issues(result) is True

    def test_reconcile_has_issues_returns_false_when_no_fts_dups(self):
        result = {
            "chroma_orphans": [], "missing_from_chroma": [],
            "duplicate_chunk_ids": [],
            "fts_duplicate_chunk_ids": [],
            "model_version_mismatches": [], "embedding_dim_mismatches": [],
            "sparse_format_version_mismatches": [], "orphaned_documents": [],
            "orphaned_document_tags": [], "stale_ingest_runs": [], "failed_ingest_runs": [],
        }
        assert _has_issues(result) is False
