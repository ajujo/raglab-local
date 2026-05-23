"""Tests for the v1.16 parallel, resumable batch-ingest pipeline.

Scope: _run_batch_ingest, _prepare_no_db, _collect_paths, _manifest_has_hash,
       IngestBatchStore, IngestDocumentStore (via pipeline integration).

All DB-free — uses tmp_path SQLite and a mock VectorStore.
encode_chunks is mocked to return deterministic fake embeddings.

Patch notes:
  - encode_chunks is imported inside _run_batch_ingest, so patch at source:
      rag_lab.embedding.encoder.encode_chunks
  - create_manifest is imported inside _run_batch_ingest, so patch at source:
      rag_lab.ingest.manifest.create_manifest
  - DATA_DIR is imported inside _manifest_has_hash, so patch at source:
      rag_lab.config.DATA_DIR
  - IngestTransaction is accessed via module globals in _run_batch_ingest, so:
      rag_lab.ingest.transaction.IngestTransaction (for rollback VectorStore)
"""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_lab.storage.docstore import DocStore
from rag_lab.ingest.transaction import (
    IngestBatchStore,
    IngestDocumentStore,
)
from rag_lab.cli_ingest import (
    _collect_paths,
    _manifest_has_hash,
    _prepare_no_db,
    _run_batch_ingest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_MD_A = """\
# Document Alpha

## Section One

This is enough content to produce at least one valid chunk with more than fifty tokens.
The document discusses various important topics that are relevant to the test suite.
It contains multiple sentences to ensure the chunking logic creates proper segments.

## Section Two

Additional content to ensure the document passes all validation checks properly.
More text to meet token requirements and produce a meaningful chunk for testing purposes.
"""

_VALID_MD_B = """\
# Document Beta

## Part One

Beta document with different content to produce a distinct content hash.
The text here is deliberately different from alpha to test hash detection.
Multiple sentences ensure that the chunking and embedding pipeline is exercised.

## Part Two

Different second section to differentiate from the alpha document clearly.
Content is varied enough that no two documents share the same checksum value.
"""


def _make_md(tmp_path: Path, name: str, content: str = _VALID_MD_A) -> Path:
    p = tmp_path / f"{name}.md"
    p.write_text(content, encoding="utf-8")
    return p


def _fake_encode(chunk_dicts, batch_size=8, device="cpu"):
    """Return fake 1024-dim dense embeddings + empty sparse."""
    n = len(chunk_dicts)
    dense = np.random.rand(n, 1024).astype(np.float32)
    sparse = {c["chunk_id"]: {} for c in chunk_dicts}
    return dense, sparse


def _make_mock_vs():
    """Return a mock VectorStore that tracks added/deleted ids."""
    vs = MagicMock()
    vs._added_ids: list = []
    vs._deleted_doc_ids: list = []

    def _add(ids, embeddings, documents, metadatas):
        vs._added_ids.extend(ids)

    def _delete(doc_id):
        vs._deleted_doc_ids.append(doc_id)
        return 0

    vs.add.side_effect = _add
    vs.delete_by_doc_id.side_effect = _delete
    return vs


@pytest.fixture
def ds(tmp_path):
    db = DocStore(db_path=tmp_path / "test.sqlite")
    db.initialize()
    yield db
    db.close()


# Decorators used in multiple tests
_patch_encode = patch("rag_lab.embedding.encoder.encode_chunks", side_effect=_fake_encode)
_patch_manifest = patch("rag_lab.ingest.manifest.create_manifest")


# ---------------------------------------------------------------------------
# _collect_paths
# ---------------------------------------------------------------------------

class TestCollectPaths:
    def test_none_returns_config_sources(self):
        from rag_lab.config import SOURCES
        result = _collect_paths(None)
        assert result == list(SOURCES)

    def test_file_path_returns_single_element(self, tmp_path):
        p = _make_md(tmp_path, "docA")
        result = _collect_paths(str(p))
        assert result == [p]

    def test_directory_globs_md_files(self, tmp_path):
        _make_md(tmp_path, "doc1")
        _make_md(tmp_path, "doc2")
        (tmp_path / "other.txt").write_text("ignored")
        result = _collect_paths(str(tmp_path))
        assert len(result) == 2
        assert all(p.suffix == ".md" for p in result)

    def test_directory_result_is_sorted(self, tmp_path):
        _make_md(tmp_path, "z_doc")
        _make_md(tmp_path, "a_doc")
        _make_md(tmp_path, "m_doc")
        result = _collect_paths(str(tmp_path))
        assert result == sorted(result)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        result = _collect_paths(str(tmp_path))
        assert result == []


# ---------------------------------------------------------------------------
# _manifest_has_hash
# ---------------------------------------------------------------------------

class TestManifestHasHash:
    def test_returns_false_when_manifest_absent(self, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("x")
        with patch("rag_lab.config.DATA_DIR", tmp_path):
            assert _manifest_has_hash(p, "abc123") is False

    def test_returns_true_for_matching_entry(self, tmp_path):
        manifest = tmp_path / "ingested.jsonl"
        manifest.write_text(
            json.dumps({"doc_id": "doc", "hash": "aabbcc"}) + "\n"
        )
        p = tmp_path / "doc.md"
        with patch("rag_lab.config.DATA_DIR", tmp_path):
            assert _manifest_has_hash(p, "aabbcc") is True

    def test_returns_false_for_different_hash(self, tmp_path):
        manifest = tmp_path / "ingested.jsonl"
        manifest.write_text(
            json.dumps({"doc_id": "doc", "hash": "aabbcc"}) + "\n"
        )
        p = tmp_path / "doc.md"
        with patch("rag_lab.config.DATA_DIR", tmp_path):
            assert _manifest_has_hash(p, "xxyyzz") is False

    def test_returns_false_for_empty_hash(self, tmp_path):
        manifest = tmp_path / "ingested.jsonl"
        manifest.write_text(json.dumps({"doc_id": "doc", "hash": "abc"}) + "\n")
        p = tmp_path / "doc.md"
        with patch("rag_lab.config.DATA_DIR", tmp_path):
            assert _manifest_has_hash(p, "") is False


# ---------------------------------------------------------------------------
# _prepare_no_db
# ---------------------------------------------------------------------------

class TestPrepareNoDb:
    def test_valid_markdown_returns_ok_result(self, tmp_path):
        p = _make_md(tmp_path, "valid_doc")
        result = _prepare_no_db(p, strict=False)
        assert result.ok
        assert result.doc_id == "valid_doc"
        assert result.content_hash != ""
        assert len(result.chunk_dicts) > 0

    def test_missing_file_returns_failed_result(self, tmp_path):
        p = tmp_path / "nonexistent.md"
        result = _prepare_no_db(p, strict=False)
        assert not result.ok
        assert result.error is not None

    def test_doc_id_equals_stem(self, tmp_path):
        p = _make_md(tmp_path, "my_great_doc")
        result = _prepare_no_db(p, strict=False)
        assert result.doc_id == "my_great_doc"

    def test_different_content_produces_different_hashes(self, tmp_path):
        p1 = _make_md(tmp_path, "doc1", content=_VALID_MD_A)
        p2 = _make_md(tmp_path, "doc2", content=_VALID_MD_B)
        r1 = _prepare_no_db(p1, strict=False)
        r2 = _prepare_no_db(p2, strict=False)
        assert r1.content_hash != r2.content_hash

    def test_thread_safe_parallel_execution(self, tmp_path):
        """Two threads preparing different docs must not interfere."""
        p1 = _make_md(tmp_path, "thread_doc1", content=_VALID_MD_A)
        p2 = _make_md(tmp_path, "thread_doc2", content=_VALID_MD_B)
        results = {}

        def worker(p):
            results[p.stem] = _prepare_no_db(p, strict=False)

        t1 = threading.Thread(target=worker, args=(p1,))
        t2 = threading.Thread(target=worker, args=(p2,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert results["thread_doc1"].ok
        assert results["thread_doc2"].ok
        assert results["thread_doc1"].doc_id == "thread_doc1"
        assert results["thread_doc2"].doc_id == "thread_doc2"


# ---------------------------------------------------------------------------
# _run_batch_ingest: happy path
# ---------------------------------------------------------------------------

class TestBatchIngestHappyPath:
    @_patch_manifest
    @_patch_encode
    def test_single_doc_committed(self, mock_enc, mock_manifest, ds, tmp_path):
        p = _make_md(tmp_path, "alpha")
        vs = _make_mock_vs()
        summary = _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
        )
        assert summary["committed"] == 1
        assert summary["skipped"] == 0
        assert summary["failed"] == 0
        assert summary["total_chunks"] > 0

    @_patch_manifest
    @_patch_encode
    def test_directory_with_multiple_docs(self, mock_enc, mock_manifest, ds, tmp_path):
        _make_md(tmp_path, "doc_a", content=_VALID_MD_A)
        _make_md(tmp_path, "doc_b", content=_VALID_MD_B)
        _make_md(tmp_path, "doc_c", content=_VALID_MD_A + "\n\n## Extra\n\nExtra section.")
        paths = sorted(tmp_path.glob("*.md"))
        vs = _make_mock_vs()

        summary = _run_batch_ingest(
            paths=paths, doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
        )
        assert summary["committed"] == 3
        assert summary["skipped"] == 0
        assert summary["failed"] == 0

    @_patch_manifest
    @_patch_encode
    def test_chunks_stored_in_docstore(self, mock_enc, mock_manifest, ds, tmp_path):
        p = _make_md(tmp_path, "stored_doc")
        vs = _make_mock_vs()
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
        )
        assert ds.count_chunks("stored_doc") > 0

    @_patch_manifest
    @_patch_encode
    def test_ingest_documents_row_is_committed(self, mock_enc, mock_manifest, ds, tmp_path):
        p = _make_md(tmp_path, "row_doc")
        vs = _make_mock_vs()
        summary = _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
        )
        batch_id = summary["batch_id"]
        idc = IngestDocumentStore(ds._conn)
        rows = idc.list_by_batch(batch_id)
        assert len(rows) == 1
        assert rows[0]["status"] == "COMMITTED"
        assert rows[0]["chunks_count"] > 0

    @_patch_manifest
    @_patch_encode
    def test_ingest_batch_status_completed(self, mock_enc, mock_manifest, ds, tmp_path):
        p = _make_md(tmp_path, "batch_doc")
        vs = _make_mock_vs()
        summary = _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
        )
        bs = IngestBatchStore(ds._conn)
        batch = bs.get(summary["batch_id"])
        assert batch["status"] == "COMPLETED"
        assert batch["committed_docs"] == 1


# ---------------------------------------------------------------------------
# SKIPPED detection
# ---------------------------------------------------------------------------

class TestSkippedDetection:
    @_patch_manifest
    @_patch_encode
    def test_re_ingest_same_content_skipped(self, mock_enc, mock_manifest, ds, tmp_path):
        p = _make_md(tmp_path, "skip_doc")
        vs = _make_mock_vs()

        # First ingest
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
        )
        chunks_after_first = ds.count_chunks("skip_doc")

        # Second ingest — same file, same hash → SKIPPED
        vs2 = _make_mock_vs()
        summary2 = _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs2,
            force=False, device="cpu", strict=False,
        )
        assert summary2["skipped"] == 1
        assert summary2["committed"] == 0
        # No new chunks written
        assert ds.count_chunks("skip_doc") == chunks_after_first
        # No additional add calls
        assert len(vs2._added_ids) == 0

    @_patch_manifest
    @_patch_encode
    def test_force_flag_skips_hash_check(self, mock_enc, mock_manifest, ds, tmp_path):
        p = _make_md(tmp_path, "force_doc")
        vs1 = _make_mock_vs()
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs1,
            force=False, device="cpu", strict=False,
        )

        vs2 = _make_mock_vs()
        summary2 = _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs2,
            force=True, device="cpu", strict=False,
        )
        assert summary2["committed"] == 1
        assert summary2["skipped"] == 0

    @_patch_manifest
    @_patch_encode
    def test_modified_file_not_skipped(self, mock_enc, mock_manifest, ds, tmp_path):
        p = _make_md(tmp_path, "mod_doc", content=_VALID_MD_A)
        vs1 = _make_mock_vs()
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs1,
            force=False, device="cpu", strict=False,
        )

        # Modify the file — content hash changes
        p.write_text(_VALID_MD_A + "\n\n## Section Three\n\nNew content added here.", encoding="utf-8")
        vs2 = _make_mock_vs()
        summary2 = _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs2,
            force=False, device="cpu", strict=False,
        )
        assert summary2["committed"] == 1
        assert summary2["skipped"] == 0

    @patch("rag_lab.cli_ingest._manifest_has_hash", return_value=True)
    @_patch_encode
    def test_manifest_fallback_skips_when_hash_matches(
        self, mock_enc, mock_manifest_hash, ds, tmp_path
    ):
        """Manifest-based SKIPPED fires when ingest_documents has no COMMITTED row."""
        p = _make_md(tmp_path, "manifest_doc")
        vs = _make_mock_vs()
        # No prior ingest — ingest_documents is empty.
        # _manifest_has_hash is patched to return True → SKIPPED via manifest.
        summary = _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
        )
        assert summary["skipped"] == 1
        assert summary["committed"] == 0


# ---------------------------------------------------------------------------
# Validation failure
# ---------------------------------------------------------------------------

class TestValidationFailure:
    @_patch_encode
    def test_validation_failure_leaves_no_chunks(self, mock_enc, ds, tmp_path):
        """A document that fails validation must not write any chunks."""
        p = tmp_path / "bad_doc.md"
        p.write_text("content", encoding="utf-8")

        vs = _make_mock_vs()
        with patch("rag_lab.cli_ingest._prepare_no_db") as mock_prep:
            from rag_lab.cli_ingest import _PrepResult
            mock_prep.return_value = _PrepResult(
                source_path=p, doc_id="bad_doc", content_hash="abc",
                validation_summary="1 ERROR", ok=False, error="validation failed: 1 ERROR",
            )
            summary = _run_batch_ingest(
                paths=[p], doc_store=ds, vector_store=vs,
                force=False, device="cpu", strict=False,
            )

        assert summary["failed"] == 1
        assert summary["committed"] == 0
        assert ds.count_chunks("bad_doc") == 0
        assert len(vs._added_ids) == 0

    @_patch_encode
    def test_validation_failure_no_ingest_transaction_opened(self, mock_enc, ds, tmp_path):
        """IngestTransaction must never be entered for failed preparation."""
        p = tmp_path / "bad2.md"
        p.write_text("x", encoding="utf-8")
        vs = _make_mock_vs()

        with patch("rag_lab.cli_ingest._prepare_no_db") as mock_prep:
            from rag_lab.cli_ingest import _PrepResult
            mock_prep.return_value = _PrepResult(
                source_path=p, doc_id="bad2", content_hash="",
                ok=False, error="validation failed",
            )
            with patch("rag_lab.ingest.transaction.IngestTransaction.__enter__") as mock_enter:
                _run_batch_ingest(
                    paths=[p], doc_store=ds, vector_store=vs,
                    force=False, device="cpu", strict=False,
                )
                mock_enter.assert_not_called()


# ---------------------------------------------------------------------------
# Write failure → rollback
# ---------------------------------------------------------------------------

class TestWriteFailureRollback:
    @_patch_manifest
    @_patch_encode
    def test_write_failure_triggers_rollback(self, mock_enc, mock_manifest, ds, tmp_path):
        """If VectorStore.add raises, IngestTransaction rolls back SQLite chunks."""
        p = _make_md(tmp_path, "fail_write")
        vs = _make_mock_vs()
        vs.add.side_effect = RuntimeError("ChromaDB unavailable")

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            inner_vs = MagicMock()
            inner_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = inner_vs

            summary = _run_batch_ingest(
                paths=[p], doc_store=ds, vector_store=vs,
                force=False, device="cpu", strict=False,
            )

        assert summary["failed"] == 1
        assert summary["committed"] == 0
        assert ds.count_chunks("fail_write") == 0

    @_patch_manifest
    @_patch_encode
    def test_batch_status_partial_when_one_fails(self, mock_enc, mock_manifest, ds, tmp_path):
        """Batch marked PARTIAL when some docs fail and some commit."""
        p_ok = _make_md(tmp_path, "ok_doc", content=_VALID_MD_A)
        p_bad = _make_md(tmp_path, "bad_doc", content=_VALID_MD_B)

        calls = [0]

        def add_side_effect(ids, embeddings, documents, metadatas):
            calls[0] += 1
            if calls[0] == 2:  # second doc fails
                raise RuntimeError("forced failure")

        vs = _make_mock_vs()
        vs.add.side_effect = add_side_effect

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            inner_vs = MagicMock()
            inner_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = inner_vs

            summary = _run_batch_ingest(
                paths=[p_ok, p_bad], doc_store=ds, vector_store=vs,
                force=False, device="cpu", strict=False,
            )

        bs = IngestBatchStore(ds._conn)
        batch = bs.get(summary["batch_id"])
        assert batch["status"] == "PARTIAL"


# ---------------------------------------------------------------------------
# Doc isolation: failure in doc N does not corrupt doc N-1
# ---------------------------------------------------------------------------

class TestDocIsolation:
    @_patch_manifest
    @_patch_encode
    def test_doc2_fails_doc1_committed_doc2_clean(
        self, mock_enc, mock_manifest, ds, tmp_path
    ):
        p1 = _make_md(tmp_path, "doc_one", content=_VALID_MD_A)
        p2 = _make_md(tmp_path, "doc_two", content=_VALID_MD_B)
        p3 = _make_md(tmp_path, "doc_three", content=_VALID_MD_A + "\n\n## C\n\nContent c.")

        call_count = [0]

        def add_side_effect(ids, embeddings, documents, metadatas):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("doc_two write failure")

        vs = _make_mock_vs()
        vs.add.side_effect = add_side_effect

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            inner_vs = MagicMock()
            inner_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = inner_vs

            summary = _run_batch_ingest(
                paths=[p1, p2, p3], doc_store=ds, vector_store=vs,
                force=False, device="cpu", strict=False,
            )

        # doc_one and doc_three committed; doc_two rolled back
        assert ds.count_chunks("doc_one") > 0
        assert ds.count_chunks("doc_three") > 0
        assert ds.count_chunks("doc_two") == 0

        idc = IngestDocumentStore(ds._conn)
        rows = {r["doc_id"]: r for r in idc.list_by_batch(summary["batch_id"])}
        assert rows["doc_one"]["status"] == "COMMITTED"
        assert rows["doc_two"]["status"] == "ROLLED_BACK"
        assert rows["doc_three"]["status"] == "COMMITTED"

    @_patch_manifest
    @_patch_encode
    def test_failed_doc_fts5_is_clean(self, mock_enc, mock_manifest, ds, tmp_path):
        """Rolled-back doc must have zero FTS5 rows."""
        p = _make_md(tmp_path, "fts_fail_doc")
        vs = _make_mock_vs()
        vs.add.side_effect = RuntimeError("inject fts fail")

        with patch("rag_lab.ingest.transaction.VectorStore") as mock_vs_cls:
            inner_vs = MagicMock()
            inner_vs.delete_by_doc_id.return_value = 0
            mock_vs_cls.return_value = inner_vs

            _run_batch_ingest(
                paths=[p], doc_store=ds, vector_store=vs,
                force=False, device="cpu", strict=False,
            )

        count = ds._conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE doc_id = ?", ("fts_fail_doc",)
        ).fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class TestResume:
    @_patch_manifest
    @_patch_encode
    def test_resume_continues_pending_docs(self, mock_enc, mock_manifest, ds, tmp_path):
        p1 = _make_md(tmp_path, "resume_doc1", content=_VALID_MD_A)
        p2 = _make_md(tmp_path, "resume_doc2", content=_VALID_MD_B)

        # Create batch with one COMMITTED and one PENDING
        bs = IngestBatchStore(ds._conn)
        idc = IngestDocumentStore(ds._conn)
        batch_id = bs.create(source_path="test")
        bs.update(batch_id, total_docs=2)

        # p1: mark as already COMMITTED (simulate partial prior run, no actual chunks)
        id1 = idc.create(batch_id, p1.stem, str(p1))
        idc.set_status(id1, "COMMITTED", chunks_count=5, finished_at="2026-01-01T00:00:00")

        # p2 is PENDING (will be resumed)
        idc.create(batch_id, p2.stem, str(p2))

        vs = _make_mock_vs()
        summary = _run_batch_ingest(
            paths=[p1, p2], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
            resume_batch_id=batch_id,
        )

        # Only p2 should have been processed (p1 was COMMITTED, skipped by resume logic)
        assert summary["committed"] == 1
        assert ds.count_chunks("resume_doc2") > 0
        # p1 was never actually ingested in this test (only its row was marked COMMITTED)
        assert ds.count_chunks("resume_doc1") == 0

    @_patch_manifest
    @_patch_encode
    def test_resume_increments_retry_count_for_failed(
        self, mock_enc, mock_manifest, ds, tmp_path
    ):
        p = _make_md(tmp_path, "retry_count_doc")
        bs = IngestBatchStore(ds._conn)
        idc = IngestDocumentStore(ds._conn)
        batch_id = bs.create(source_path="test")
        bs.update(batch_id, total_docs=1)

        # Pre-create row as FAILED with retry_count=1
        row_id = idc.create(batch_id, p.stem, str(p))
        idc.set_status(row_id, "FAILED", retry_count=1, error_message="prior error")

        vs = _make_mock_vs()
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
            resume_batch_id=batch_id,
        )

        row = idc.get_by_id(row_id)
        assert row["retry_count"] == 2  # incremented from 1

    @_patch_manifest
    @_patch_encode
    def test_resume_no_duplicate_chunks(self, mock_enc, mock_manifest, ds, tmp_path):
        """Re-running with force=True must not duplicate chunks (INSERT OR REPLACE)."""
        p = _make_md(tmp_path, "no_dup_doc")
        vs1 = _make_mock_vs()

        # First full ingest
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs1,
            force=False, device="cpu", strict=False,
        )
        chunks_after_first = ds.count_chunks("no_dup_doc")
        assert chunks_after_first > 0

        # Second ingest with force=True — same chunk_ids → INSERT OR REPLACE
        vs2 = _make_mock_vs()
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs2,
            force=True, device="cpu", strict=False,
        )
        # Same chunk count — no duplication
        assert ds.count_chunks("no_dup_doc") == chunks_after_first


# ---------------------------------------------------------------------------
# Workers > 1
# ---------------------------------------------------------------------------

class TestParallelWorkers:
    @_patch_manifest
    @_patch_encode
    def test_workers_2_produces_same_result_as_1(
        self, mock_enc, mock_manifest, tmp_path
    ):
        """Multiple workers produce the same committed output as serial mode."""
        paths = [
            _make_md(tmp_path, f"par_doc_{i}",
                     content=_VALID_MD_A + f"\n\n## Unique {i}\n\nContent {i}.")
            for i in range(4)
        ]

        # Run with workers=1
        db1 = DocStore(db_path=tmp_path / "db1.sqlite")
        db1.initialize()
        vs1 = _make_mock_vs()
        s1 = _run_batch_ingest(
            paths=paths, doc_store=db1, vector_store=vs1,
            force=False, device="cpu", strict=False, workers=1,
        )
        chunks1 = db1.count()
        db1.close()

        # Run with workers=2 on a fresh DB
        db2 = DocStore(db_path=tmp_path / "db2.sqlite")
        db2.initialize()
        vs2 = _make_mock_vs()
        s2 = _run_batch_ingest(
            paths=paths, doc_store=db2, vector_store=vs2,
            force=False, device="cpu", strict=False, workers=2,
        )
        chunks2 = db2.count()
        db2.close()

        assert s1["committed"] == s2["committed"] == 4
        assert chunks1 == chunks2

    @_patch_manifest
    @_patch_encode
    def test_workers_only_main_thread_writes_db(
        self, mock_enc, mock_manifest, ds, tmp_path
    ):
        """Verify single-writer guarantee: worker threads must not call ds.add."""
        paths = [
            _make_md(tmp_path, f"sw_doc_{i}",
                     content=_VALID_MD_A + f"\n\n## S{i}\n\nContent {i}.")
            for i in range(3)
        ]
        write_threads: list = []
        original_add = ds.add

        def recording_add(chunks):
            write_threads.append(threading.current_thread().name)
            return original_add(chunks)

        ds.add = recording_add
        vs = _make_mock_vs()

        _run_batch_ingest(
            paths=paths, doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False, workers=2,
        )

        # Every write must have come from the test thread (not an ingest-prep worker)
        for name in write_threads:
            assert "ingest-prep" not in name, (
                f"Worker thread {name!r} wrote to DocStore — single-writer violated"
            )


# ---------------------------------------------------------------------------
# Cache fingerprint
# ---------------------------------------------------------------------------

class TestCacheFingerprint:
    @_patch_manifest
    @_patch_encode
    def test_fingerprint_changes_after_commit(self, mock_enc, mock_manifest, ds, tmp_path):
        from rag_lab.cache.query_cache import get_corpus_fingerprint
        fp_before = get_corpus_fingerprint(ds._conn)
        p = _make_md(tmp_path, "fp_doc")
        vs = _make_mock_vs()
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
        )
        fp_after = get_corpus_fingerprint(ds._conn)
        assert fp_before != fp_after

    @_patch_manifest
    @_patch_encode
    def test_fingerprint_unchanged_when_all_skipped(
        self, mock_enc, mock_manifest, ds, tmp_path
    ):
        from rag_lab.cache.query_cache import get_corpus_fingerprint
        p = _make_md(tmp_path, "fp_skip_doc")
        vs = _make_mock_vs()
        # First ingest
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs,
            force=False, device="cpu", strict=False,
        )
        fp_after_first = get_corpus_fingerprint(ds._conn)

        # Re-ingest same file → SKIPPED
        vs2 = _make_mock_vs()
        _run_batch_ingest(
            paths=[p], doc_store=ds, vector_store=vs2,
            force=False, device="cpu", strict=False,
        )
        fp_after_second = get_corpus_fingerprint(ds._conn)
        assert fp_after_first == fp_after_second

    @_patch_encode
    def test_fingerprint_unchanged_on_validation_failure(
        self, mock_enc, ds, tmp_path
    ):
        from rag_lab.cache.query_cache import get_corpus_fingerprint
        p = tmp_path / "fp_fail.md"
        p.write_text("x", encoding="utf-8")
        vs = _make_mock_vs()

        fp_before = get_corpus_fingerprint(ds._conn)

        with patch("rag_lab.cli_ingest._prepare_no_db") as mock_prep:
            from rag_lab.cli_ingest import _PrepResult
            mock_prep.return_value = _PrepResult(
                source_path=p, doc_id="fp_fail", content_hash="",
                ok=False, error="validation failed",
            )
            _run_batch_ingest(
                paths=[p], doc_store=ds, vector_store=vs,
                force=False, device="cpu", strict=False,
            )

        fp_after = get_corpus_fingerprint(ds._conn)
        assert fp_before == fp_after


# ---------------------------------------------------------------------------
# IngestDocumentStore unit tests
# ---------------------------------------------------------------------------

class TestIngestDocumentStore:
    @pytest.fixture
    def idc(self, ds):
        return IngestDocumentStore(ds._conn)

    @pytest.fixture
    def batch_id(self, ds):
        bs = IngestBatchStore(ds._conn)
        return bs.create(source_path="test")

    def test_create_returns_int_id(self, idc, batch_id):
        row_id = idc.create(batch_id, "doc_x", "/path/doc_x.md")
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_find_committed_returns_false_for_unknown(self, idc):
        assert idc.find_committed("no_doc", "abc") is False

    def test_find_committed_returns_false_for_empty_hash(self, idc, batch_id):
        row_id = idc.create(batch_id, "doc_empty", "/path.md")
        idc.set_status(row_id, "COMMITTED")
        assert idc.find_committed("doc_empty", "") is False

    def test_find_committed_returns_true_after_commit(self, idc, batch_id):
        row_id = idc.create(batch_id, "doc_yes", "/path.md", content_hash="cafebabe")
        idc.set_status(row_id, "COMMITTED")
        assert idc.find_committed("doc_yes", "cafebabe") is True

    def test_find_committed_returns_false_for_different_hash(self, idc, batch_id):
        row_id = idc.create(batch_id, "doc_hash", "/path.md", content_hash="aaa")
        idc.set_status(row_id, "COMMITTED")
        assert idc.find_committed("doc_hash", "bbb") is False

    def test_find_committed_false_for_non_committed_status(self, idc, batch_id):
        row_id = idc.create(batch_id, "doc_fail", "/path.md", content_hash="aaa")
        idc.set_status(row_id, "FAILED")
        assert idc.find_committed("doc_fail", "aaa") is False

    def test_list_resumable_returns_pending_failed_rolled_back(self, idc, batch_id):
        id1 = idc.create(batch_id, "d1", "/p1.md")
        id2 = idc.create(batch_id, "d2", "/p2.md")
        id3 = idc.create(batch_id, "d3", "/p3.md")
        id4 = idc.create(batch_id, "d4", "/p4.md")
        idc.set_status(id2, "COMMITTED")
        idc.set_status(id3, "FAILED")
        idc.set_status(id4, "ROLLED_BACK")
        resumable = idc.list_resumable(batch_id)
        statuses = {r["doc_id"]: r["status"] for r in resumable}
        assert "d1" in statuses and statuses["d1"] == "PENDING"
        assert "d2" not in statuses
        assert "d3" in statuses
        assert "d4" in statuses

    def test_list_failed_global(self, idc, batch_id):
        id1 = idc.create(batch_id, "f1", "/p1.md")
        id2 = idc.create(batch_id, "f2", "/p2.md")
        idc.set_status(id1, "FAILED")
        idc.set_status(id2, "COMMITTED")
        failed = idc.list_failed()
        doc_ids = [r["doc_id"] for r in failed]
        assert "f1" in doc_ids
        assert "f2" not in doc_ids

    def test_get_by_id(self, idc, batch_id):
        row_id = idc.create(batch_id, "lookup_doc", "/p.md")
        row = idc.get_by_id(row_id)
        assert row is not None
        assert row["doc_id"] == "lookup_doc"

    def test_get_by_id_nonexistent_returns_none(self, idc):
        assert idc.get_by_id(99999) is None


# ---------------------------------------------------------------------------
# IngestBatchStore unit tests
# ---------------------------------------------------------------------------

class TestIngestBatchStore:
    @pytest.fixture
    def bs(self, ds):
        return IngestBatchStore(ds._conn)

    def test_create_returns_12_char_hex(self, bs):
        bid = bs.create()
        assert len(bid) == 12
        assert all(c in "0123456789abcdef" for c in bid)

    def test_get_returns_in_progress(self, bs):
        bid = bs.create()
        batch = bs.get(bid)
        assert batch is not None
        assert batch["status"] == "IN_PROGRESS"

    def test_get_nonexistent_returns_none(self, bs):
        assert bs.get("nonexistent") is None

    def test_finalize_completed_when_all_committed(self, bs, ds):
        idc = IngestDocumentStore(ds._conn)
        bid = bs.create()
        id1 = idc.create(bid, "d1", "/p1.md")
        id2 = idc.create(bid, "d2", "/p2.md")
        idc.set_status(id1, "COMMITTED", chunks_count=3)
        idc.set_status(id2, "COMMITTED", chunks_count=5)
        bs.finalize(bid)
        batch = bs.get(bid)
        assert batch["status"] == "COMPLETED"
        assert batch["committed_docs"] == 2
        assert batch["total_chunks"] == 8

    def test_finalize_failed_when_all_failed(self, bs, ds):
        idc = IngestDocumentStore(ds._conn)
        bid = bs.create()
        id1 = idc.create(bid, "d1", "/p1.md")
        idc.set_status(id1, "FAILED")
        bs.finalize(bid)
        assert bs.get(bid)["status"] == "FAILED"

    def test_finalize_partial_when_mixed(self, bs, ds):
        idc = IngestDocumentStore(ds._conn)
        bid = bs.create()
        id1 = idc.create(bid, "d1", "/p1.md")
        id2 = idc.create(bid, "d2", "/p2.md")
        idc.set_status(id1, "COMMITTED", chunks_count=2)
        idc.set_status(id2, "FAILED")
        bs.finalize(bid)
        assert bs.get(bid)["status"] == "PARTIAL"

    def test_get_latest_incomplete_returns_most_recent(self, bs, ds):
        bid1 = bs.create()
        bid2 = bs.create()
        bid3 = bs.create()
        # Finalize bid1 and bid2
        bs.update(bid1, status="COMPLETED")
        bs.update(bid2, status="FAILED")
        result = bs.get_latest_incomplete()
        assert result == bid3

    def test_get_latest_incomplete_returns_none_when_all_done(self, bs):
        bid = bs.create()
        bs.update(bid, status="COMPLETED")
        assert bs.get_latest_incomplete() is None

    def test_list_batches_contains_all_created(self, bs):
        bid1 = bs.create()
        bid2 = bs.create()
        bid3 = bs.create()
        batches = bs.list_batches(limit=10)
        ids = {b["batch_id"] for b in batches}
        assert bid1 in ids
        assert bid2 in ids
        assert bid3 in ids

    def test_list_batches_respects_limit(self, bs):
        for _ in range(5):
            bs.create()
        assert len(bs.list_batches(limit=3)) == 3
