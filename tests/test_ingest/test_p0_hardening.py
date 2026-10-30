"""P0 hardening regressions for the adversarial-review findings (round 2).

Covers: fail-closed vector snapshot/restore, journal-before-writes, recovery
superseding, legacy run ownership, identity ownership across batches/resume,
manifest multi-entry, resolved identity in the control table, and filters that
must not fail open when the DocStore is lazily unopened.
"""

import json

import numpy as np
import pytest

from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore
from rag_lab.ingest import versioning
from rag_lab.ingest.transaction import IngestRunStore, IngestTransaction
from rag_lab.exceptions import VersioningError


# ---------------------------------------------------------------------------
# Strict fake vector store (declares supports_versioning -> fail-closed)
# ---------------------------------------------------------------------------

class StrictFakeVS:
    supports_versioning = True

    def __init__(self):
        self.vectors = {}  # chunk_id -> (doc_id, emb_list, document, metadata)
        self.fail_snapshot = False
        self.fail_delete = False
        self.fail_restore = False

    def _add(self, doc_id, cid):
        self.vectors[cid] = (doc_id, [0.1] * 8, f"doc-{cid}", {"doc_id": doc_id})

    def get_by_doc_id(self, doc_id):
        if self.fail_snapshot:
            raise RuntimeError("snapshot boom")
        ids, embs, docs, metas = [], [], [], []
        for cid, (did, emb, doc, meta) in self.vectors.items():
            if did == doc_id:
                ids.append(cid); embs.append(emb); docs.append(doc); metas.append(meta)
        return {"ids": ids, "embeddings": embs, "documents": docs, "metadatas": metas}

    def delete_by_doc_id(self, doc_id):
        if self.fail_delete:
            raise RuntimeError("delete boom")
        for cid in [c for c, v in self.vectors.items() if v[0] == doc_id]:
            del self.vectors[cid]
        return 0

    def delete_ids(self, ids):
        if self.fail_delete:
            raise RuntimeError("delete boom")
        for cid in ids:
            self.vectors.pop(cid, None)
        return len(ids)

    def restore_vectors(self, ids, embeddings, documents, metadatas):
        if self.fail_restore:
            raise RuntimeError("restore boom")
        for cid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self.vectors[cid] = (meta.get("doc_id", ""), emb, doc, meta)


def _chunk(cid, doc_id, text=None):
    return {
        "chunk_id": cid, "doc_id": doc_id, "text": text or f"text {cid}",
        "heading_path": "H", "tipo": "texto", "posicion_relativa": 0.0,
        "n_tokens": 3, "line_start": 0, "line_end": 1,
    }


@pytest.fixture
def ds(tmp_path):
    d = DocStore(db_path=tmp_path / "h.sqlite")
    d.initialize()
    yield d
    d.close()


# ---------------------------------------------------------------------------
# SOL-01 — vector failures must not be swallowed
# ---------------------------------------------------------------------------

def test_begin_aborts_when_snapshot_fails(ds):
    vs = StrictFakeVS()
    vs._add("d", "old")
    ds.add([_chunk("old", "d")])
    vs.fail_snapshot = True
    with pytest.raises(VersioningError):
        versioning.begin_replace("d", ds._conn, vs)
    # nothing was overwritten and no attempt was journaled
    assert {r[0] for r in ds._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id='d'")} == {"old"}
    assert ds._conn.execute("SELECT COUNT(*) FROM version_ops").fetchone()[0] == 0


def test_transaction_enter_aborts_before_writes(ds):
    vs = StrictFakeVS()
    vs._add("d", "old")
    ds.add([_chunk("old", "d")])
    vs.fail_snapshot = True
    with pytest.raises(VersioningError):
        IngestTransaction("d", None, ds, vector_store=vs, chunk_ids=["new"]).__enter__()
    assert {r[0] for r in ds._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id='d'")} == {"old"}


def test_rollback_restore_failure_left_pending(ds):
    vs = StrictFakeVS()
    vs._add("d", "old")
    ds.add([_chunk("old", "d")])
    handle = versioning.begin_replace("d", ds._conn, vs)
    ds.add([_chunk("new", "d")])
    vs.fail_restore = True
    with pytest.raises(VersioningError):
        versioning.rollback_replace(handle, ds._conn, vs)
    # The operation must remain pending (not DONE) so recovery can retry.
    states = {r[0] for r in ds._conn.execute("SELECT state FROM version_ops")}
    assert "RESTORING" in states and "DONE" not in states


# ---------------------------------------------------------------------------
# SOL-03 / SOL-05 — journal before writes; completed rollback supersedes commit
# ---------------------------------------------------------------------------

def test_writing_operation_journaled_before_commit(ds):
    vs = StrictFakeVS()
    vs._add("d", "old")
    ds.add([_chunk("old", "d")])
    versioning.begin_replace("d", ds._conn, vs)
    states = {r[0] for r in ds._conn.execute("SELECT state FROM version_ops")}
    assert "WRITING" in states
    # Recovery of a WRITING attempt restores the previous version.
    versioning.recover_pending_ops(ds._conn, vs)
    assert {r[0] for r in ds._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id='d'")} == {"old"}
    assert ds._conn.execute("SELECT COUNT(*) FROM version_ops WHERE state!='DONE'").fetchone()[0] == 0


def test_completed_rollback_supersedes_pending_commit(ds):
    vs = StrictFakeVS()
    vs._add("d", "old")
    ds.add([_chunk("old", "d")])
    handle = versioning.begin_replace("d", ds._conn, vs)
    # Simulate a commit that crashed before publishing, leaving obsolete=['old'].
    ds._conn.execute(
        "INSERT INTO version_ops(doc_id, op, run_id, state, payload_json) "
        "VALUES ('d', 'COMMIT', 'run2', 'COMMITTING', ?)",
        (json.dumps({"new_run": "run2", "new_ids": [], "obsolete": ["old"]}),),
    )
    ds._conn.commit()
    # A successful rollback of the attempt must cancel that pending commit.
    versioning.rollback_replace(handle, ds._conn, vs)
    versioning.recover_pending_ops(ds._conn, vs)
    assert {r[0] for r in ds._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id='d'")} == {"old"}


# ---------------------------------------------------------------------------
# SOL-06 — legacy ownership is deterministic and excludes the running attempt
# ---------------------------------------------------------------------------

def test_latest_committed_run_excludes_in_progress(ds):
    store = IngestRunStore(ds._conn)
    store.create("r_old", "d", None)
    store.update("r_old", status="COMMITTED")
    store.create("r_new", "d", None)  # IN_PROGRESS
    assert versioning._latest_committed_run(ds._conn, "d") == "r_old"


def test_begin_does_not_capture_running_attempt_as_previous(ds):
    from rag_lab.ingest import versioning as v
    vs = StrictFakeVS()
    # Existing committed version from a legacy run.
    ds.add([_chunk("old", "d")])
    store = IngestRunStore(ds._conn)
    store.create("legacy_old", "d", None)
    store.update("legacy_old", status="COMMITTED")
    # A new attempt is created (IN_PROGRESS) before begin_replace.
    store.create("attempt_new", "d", None)

    handle = v.begin_replace("d", ds._conn, vs)
    assert handle.old_run_id == "legacy_old"


# ---------------------------------------------------------------------------
# SOL-08 — manifest must keep distinct explicit identities
# ---------------------------------------------------------------------------

def test_manifest_records_multiple_explicit_ids(tmp_path, monkeypatch):
    from rag_lab.ingest import manifest as m
    monkeypatch.setattr(m, "DATA_DIR", tmp_path)
    a = tmp_path / "fileA.md"; a.write_text("# A\n")
    b = tmp_path / "fileB.md"; b.write_text("# B\n")
    m.create_manifest(a, force=True, doc_id="id-a")
    m.create_manifest(b, force=True, doc_id="id-b")
    entries = [json.loads(l) for l in (tmp_path / "ingested.jsonl").read_text().splitlines() if l.strip()]
    ids = {e["doc_id"] for e in entries}
    assert ids == {"id-a", "id-b"}


# ---------------------------------------------------------------------------
# SOL-09 + SOL-07 — resolved identity in control table; cross-batch collision
# ---------------------------------------------------------------------------

def _fake_encode(chunk_dicts, batch_size=8, device="cpu"):
    n = len(chunk_dicts)
    return np.random.RandomState(0).rand(n, 1024).astype(np.float32), {
        c["chunk_id"]: {} for c in chunk_dicts
    }


@pytest.fixture
def batch_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    monkeypatch.setattr("rag_lab.ingest.cleaner.DATA_DIR", data_dir)
    monkeypatch.setattr("rag_lab.ingest.manifest.DATA_DIR", data_dir)
    monkeypatch.setattr("rag_lab.config.DATA_DIR", data_dir)
    monkeypatch.setattr("rag_lab.embedding.encoder.encode_chunks", _fake_encode)
    d = DocStore(db_path=tmp_path / "b.sqlite"); d.initialize()
    vs = VectorStore(storage_path=tmp_path / "chroma"); vs.initialize()
    yield {"ds": d, "vs": vs, "tmp": tmp_path}
    d.close()


_WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "


def _front(tag, doc_id):
    return f"---\ndoc_id: {doc_id}\ntitle: T {tag}\n---\n\n# {tag}\n\n## S\n\n" + (_WORDS * 60)


def test_resolved_identity_recorded_and_cross_batch_collision(batch_env):
    from rag_lab.cli_ingest import _run_batch_ingest
    ds, vs, tmp = batch_env["ds"], batch_env["vs"], batch_env["tmp"]
    (tmp / "a").mkdir(); (tmp / "b").mkdir()
    p1 = tmp / "a" / "guia.md"; p1.write_text(_front("Uno", "shared-id"))
    p2 = tmp / "b" / "guia.md"; p2.write_text(_front("Dos", "shared-id"))

    s1 = _run_batch_ingest(
        paths=[p1], doc_store=ds, vector_store=vs, force=False, device="cpu", strict=False,
    )
    assert s1["committed"] == 1
    # SOL-09: the control table records the resolved frontmatter id.
    row = ds._conn.execute(
        "SELECT doc_id FROM ingest_documents ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == "shared-id"

    # SOL-07: a second batch from a different path claiming the same id is refused.
    s2 = _run_batch_ingest(
        paths=[p2], doc_store=ds, vector_store=vs, force=False, device="cpu", strict=False,
    )
    assert s2["committed"] == 0
    assert s2["failed"] == 1
    owner = ds._conn.execute("SELECT path FROM documents WHERE doc_id='shared-id'").fetchone()
    assert owner[0] == str(p1)


# ---------------------------------------------------------------------------
# SOL-12 — filter must not fail open when DocStore is lazily unopened
# ---------------------------------------------------------------------------

def test_filter_fails_closed_when_docstore_unopened(tmp_path):
    db = tmp_path / "f.sqlite"
    ds = DocStore(db_path=db); ds.initialize()
    ds._conn.execute("INSERT INTO documents(doc_id, title, status) VALUES ('D','D','active')")
    ds._conn.commit()
    ds.add([_chunk("d1", "D")])
    fts = FTSStore(db_path=db); fts.initialize()
    vs = VectorStore(storage_path=tmp_path / "chroma"); vs.initialize()
    vs.add(
        ids=["d1"], embeddings=np.ones((1, 1024), dtype=np.float32),
        documents=["alpha"], metadatas=[{"doc_id": "D", "heading_path": "H"}],
    )
    ds.close()  # unopened instance passed to hybrid_search

    lazy_ds = DocStore(db_path=db)  # NOT initialized
    from rag_lab.retrieval.hybrid_search import hybrid_search
    from rag_lab.retrieval.filters import FilterSpec

    out = hybrid_search(
        "alpha", vs, lazy_ds, fts, query_dense=np.ones(1024, dtype=np.float32),
        top_k=10, diversity_mode="off",
        filter_spec=FilterSpec(source_id="missing", status=None),
    )
    assert out == []
    fts.close()


# ---------------------------------------------------------------------------
# SOL-10 — conservative estimator for pathological (non-text) inputs
# ---------------------------------------------------------------------------

def test_estimate_tokens_conservative_for_emoji():
    from rag_lab.utils.tokenizer import estimate_tokens, count_tokens
    emoji = "\U0001F9EC" * 5000
    assert count_tokens(emoji) < 100  # embedding tokenizer collapses it
    est = estimate_tokens(emoji)
    assert est >= len(emoji.encode("utf-8")) // 2


# ---------------------------------------------------------------------------
# SOL-11 — callers must use only the chunks actually sent to the model
# ---------------------------------------------------------------------------

def test_cli_uses_only_used_chunks(monkeypatch):
    from unittest.mock import Mock
    import rag_lab.cli as cli
    from rag_lab.generation.prompt_builder import PromptPlan

    monkeypatch.setattr(cli, "VectorStore", lambda: Mock())
    monkeypatch.setattr(cli, "FTSStore", lambda: Mock())
    monkeypatch.setattr(cli, "DocStore", lambda: Mock())
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli, "encode_chunks", lambda *a, **k: ([[0.0]], {"x": {1: 1.0}}))
    monkeypatch.setattr(cli, "hybrid_search", lambda *a, **k: [
        {"chunk_id": "keep", "doc_id": "d", "text": "a", "score": 1.0},
        {"chunk_id": "drop", "doc_id": "d", "text": "b", "score": 0.5},
    ])
    monkeypatch.setattr(cli, "rerank", lambda q, rows, **k: rows)
    plan = PromptPlan(
        system="s", user="u",
        used_chunks=[{"chunk_id": "keep", "doc_id": "d", "text": "a", "score": 1.0}],
        dropped_chunks=[{"chunk_id": "drop", "doc_id": "d", "text": "b", "score": 0.5}],
        prompt_tokens=10, output_tokens=10,
    )
    monkeypatch.setattr(cli, "build_prompt_plan", lambda q, rows: plan)
    monkeypatch.setattr(cli, "generate_response", lambda *a, **k: "resp")

    captured = {}

    def fake_verify(response, chunks, scores, **k):
        captured["ids"] = [c["chunk_id"] for c in chunks]
        m = Mock()
        m.response = "resp"
        m.get_warnings.return_value = []
        m.format_verification_block.return_value = ""
        m.score_result.final_score = 1.0
        m.score_result.confidence_level.value = "HIGH"
        return m

    monkeypatch.setattr(cli, "verify_and_score", fake_verify)
    monkeypatch.setattr(cli, "_print_feedback_hints", lambda *a, **k: None)

    cli.query(
        "q", hyde=False, rewrite=False, fast=False, top_k=5,
        no_feedback=True, profile=False, no_cache=True,
        cpu_embedding=True, cpu_reranker=True,
    )
    assert captured["ids"] == ["keep"]


# ---------------------------------------------------------------------------
# R2-01 — a failed recovery must block a new begin and keep the pre-image
# ---------------------------------------------------------------------------

def test_failed_recovery_blocks_new_begin_and_preserves_preimage(ds):
    vs = StrictFakeVS()
    vs._add("d", "old")
    ds.add([_chunk("old", "d")])
    handle = versioning.begin_replace("d", ds._conn, vs)
    ds.add([_chunk("new", "d")])

    vs.fail_restore = True
    with pytest.raises(VersioningError):
        versioning.rollback_replace(handle, ds._conn, vs)

    snap_before = ds._conn.execute(
        "SELECT COUNT(*) FROM version_snapshot_vectors WHERE doc_id='d'"
    ).fetchone()[0]
    assert snap_before == 1

    # A new begin must fail closed instead of overwriting the only pre-image.
    with pytest.raises(VersioningError):
        versioning.begin_replace("d", ds._conn, vs)
    assert ds._conn.execute(
        "SELECT COUNT(*) FROM version_snapshot_vectors WHERE doc_id='d'"
    ).fetchone()[0] == 1
    # SQL still exposes the previous version.
    assert {
        r[0] for r in ds._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id='d'")
    } == {"old"}


# ---------------------------------------------------------------------------
# R2-04 — recovery finishes the control rows and is generation-idempotent
# ---------------------------------------------------------------------------

def test_recovery_finishes_run_and_document_row(ds):
    ds.add([_chunk("old", "d")])
    ds._conn.execute(
        "INSERT OR REPLACE INTO document_versions(doc_id, active_run_id, generation, chunk_ids) "
        "VALUES ('d','run_old',1,?)", (json.dumps(["old"]),),
    )
    ds._conn.execute(
        "INSERT INTO ingest_runs(run_id, doc_id, started_at, status) "
        "VALUES ('run_new','d', datetime('now'), 'IN_PROGRESS')"
    )
    ds._conn.execute(
        "INSERT INTO ingest_documents(batch_id, run_id, doc_id, path, status) "
        "VALUES ('b1','run_new','d','/x/d.md','WRITING')"
    )
    ds._conn.execute(
        "INSERT OR REPLACE INTO version_snapshot_chunks(chunk_id, doc_id, text) VALUES ('old','d','old')"
    )
    ds._conn.execute(
        "INSERT INTO version_ops(doc_id, op, run_id, state, payload_json) "
        "VALUES ('d','REPLACE','run_new','WRITING',?)",
        (json.dumps({"old_run": "run_old", "old_ids": ["old"], "attempt_run": "run_new"}),),
    )
    ds._conn.commit()

    versioning.recover_pending_ops(ds._conn, None)
    assert ds._conn.execute(
        "SELECT status FROM ingest_runs WHERE run_id='run_new'"
    ).fetchone()[0] == "ROLLED_BACK"
    assert ds._conn.execute(
        "SELECT status FROM ingest_documents WHERE run_id='run_new'"
    ).fetchone()[0] == "ROLLED_BACK"


def test_replayed_commit_preserves_generation(ds):
    vs = StrictFakeVS()
    ds.add([_chunk("a", "d")])
    ds._conn.execute(
        "INSERT OR REPLACE INTO document_versions(doc_id, active_run_id, generation, chunk_ids) "
        "VALUES ('d','r0',1,?)", (json.dumps(["a"]),),
    )
    ds._conn.commit()
    payload = {
        "phase": "commit", "old_run": "r0", "old_ids": ["a"],
        "new_run": "r1", "new_ids": ["b"], "obsolete": [], "generation": 2,
    }
    op = ds._conn.execute(
        "INSERT INTO version_ops(doc_id, op, run_id, state, payload_json) "
        "VALUES ('d','COMMIT','r1','COMMITTING',?)", (json.dumps(payload),)
    ).lastrowid
    ds._conn.commit()

    versioning.recover_pending_ops(ds._conn, vs)
    g1 = ds._conn.execute("SELECT generation FROM document_versions WHERE doc_id='d'").fetchone()[0]
    # Replay the same operation (duplicate recovery).
    ds._conn.execute("UPDATE version_ops SET state='COMMITTING' WHERE id=?", (op,))
    ds._conn.commit()
    versioning.recover_pending_ops(ds._conn, vs)
    g2 = ds._conn.execute("SELECT generation FROM document_versions WHERE doc_id='d'").fetchone()[0]
    assert g1 == g2 == 2


# ---------------------------------------------------------------------------
# R2-02 — same content from a different path is the same document (allowed)
# ---------------------------------------------------------------------------

def test_same_content_other_path_is_allowed(batch_env):
    from rag_lab.cli_ingest import _run_batch_ingest
    ds, vs, tmp = batch_env["ds"], batch_env["vs"], batch_env["tmp"]
    (tmp / "a").mkdir(); (tmp / "b").mkdir()
    content = _front("Uno", "shared-id")
    p1 = tmp / "a" / "guia.md"; p1.write_text(content)
    p2 = tmp / "b" / "guia.md"; p2.write_text(content)  # identical content

    assert _run_batch_ingest(
        paths=[p1], doc_store=ds, vector_store=vs, force=False, device="cpu", strict=False,
    )["committed"] == 1
    # P0-C: identical content from another path is a NO-OP with no documental
    # write and no generation bump, even with force=True.
    gen_before = ds._conn.execute(
        "SELECT generation FROM document_versions WHERE doc_id='shared-id'"
    ).fetchone()[0]
    s2 = _run_batch_ingest(
        paths=[p2], doc_store=ds, vector_store=vs, force=True, device="cpu", strict=False,
    )
    assert s2["committed"] == 0
    assert s2["skipped"] == 1
    assert s2["failed"] == 0
    gen_after = ds._conn.execute(
        "SELECT generation FROM document_versions WHERE doc_id='shared-id'"
    ).fetchone()[0]
    assert gen_after == gen_before
