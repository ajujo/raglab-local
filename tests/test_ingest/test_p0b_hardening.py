"""P0-B hardening regressions for the five final blockers (R3-01..R3-05).

Covers:
  * P0B-01 confirmed-owner semantics (a FAILED foreign run cannot grant ownership).
  * P0B-02 attempt_run survives RESTORING; recovery reconciles run/batch rows.
  * P0B-03 ``ingest --resume`` recovers before selecting pending documents.
  * P0B-04 identical legacy content is allowed; distinct content is refused.
  * P0B-05 the prompt budget is a guaranteed bound (byte cap / backend tokenizer).
"""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from rag_lab import config
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.vector_store import VectorStore
from rag_lab.ingest import versioning as v
from rag_lab.ingest.transaction import (
    IngestTransaction, IngestRunStore, IngestDocumentStore, IngestBatchStore,
)
from rag_lab.ingest.manifest import compute_md5
from rag_lab.exceptions import VersioningError, LLMContextOverflowError
from rag_lab.cli_ingest import _run_batch_ingest, _ingest_one


def _chunk(cid, doc_id="shared"):
    return dict(chunk_id=cid, doc_id=doc_id, text="reference " + cid,
                heading_path="H", tipo="texto", posicion_relativa=0.0,
                n_tokens=3, line_start=1, line_end=2)


def _body(tag="A"):
    return ("---\ndoc_id: shared\ntitle: Reference\n---\n\n# Reference " + tag +
            "\n\n## Section\n\n" +
            ("alpha beta gamma delta epsilon zeta theta kappa lambda " + tag + " ") * 90)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    d = DocStore(db_path=tmp_path / "db.sqlite"); d.initialize()
    vs = VectorStore(storage_path=tmp_path / "chroma"); vs.initialize()
    data = tmp_path / "data"; data.mkdir()
    monkeypatch.setattr("rag_lab.config.DATA_DIR", data)
    monkeypatch.setattr("rag_lab.ingest.cleaner.DATA_DIR", data)
    monkeypatch.setattr("rag_lab.ingest.manifest.DATA_DIR", data)

    def encode(rows, **kw):
        return np.ones((len(rows), 1024), dtype=np.float32), {c["chunk_id"]: {} for c in rows}

    monkeypatch.setattr("rag_lab.embedding.encoder.encode_chunks", encode)
    yield d, vs, tmp_path
    d.close()


def _add(d, vs, cid):
    d.add([_chunk(cid)])
    vs.add(ids=[cid], embeddings=np.ones((1, 1024), dtype=np.float32),
           documents=["reference " + cid], metadatas=[{"doc_id": "shared", "heading_path": "H"}])


def _ids(d, vs):
    return ({r[0] for r in d._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id='shared'")},
            set(vs.get_by_doc_id("shared")["ids"]))


def _seed(d, vs, t):
    _add(d, vs, "old")
    r = IngestRunStore(d._conn)
    r.create("owner", "shared", str(t / "a" / "source.md"))
    r.update("owner", status="COMMITTED")
    return d, vs, t


def _run(d, vs, p, force=True, resume=None):
    return _run_batch_ingest(paths=[p], doc_store=d, vector_store=vs, force=force,
                             device="cpu", strict=False, resume_batch_id=resume)


# ---------------------------------------------------------------------------
# P0B-01 — a FAILED foreign run must not grant ownership
# ---------------------------------------------------------------------------

def test_failed_foreign_run_cannot_grant_ownership(stores):
    d, vs, t = _seed(*stores)
    p = t / "b" / "source.md"; p.parent.mkdir(); p.write_text(_body("FOREIGN"))
    r = IngestRunStore(d._conn)
    r.create("foreign_failed", "shared", str(p)); r.update("foreign_failed", status="FAILED")
    summary = _run(d, vs, p)
    assert summary["committed"] == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


def test_failed_foreign_run_cannot_repoint_active_owner(stores):
    d, vs, t = _seed(*stores)
    # Active version exists and belongs to 'owner' (a/source.md).
    d._conn.execute("INSERT OR REPLACE INTO document_versions(doc_id, active_run_id, generation, chunk_ids) "
                    "VALUES ('shared','owner',1,?)", (json.dumps(["old"]),))
    d._conn.commit()
    p = t / "b" / "source.md"; p.parent.mkdir(); p.write_text(_body("FOREIGN"))
    r = IngestRunStore(d._conn)
    r.create("foreign_failed", "shared", str(p)); r.update("foreign_failed", status="FAILED")
    assert _run(d, vs, p)["committed"] == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


# ---------------------------------------------------------------------------
# P0B-04 — legacy content identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["batch", "retry"])
def test_identical_legacy_content_allowed(stores, route):
    # P0-C: identical legacy content from another path is a NO-OP without
    # writes when the exact persisted representation is demonstrated.  The old
    # concatenated single chunk is not such a representation, so seed the real
    # chunk set and require NO-OP (not a new generation).
    from rag_lab.ingest.cleaner import clean_document
    from rag_lab.chunking.splitter import chunk_document

    d, vs, t = _seed(*stores)
    p = t / "b" / "source.md"; p.parent.mkdir(); p.write_text(_body())
    a = t / "a" / "source.md"; a.parent.mkdir(); a.write_bytes(p.read_bytes())
    (config.DATA_DIR / "ingested.jsonl").unlink(missing_ok=True)

    d._conn.execute("DELETE FROM chunks WHERE doc_id='shared'")
    d._conn.execute("DELETE FROM chunks_fts WHERE doc_id='shared'")
    d._conn.commit()
    vs.delete_by_doc_id("shared")
    cleaned = clean_document(a).read_text(encoding="utf-8")
    chunks = chunk_document(cleaned, doc_id="shared",
                            max_tokens=config.CHUNK_MAX_TOKENS,
                            overlap=config.CHUNK_OVERLAP)
    dicts = [c.to_dict() for c in chunks]
    for c in dicts:
        c["embedding_model_name"] = config.EMBEDDING_MODEL
        c["embedding_model_version"] = config.EMBEDDING_MODEL_VERSION
        c["embedding_dim"] = config.EMBEDDING_DIM
        c["sparse_format_version"] = config.SPARSE_FORMAT_VERSION
    d.add(dicts)
    vs.add(ids=[c["chunk_id"] for c in dicts],
           embeddings=np.ones((len(dicts), 1024), dtype=np.float32),
           documents=[c["text"] for c in dicts],
           metadatas=[{"doc_id": "shared", "heading_path": c.get("heading_path", "")}
                      for c in dicts])
    expected = {c["chunk_id"] for c in dicts}
    if route == "batch":
        res = _run(d, vs, p)
        assert res["committed"] == 0 and res["skipped"] == 1, res
    else:
        assert _ingest_one(p, d, vs, False, "cpu", False) == 0
    assert _ids(d, vs) == (expected, expected)
    assert d._conn.execute(
        "SELECT 1 FROM document_versions WHERE doc_id='shared'").fetchone() is None


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_distinct_legacy_content_refused(stores, route):
    d, vs, t = _seed(*stores)
    p = t / "b" / "source.md"; p.parent.mkdir(); p.write_text(_body("DIFFERENT"))
    (config.DATA_DIR / "ingested.jsonl").write_text(
        json.dumps(dict(doc_id="shared", path=str(t / "a" / "source.md"), hash="oldhash")) + "\n")
    if route == "batch":
        assert _run(d, vs, p)["failed"] == 1
    else:
        assert _ingest_one(p, d, vs, True, "cpu", False) == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


# ---------------------------------------------------------------------------
# P0B-02 — RESTORING keeps attempt_run; recovery reconciles control rows
# ---------------------------------------------------------------------------

def _attempt(d, vs, t):
    txn = IngestTransaction("shared", t / "a" / "source.md", d, vector_store=vs, chunk_ids=["new"])
    txn.__enter__()
    b = IngestBatchStore(d._conn).create()
    docs = IngestDocumentStore(d._conn)
    row = docs.create(b, "shared", str(t / "a" / "source.md"))
    docs.update(row, status="WRITING", run_id=txn.run_id)
    _add(d, vs, "new")
    return txn, b, row


@pytest.mark.parametrize("state", ["WRITING", "RESTORING"])
def test_recovery_reconciles_control_rows(stores, monkeypatch, state):
    d, vs, t = _seed(*stores)
    txn, b, row = _attempt(d, vs, t)
    if state == "RESTORING":
        with monkeypatch.context() as m:
            m.setattr(vs, "restore_vectors", Mock(side_effect=RuntimeError("interrupted")))
            with pytest.raises(VersioningError):
                txn.rollback()
    v.recover_pending_ops(d._conn, vs)
    rs = IngestRunStore(d._conn).get(txn.run_id)["status"]
    ds = IngestDocumentStore(d._conn).get_by_id(row)["status"]
    resumable = IngestDocumentStore(d._conn).list_resumable(b)
    assert _ids(d, vs) == ({"old"}, {"old"})
    assert rs == "ROLLED_BACK" and ds == "ROLLED_BACK" and len(resumable) == 1


def test_attempt_run_survives_rollback_payload(stores, monkeypatch):
    d, vs, t = _seed(*stores)
    txn, b, row = _attempt(d, vs, t)
    with monkeypatch.context() as m:
        m.setattr(vs, "restore_vectors", Mock(side_effect=RuntimeError("interrupted")))
        with pytest.raises(VersioningError):
            txn.rollback()
    payload = json.loads(d._conn.execute(
        "SELECT payload_json FROM version_ops WHERE doc_id='shared' AND state='RESTORING'"
    ).fetchone()[0])
    assert payload.get("attempt_run") == txn.run_id


# ---------------------------------------------------------------------------
# P0B-03 — CLI resume recovers before selecting
# ---------------------------------------------------------------------------

def test_cli_resume_recovers_before_selection(stores, monkeypatch):
    import rag_lab.cli_ingest as cli
    from typer import Typer
    from typer.testing import CliRunner

    d, vs, t = _seed(*stores)
    txn, b, row = _attempt(d, vs, t)
    monkeypatch.setattr("rag_lab.storage.docstore.DocStore", lambda: d)
    monkeypatch.setattr("rag_lab.storage.vector_store.VectorStore", lambda: vs)
    monkeypatch.setattr(d, "initialize", lambda: None)
    monkeypatch.setattr(d, "close", lambda: None)
    calls = []
    monkeypatch.setattr(cli, "_run_batch_ingest",
                        lambda **kw: calls.append(kw) or dict(batch_id=b, committed=0, skipped=0,
                                                              failed=0, total_chunks=0, elapsed_s=0))
    app = Typer(); app.add_typer(cli.ingest_app, name="ingest")
    out = CliRunner().invoke(app, ["ingest", "--resume", "--cpu-embedding"])
    assert out.exit_code == 0
    assert calls, "--resume must recover the WRITING journal before choosing resumable rows"


# ---------------------------------------------------------------------------
# P0B-05 — the budget is a guaranteed bound
# ---------------------------------------------------------------------------

def test_new_mixture_is_rejected_locally_without_tokenizer(monkeypatch):
    from rag_lab.generation import llm_client as lc
    from rag_lab.generation.prompt_builder import build_prompt_plan
    from rag_lab.utils import tokenizer as tk

    # Simulate no configured backend tokenizer.
    monkeypatch.setattr(tk, "_load_backend_tokenizer", lambda: None)
    q = "Describe este contenido: " + "\U0001F9EC" * 1300 + " a" * 5200
    with pytest.raises(LLMContextOverflowError):
        build_prompt_plan(q, [])
    backend = Mock()
    monkeypatch.setattr(lc, "_get_client", backend)
    with pytest.raises(LLMContextOverflowError):
        lc.generate_response("system", q, prompt_tokens=1)
    assert backend.call_count == 0


def test_byte_estimate_is_an_upper_bound(monkeypatch):
    from rag_lab.utils import tokenizer as tk
    monkeypatch.setattr(tk, "_load_backend_tokenizer", lambda: None)
    for text in ["hello world", "\U0001F9EC" * 100, "a" * 5000,
                 "Describe: " + "\U0001F9EC" * 300]:
        assert tk.estimate_tokens(text) >= len(text.encode("utf-8"))
