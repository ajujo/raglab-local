"""P0-B round-2 regressions (PB-R1..PB-R5).

Convert the round-1 adversarial findings into product regressions:
  PB-R1 mutated legacy source is not confirmed content.
  PB-R2 ownership evidence must be bound to the active version and coherent.
  PB-R3 controls are reconciled BEFORE the journal is marked DONE.
  PB-R4 administrative RESTORING reconciles its target run.
  PB-R5 every non-empty input (incl. whitespace) is bounded by bytes/tokenizer.
"""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from rag_lab.storage.docstore import DocStore
from rag_lab.storage.vector_store import VectorStore
from rag_lab.storage.metadata_store import MetadataStore
from rag_lab.ingest.transaction import (
    IngestTransaction, IngestRunStore, IngestBatchStore, IngestDocumentStore,
)
from rag_lab.ingest import versioning as v
from rag_lab.ingest.manifest import compute_md5
from rag_lab.exceptions import VersioningError, LLMContextOverflowError
import rag_lab.cli_ingest as cli


class Cut(BaseException):
    """Abrupt cut, not a business exception handled by the pipeline."""


def _body(tag):
    return ("---\ndoc_id: shared\ntitle: Audit\n---\n\n# Audit\n\n## Section\n\n"
            + ("alpha beta gamma delta epsilon zeta lambda " + tag + " ") * 90)


def _chunk(cid, text=None):
    return dict(chunk_id=cid, doc_id="shared", text=text or "reference " + cid,
                heading_path="H", tipo="texto", posicion_relativa=0.0,
                n_tokens=3, line_start=1, line_end=2)


def _add(d, vs, cid, text=None):
    c = _chunk(cid, text)
    d.add([c])
    vs.add(ids=[cid], embeddings=np.ones((1, 1024), dtype=np.float32),
           documents=[c["text"]], metadatas=[{"doc_id": "shared", "heading_path": "H"}])


def _ids(d, vs):
    return ({r[0] for r in d._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id='shared'")},
            set(vs.get_by_doc_id("shared")["ids"]))


@pytest.fixture
def stores(tmp_path, monkeypatch):
    d = DocStore(db_path=tmp_path / "db.sqlite"); d.initialize()
    vs = VectorStore(storage_path=tmp_path / "chroma"); vs.initialize()
    data = tmp_path / "data"; data.mkdir()
    monkeypatch.setattr("rag_lab.config.DATA_DIR", data)
    monkeypatch.setattr("rag_lab.ingest.cleaner.DATA_DIR", data)
    monkeypatch.setattr("rag_lab.ingest.manifest.DATA_DIR", data)
    monkeypatch.setattr("rag_lab.embedding.encoder.encode_chunks",
                        lambda rows, **kw: (np.ones((len(rows), 1024), dtype=np.float32),
                                            {c["chunk_id"]: {} for c in rows}))
    yield d, vs, tmp_path, data
    d.close()


def _seed(stores, metadata=False, active=False):
    d, vs, t, data = stores
    a = t / "a" / "source.md"; a.parent.mkdir(); a.write_text(_body("ORIGINAL"))
    b = t / "b" / "source.md"; b.parent.mkdir(); b.write_text(_body("FOREIGN"))
    _add(d, vs, "old", a.read_text())
    r = IngestRunStore(d._conn)
    r.create("owner", "shared", str(a)); r.update("owner", status="COMMITTED")
    if metadata:
        MetadataStore(conn=d._conn).upsert_document("shared", path=str(a), content_hash=compute_md5(a))
    if active:
        d._conn.execute("INSERT INTO document_versions(doc_id, active_run_id, generation, chunk_ids) "
                        "VALUES ('shared','owner',7,?)", (json.dumps(["old"]),))
        d._conn.commit()
    return d, vs, t, data, a, b


def _ingest(d, vs, p, route="batch"):
    if route == "batch":
        return cli._run_batch_ingest([p], d, vs, True, "cpu", False)["committed"]
    return cli._ingest_one(p, d, vs, True, "cpu", False)


# ---------------------------------------------------------------------------
# PB-R1 — mutated legacy source must not authorize foreign content
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["batch", "retry"])
@pytest.mark.parametrize("manifest", ["old_hash", "absent", "missing_hash"])
def test_mutated_legacy_source_is_not_confirmed_content(stores, route, manifest):
    d, vs, t, data, a, b = _seed(stores)
    oldhash = compute_md5(a)
    if manifest != "absent":
        e = {"doc_id": "shared", "path": str(a)}
        if manifest == "old_hash":
            e["hash"] = oldhash
        (data / "ingested.jsonl").write_text(json.dumps(e) + "\n")
    # The source at A changed after its confirmed ingest; B matches A now but not
    # the stored pre-image.
    a.write_bytes(b.read_bytes())
    assert _ingest(d, vs, b, route) == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_verified_identical_legacy_positive(stores, route):
    # P0-C policy: another path with an EXACT persisted representation is a
    # NO-OP without writes (no generation, no ownership change).  The old
    # single concatenated chunk is not an exact representation, so we seed the
    # real chunk set that the incoming source reproduces.
    from rag_lab import config
    from rag_lab.ingest.cleaner import clean_document
    from rag_lab.chunking.splitter import chunk_document

    d, vs, t, data, a, b = _seed(stores)
    b.write_bytes(a.read_bytes())
    (data / "ingested.jsonl").unlink(missing_ok=True)

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
    before_gen = d._conn.execute(
        "SELECT 1 FROM document_versions WHERE doc_id='shared'").fetchone()
    assert before_gen is None
    assert _ingest(d, vs, b, route) == 0
    assert _ids(d, vs) == (expected, expected)
    assert d._conn.execute(
        "SELECT 1 FROM document_versions WHERE doc_id='shared'").fetchone() is None


# ---------------------------------------------------------------------------
# PB-R2 — evidence must be bound to the active version / coherent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["FAILED", "IN_PROGRESS"])
@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("route", ["batch", "retry"])
def test_foreign_attempt_matrix(stores, status, metadata, active, route):
    d, vs, t, data, a, b = _seed(stores, metadata, active)
    r = IngestRunStore(d._conn)
    r.create("foreign", "shared", str(b)); r.update("foreign", status=status)
    assert _ingest(d, vs, b, route) == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_contradictory_metadata_cannot_override_active(stores, route):
    d, vs, t, data, a, b = _seed(stores, False, True)
    MetadataStore(conn=d._conn).upsert_document("shared", path=str(b), content_hash=compute_md5(b))
    assert _ingest(d, vs, b, route) == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_stale_committed_batch_hash_not_bound_to_active(stores, route):
    d, vs, t, data, a, b = _seed(stores)
    original = a.read_bytes()
    a.write_bytes(b.read_bytes())
    assert _ingest(d, vs, a) > 0  # same owner legitimately changes content
    active = d._conn.execute(
        "SELECT active_run_id FROM document_versions WHERE doc_id='shared'").fetchone()[0]
    assert v.admin_rollback(active, d._conn, vs)["status"] == "rolled_back"
    assert _ids(d, vs) == ({"old"}, {"old"})
    a.write_bytes(original)  # A matches the restored version again
    assert _ingest(d, vs, b, route) == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


# ---------------------------------------------------------------------------
# PB-R3 / PB-R4 — controls reconciled before DONE; admin RESTORING target
# ---------------------------------------------------------------------------

def _attempt(stores):
    d, vs, t, data, a, b = _seed(stores, True, True)
    tx = IngestTransaction("shared", a, d, vector_store=vs, chunk_ids=["new"]); tx.__enter__()
    batch = IngestBatchStore(d._conn).create()
    docs = IngestDocumentStore(d._conn)
    row = docs.create(batch, "shared", str(a)); docs.update(row, status="WRITING", run_id=tx.run_id)
    _add(d, vs, "new")
    return d, vs, t, data, a, b, tx, batch, row


def _control(d, tx, row):
    return (IngestRunStore(d._conn).get(tx.run_id)["status"],
            IngestDocumentStore(d._conn).get_by_id(row)["status"])


def test_commit_done_boundary_public_batch(stores, monkeypatch):
    d, vs, t, data = stores
    p = t / "source.md"; p.write_text(_body("NEW"))
    monkeypatch.setattr("rag_lab.ingest.manifest.create_manifest",
                        Mock(side_effect=Cut("power loss after commit")))
    with pytest.raises(Cut):
        cli._run_batch_ingest([p], d, vs, True, "cpu", False)
    row = d._conn.execute(
        "SELECT r.status,i.status,o.state FROM ingest_runs r "
        "JOIN ingest_documents i ON i.run_id=r.run_id "
        "JOIN version_ops o ON o.doc_id=r.doc_id").fetchone()
    # Controls must already be coherent even though recovery is a no-op here.
    assert row == ("COMMITTED", "COMMITTED", "DONE")


def test_rollback_done_boundary(stores, monkeypatch):
    d, vs, t, data, a, b, tx, batch, row = _attempt(stores)
    real = v.rollback_replace

    def cut(*args, **kw):
        real(*args, **kw)
        raise Cut("power loss after rollback journal DONE")

    with monkeypatch.context() as m:
        m.setattr(v, "rollback_replace", cut)
        with pytest.raises(Cut):
            tx.rollback()
    v.recover_pending_ops(d._conn, vs)
    assert _control(d, tx, row) == ("ROLLED_BACK", "ROLLED_BACK")


def test_admin_restoring_reconciles_target(stores, monkeypatch):
    d, vs, t, data, a, b, tx, batch, row = _attempt(stores)
    tx.__exit__(None, None, None)
    IngestDocumentStore(d._conn).update(row, status="COMMITTED")
    with monkeypatch.context() as m:
        m.setattr(vs, "restore_vectors", Mock(side_effect=RuntimeError("admin interrupted")))
        with pytest.raises(VersioningError):
            v.admin_rollback(tx.run_id, d._conn, vs)
    v.recover_pending_ops(d._conn, vs)
    assert _ids(d, vs) == ({"old"}, {"old"})
    assert _control(d, tx, row) == ("ROLLED_BACK", "ROLLED_BACK")


# ---------------------------------------------------------------------------
# PB-R5 — whitespace is not free
# ---------------------------------------------------------------------------

def test_whitespace_is_bounded(monkeypatch):
    from rag_lab.utils import tokenizer as tk
    monkeypatch.setattr(tk, "_load_backend_tokenizer", lambda: None)
    for ws in ["\u2003" * 10000, "\n\t" * 10000, "\r\n\t " * 10000]:
        assert tk.estimate_tokens(ws) >= len(ws.encode("utf-8"))
    from rag_lab.generation.prompt_builder import build_prompt_plan
    with pytest.raises(LLMContextOverflowError):
        build_prompt_plan("\u2003" * 10000, [])


# ---------------------------------------------------------------------------
# R2-F1 — confirmed hashes must agree; manifest cannot override documents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["batch", "retry"])
def test_mutated_active_hash_cannot_grant_foreign_write(stores, route):
    d, vs, t, data, a, b = _seed(stores, metadata=True, active=True)
    # Tamper only the active-version hash to match B; batch/documents keep A's hash.
    d._conn.execute("UPDATE document_versions SET content_hash=? WHERE doc_id='shared'",
                    (compute_md5(b),))
    d._conn.commit()
    assert _ingest(d, vs, b, route) == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_confirmed_document_hash_beats_edited_manifest(stores, route):
    d, vs, t, data, a, b = _seed(stores, metadata=True, active=True)
    # Active/batch hashes absent; documents holds A's hash. Edit file+manifest to B.
    a.write_bytes(b.read_bytes())
    (data / "ingested.jsonl").write_text(
        json.dumps({"doc_id": "shared", "path": str(a), "hash": compute_md5(b)}) + "\n")
    assert _ingest(d, vs, b, route) == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


# ---------------------------------------------------------------------------
# R2-F2 — edited manifest + edited source do not certify stored bytes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["batch", "retry"])
def test_edited_legacy_manifest_and_source_do_not_certify_stored_bytes(stores, route):
    d, vs, t, data, a, b = _seed(stores)
    a.write_bytes(b.read_bytes())
    (data / "ingested.jsonl").write_text(
        json.dumps({"doc_id": "shared", "path": str(a), "hash": compute_md5(b)}) + "\n")
    assert _ingest(d, vs, b, route) == 0
    assert _ids(d, vs) == ({"old"}, {"old"})


# ---------------------------------------------------------------------------
# R2-F3 — documents path must not contradict the legacy COMMITTED run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["batch", "retry"])
def test_legacy_documents_path_cannot_disagree_with_committed_run(stores, route):
    d, vs, t, data, a, b = _seed(stores)
    MetadataStore(conn=d._conn).upsert_document("shared", path=str(b), content_hash=compute_md5(b))
    assert _ingest(d, vs, b, route) == 0
    assert _ids(d, vs) == ({"old"}, {"old"})
