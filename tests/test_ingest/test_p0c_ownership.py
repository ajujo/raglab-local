"""P0-C ownership contract regressions (integration + pure core).

Uses real temporary DocStore/VectorStore (SQLite + FTS5 + Chroma) and a
deterministic fake encoder, like the P0-B round-3 probes.  These tests encode
the *approved* policy: an exact equivalence from another path is a NO-OP with
no documental writes and no ownership/generation change.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rag_lab.cli_ingest as ci
from rag_lab.ingest import ownership
from rag_lab.ingest.manifest import compute_md5
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.vector_store import VectorStore


def body(label="ORIGINAL", identity="audit-three", tags="review"):
    intro = f"---\ndoc_id: {identity}\ntitle: Independent audit\ntags: [{tags}]\n---\n\n"
    return intro + "\n\n".join(
        f"# Chapter {s}\n\n" + "\n".join(
            f"Paragraph {s}-{i}: {label} " + "alpha beta gamma delta lambda epsilon theta " * 16 + "."
            for i in range(15)
        )
        for s in range(3)
    ) + "\n"


def small_body(label="ORIGINAL", identity="audit-small", tags="review"):
    intro = f"---\ndoc_id: {identity}\ntitle: Small audit\ntags: [{tags}]\n---\n\n"
    return intro + "# Chapter 1\n\n" + f"Paragraph: {label} " + "alpha beta gamma " * 10 + "\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    d = DocStore(tmp_path / "audit.sqlite")
    d.initialize()
    vs = VectorStore(storage_path=tmp_path / "vectors")
    vs.initialize()
    data = tmp_path / "data"
    data.mkdir()
    for module in ["rag_lab.config", "rag_lab.ingest.cleaner", "rag_lab.ingest.manifest"]:
        monkeypatch.setattr(module + ".DATA_DIR", data)
    monkeypatch.setattr(
        "rag_lab.embedding.encoder.encode_chunks",
        lambda rows, **kw: (np.ones((len(rows), 1024), dtype=np.float32),
                            {c["chunk_id"]: {} for c in rows}),
    )
    a = tmp_path / "a" / "audit-three.md"
    a.parent.mkdir()
    a.write_text(body())
    b = tmp_path / "b" / "audit-three.md"
    b.parent.mkdir()
    b.write_text(body("ALIEN"))
    yield SimpleNamespace(d=d, vs=vs, data=data, t=tmp_path, a=a, b=b)
    d.close()


def ingest(e, p, route="batch", force=False):
    if route == "retry":
        return ci._ingest_one(p, e.d, e.vs, force, "cpu", False)
    return ci._run_batch_ingest([p], e.d, e.vs, force, "cpu", False)


def rows(conn, sql, params=()):
    return sorted(tuple(r) for r in conn.execute(sql, params).fetchall())


def doc_state(e, doc_id="audit-three"):
    """Hashable snapshot of DOCUMENTAL state (control tables excluded)."""
    c = e.d._conn
    vec = e.vs.get_by_doc_id(doc_id)
    manifest = (e.data / "ingested.jsonl")
    return {
        "chunks": rows(c, "SELECT chunk_id,text FROM chunks WHERE doc_id=?", (doc_id,)),
        "fts": rows(c, "SELECT chunk_id,text FROM chunks_fts WHERE doc_id=?", (doc_id,)),
        "versions": rows(c, "SELECT active_run_id,generation,chunk_ids,content_hash "
                            "FROM document_versions WHERE doc_id=?", (doc_id,)),
        "documents": rows(c, "SELECT doc_id,path,content_hash,title,domain,source_type,"
                             "language,version FROM documents WHERE doc_id=?", (doc_id,)),
        "tags": rows(c, "SELECT t.name FROM document_tags dt JOIN tags t "
                        "ON dt.tag_id=t.tag_id WHERE dt.doc_id=?", (doc_id,)),
        "snap_chunks": rows(c, "SELECT chunk_id,text FROM version_snapshot_chunks "
                               "WHERE doc_id=?", (doc_id,)),
        "snap_docs": rows(c, "SELECT run_id,doc_json,tags_json,content_hash "
                             "FROM version_snapshot_docs WHERE doc_id=?", (doc_id,)),
        "ops": rows(c, "SELECT id,op,run_id,state,payload_json FROM version_ops "
                       "WHERE doc_id=?", (doc_id,)),
        "vec_ids": tuple(sorted(str(i) for i in vec["ids"])),
        "vec_docs": tuple(sorted((str(i), t) for i, t in zip(vec["ids"], vec["documents"]))),
        "manifest": manifest.read_text() if manifest.exists() else None,
    }


def generation(e, doc_id="audit-three"):
    row = e.d._conn.execute(
        "SELECT generation FROM document_versions WHERE doc_id=?", (doc_id,)
    ).fetchone()
    return row[0] if row else None


def seed(e, p=None):
    res = ingest(e, p or e.a)
    assert res["committed"] == 1, res
    return doc_state(e)


# ---------------------------------------------------------------------------
# Core contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["batch", "retry"])
def test_create_new_document(env, route):
    e = env
    res = ingest(e, e.a, route)
    committed = res["committed"] if route == "batch" else (1 if res > 0 else 0)
    assert committed == 1
    assert generation(e) == 1


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_noop_owner_same_result(env, route):
    e = env
    seed(e)
    before = doc_state(e)
    gen = generation(e)
    res = ingest(e, e.a, route, force=False)
    if route == "batch":
        assert res["committed"] == 0 and res["skipped"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before
    assert generation(e) == gen


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_replace_owner_changed_content(env, route):
    e = env
    seed(e)
    gen = generation(e)
    e.a.write_text(body("UPDATED"))
    res = ingest(e, e.a, route, force=False)
    committed = res["committed"] if route == "batch" else (1 if res > 0 else 0)
    assert committed == 1
    assert generation(e) == gen + 1


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_noop_equivalent_other_source_has_no_documental_writes(env, route):
    e = env
    seed(e)
    before = doc_state(e)
    gen = generation(e)
    # e.b has the ORIGINAL bytes (same content), but a different path.
    e.b.write_bytes(e.a.read_bytes())
    res = ingest(e, e.b, route, force=False)
    if route == "batch":
        assert res["committed"] == 0 and res["skipped"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before
    assert generation(e) == gen


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_deny_other_source_different_content(env, route):
    e = env
    seed(e)
    before = doc_state(e)
    res = ingest(e, e.b, route, force=False)  # ALIEN
    if route == "batch":
        assert res["committed"] == 0 and res["failed"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_force_owner_replaces(env, route):
    e = env
    seed(e)
    gen = generation(e)
    res = ingest(e, e.a, route, force=True)
    committed = res["committed"] if route == "batch" else (1 if res > 0 else 0)
    assert committed == 1
    assert generation(e) == gen + 1


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_force_other_source_still_noop(env, route):
    e = env
    seed(e)
    before = doc_state(e)
    gen = generation(e)
    e.b.write_bytes(e.a.read_bytes())
    res = ingest(e, e.b, route, force=True)
    if route == "batch":
        assert res["committed"] == 0 and res["skipped"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before
    assert generation(e) == gen


# ---------------------------------------------------------------------------
# R3-F1..F4 / adversarial states
# ---------------------------------------------------------------------------

def test_contradictory_committed_hashes_denied(env):
    e = env
    seed(e)
    before = doc_state(e)
    c = e.d._conn
    run = c.execute("SELECT active_run_id FROM document_versions").fetchone()[0]
    c.execute("UPDATE ingest_documents SET content_hash=? WHERE run_id=?",
              (compute_md5(e.b), run))
    from rag_lab.ingest.transaction import IngestBatchStore, IngestDocumentStore
    batch = IngestBatchStore(c).create()
    row = IngestDocumentStore(c).create(batch, "audit-three", str(e.a), compute_md5(e.a))
    IngestDocumentStore(c).update(row, status="COMMITTED", run_id=run)
    c.commit()
    res = ingest(e, e.b)
    assert res["committed"] == 0 and res["failed"] == 1, res
    assert doc_state(e) == before


def test_mutated_active_hash_denied(env):
    e = env
    seed(e)
    before = doc_state(e)
    e.d._conn.execute("UPDATE document_versions SET content_hash=?",
                      (compute_md5(e.b),))
    e.d._conn.commit()
    res = ingest(e, e.b)
    assert res["committed"] == 0 and res["failed"] == 1, res
    get = e.d._conn.execute("SELECT chunk_id,text FROM chunks WHERE doc_id='audit-three'").fetchall()
    assert len(get) > 0


def test_concat_is_not_identity(env):
    e = env
    seed(e)
    before = doc_state(e)
    texts = [r[0] for r in e.d._conn.execute(
        "SELECT text FROM chunks WHERE doc_id='audit-three' ORDER BY rowid")]
    e.b.write_text("".join(texts))
    res = ingest(e, e.b)
    assert res["committed"] == 0 and res["failed"] == 1, res
    assert doc_state(e) == before


def test_legacy_exact_no_manifest_is_noop(env):
    e = env
    seed(e)
    c = e.d._conn
    c.execute("DELETE FROM document_versions")
    c.execute("DELETE FROM ingest_documents")
    c.execute("DELETE FROM documents")
    c.commit()
    e.b.write_bytes(e.a.read_bytes())
    (e.data / "ingested.jsonl").unlink(missing_ok=True)
    before = doc_state(e)
    res = ingest(e, e.b)
    assert res["committed"] == 0 and res["skipped"] == 1, res
    assert doc_state(e) == before
    assert e.d._conn.execute("SELECT 1 FROM document_versions").fetchone() is None


def test_legacy_incomplete_evidence_denied(env):
    e = env
    seed(e)
    c = e.d._conn
    c.execute("DELETE FROM document_versions")
    c.execute("DELETE FROM ingest_documents")
    c.execute("DELETE FROM documents")
    c.execute("DELETE FROM ingest_runs")
    c.commit()
    before = doc_state(e)
    res = ingest(e, e.b)
    assert res["committed"] == 0 and res["failed"] == 1, res
    assert doc_state(e) == before


def test_sql_chroma_disagreement_denied(env):
    e = env
    seed(e)
    before = doc_state(e)
    ids = e.d._conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id='audit-three'").fetchall()
    e.vs.delete_ids([ids[0][0]])
    e.b.write_bytes(e.a.read_bytes())
    res = ingest(e, e.b)
    assert res["committed"] == 0 and res["failed"] == 1, res
    assert doc_state(e)["chunks"] == before["chunks"]


def test_pending_operation_denied_by_core(env):
    # The batch pipeline recovers before authorizing (P0B-03).  This test
    # exercises the core defense for callers that bypass recovery.
    e = env
    seed(e)
    e.d._conn.execute(
        "INSERT INTO version_ops (doc_id, op, run_id, state, created_at, payload_json) "
        "VALUES ('audit-three','REPLACE',NULL,'WRITING',datetime('now'),'{}')")
    e.d._conn.commit()
    e.b.write_bytes(e.a.read_bytes())
    decision = ownership.evaluate(
        e.d._conn, e.vs,
        doc_id="audit-three", origin_path=e.b, content_hash=compute_md5(e.b),
        chunk_dicts=[dict(chunk_id=r[0], text=r[1]) for r in e.d._conn.execute(
            "SELECT chunk_id,text FROM chunks WHERE doc_id='audit-three'")],
        classification=ownership.classification_from_fields(
            "t", "d", "s", "l", "v", []),
        force=False,
    )
    assert decision.kind is ownership.DecisionKind.DENY_INCOHERENT


def test_failed_foreign_run_never_grants_ownership(env):
    e = env
    seed(e)
    before = doc_state(e)
    c = e.d._conn
    c.execute("INSERT INTO ingest_runs (run_id,doc_id,source_path,started_at,status) "
              "VALUES ('failed-run','audit-three',?,datetime('now'),'FAILED')", (str(e.b),))
    c.commit()
    res = ingest(e, e.b)
    assert res["committed"] == 0 and res["failed"] == 1, res
    assert doc_state(e) == before


def test_new_id_with_stale_chunks_without_evidence_denied(env):
    e = env
    e.d.add([dict(chunk_id="orphan-1", doc_id="orphan", text="stale",
                  heading_path="H", tipo="texto", posicion_relativa=0.0,
                  n_tokens=1, line_start=1, line_end=2)])
    e.d._conn.commit()
    orphan = e.t / "orphan" / "orphan.md"
    orphan.parent.mkdir()
    orphan.write_text(body("ORIGINAL", identity="orphan"))
    res = ingest(e, orphan)
    assert res["committed"] == 0 and res["failed"] == 1, res


# ---------------------------------------------------------------------------
# Pipeline dimensions
# ---------------------------------------------------------------------------

def test_classification_change_same_chunks_is_replace(env):
    e = env
    seed(e)
    gen = generation(e)
    e.a.write_text(body(tags="updated"))
    res = ingest(e, e.a)
    assert res["committed"] == 1 and res["skipped"] == 0, res
    assert generation(e) == gen + 1
    tags = [r[0] for r in e.d._conn.execute(
        "SELECT t.name FROM document_tags dt JOIN tags t ON dt.tag_id=t.tag_id "
        "WHERE dt.doc_id='audit-three'")]
    assert "updated" in tags


def test_embedding_change_same_chunks_is_replace(env, monkeypatch):
    e = env
    seed(e)
    gen = generation(e)
    monkeypatch.setattr("rag_lab.config.EMBEDDING_MODEL_VERSION", "different-version")
    res = ingest(e, e.a)
    assert res["committed"] == 1 and res["skipped"] == 0, res
    assert generation(e) == gen + 1


def test_params_change_without_effect_is_noop(env, monkeypatch):
    e = env
    small = e.t / "small" / "audit-small.md"
    small.parent.mkdir()
    small.write_text(small_body())
    res = ingest(e, small)
    assert res["committed"] == 1, res
    before = doc_state(e, "audit-small")
    gen = generation(e, "audit-small")
    monkeypatch.setattr("rag_lab.config.CHUNK_MAX_TOKENS", 100000)
    res = ingest(e, small)
    assert res["committed"] == 0 and res["skipped"] == 1, res
    assert doc_state(e, "audit-small") == before
    assert generation(e, "audit-small") == gen


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

def _chunk(cid, text):
    return ownership.ChunkRef(cid, ownership.text_sha256(text))


def _state(**kw):
    base = dict(
        doc_id="d", has_version=True, active_run_id="r", generation=2,
        active_content_hash="h", owner_path="/a.md", owner_run_status="COMMITTED",
        documents_path="/a.md", documents_content_hash="h", committed_hashes=("h",),
        chunks=(_chunk("c1", "one"), _chunk("c2", "two")),
        chunk_ids_version=("c1", "c2"), chroma_verifiable=True, chroma_matches=True,
        fts_matches=True, embedding=("m", "v", 3, 1), embedding_incoherent=False,
        classification=("t", "d", "s", "l", "v", ("tag",)), has_documents=True,
        pending_ops=False, hash_column_available=True,
    )
    base.update(kw)
    return ownership.PersistedState(**base)


def _incoming(**kw):
    base = dict(
        doc_id="d", origin_path="/a.md", content_hash="h",
        chunks=(_chunk("c1", "one"), _chunk("c2", "two")),
        embedding=("m", "v", 3, 1),
        classification=("t", "d", "s", "l", "v", ("tag",)),
    )
    base.update(kw)
    return ownership.IncomingState(**base)


def test_pure_decision_matrix():
    s = _state()
    assert ownership.decide(s, _incoming()).kind is ownership.DecisionKind.NOOP_OWNER
    assert ownership.decide(s, _incoming(origin_path="/b.md")).kind is \
        ownership.DecisionKind.NOOP_EQUIVALENT_OTHER_SOURCE
    assert ownership.decide(s, _incoming(origin_path="/b.md", content_hash="x",
                                         chunks=(_chunk("c1", "one"),))).kind is \
        ownership.DecisionKind.DENY_NOT_OWNER
    assert ownership.decide(s, _incoming(classification=("t2", "d", "s", "l", "v", ("tag",)))).kind \
        is ownership.DecisionKind.REPLACE
    assert ownership.decide(_state(pending_ops=True), _incoming()).kind is \
        ownership.DecisionKind.DENY_INCOHERENT
    assert ownership.decide(_state(committed_hashes=("h", "other")), _incoming()).kind is \
        ownership.DecisionKind.DENY_INCOHERENT
    assert ownership.decide(_state(chroma_matches=False), _incoming()).kind is \
        ownership.DecisionKind.DENY_INCOHERENT


def test_pure_force_owner_but_not_other_source():
    s = _state()
    assert ownership.decide(s, _incoming(), force=True).kind is ownership.DecisionKind.REPLACE
    assert ownership.decide(s, _incoming(origin_path="/b.md"), force=True).kind is \
        ownership.DecisionKind.NOOP_EQUIVALENT_OTHER_SOURCE


# ---------------------------------------------------------------------------
# Additional P0-C contract regressions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["batch", "retry"])
def test_other_source_classification_change_denied(env, route):
    e = env
    seed(e)
    before = doc_state(e)
    e.b.write_text(body(tags="different"))  # same chunks, different classification
    res = ingest(e, e.b, route)
    if route == "batch":
        assert res["committed"] == 0 and res["failed"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_other_source_embedding_change_denied(env, route, monkeypatch):
    e = env
    seed(e)
    before = doc_state(e)
    monkeypatch.setattr("rag_lab.config.EMBEDDING_MODEL_VERSION", "different-version")
    e.b.write_bytes(e.a.read_bytes())
    res = ingest(e, e.b, route)
    if route == "batch":
        assert res["committed"] == 0 and res["failed"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_legacy_exact_no_manifest_is_noop_both_routes(env, route):
    e = env
    seed(e)
    c = e.d._conn
    c.execute("DELETE FROM document_versions")
    c.execute("DELETE FROM ingest_documents")
    c.execute("DELETE FROM documents")
    c.commit()
    (e.data / "ingested.jsonl").unlink(missing_ok=True)
    e.b.write_bytes(e.a.read_bytes())
    before = doc_state(e)
    res = ingest(e, e.b, route)
    if route == "batch":
        assert res["committed"] == 0 and res["skipped"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before


def test_revalidation_blocks_changed_state(env, monkeypatch):
    e = env
    seed(e)  # initial commit
    e.a.write_text(body("UPDATED"))  # owner replacement candidate
    before = doc_state(e)

    from rag_lab.ingest import ownership as real_ownership
    real_evaluate = real_ownership.evaluate
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_evaluate(*args, **kwargs)
        # Simulate a state change observed at publish time.
        return real_ownership.Decision(
            real_ownership.DecisionKind.DENY_INCOHERENT,
            "state changed between authorize and publish",
        )

    monkeypatch.setattr("rag_lab.ingest.ownership.evaluate", flaky)
    res = ingest(e, e.a)
    assert res["committed"] == 0 and res["failed"] == 1, res
    assert doc_state(e) == before


def test_rollback_after_authorized_replace_restores_noop_state(env):
    e = env
    seed(e)
    original = e.a.read_bytes()
    e.a.write_text(body("UPDATED"))
    assert ingest(e, e.a)["committed"] == 1
    active = e.d._conn.execute(
        "SELECT active_run_id FROM document_versions WHERE doc_id='audit-three'"
    ).fetchone()[0]
    from rag_lab.ingest import versioning
    result = versioning.admin_rollback(active, e.d._conn, e.vs)
    assert result["status"] == "rolled_back"
    # Restore the source bytes to the rolled-back version, then the restored
    # version must authorize it as a NO-OP.
    e.a.write_bytes(original)
    before = doc_state(e)
    res = ingest(e, e.a, force=False)
    assert res["committed"] == 0 and res["skipped"] == 1, res
    assert doc_state(e) == before


def test_legacy_contradictory_committed_hashes_denied(env):
    e = env
    seed(e)
    c = e.d._conn
    run = c.execute("SELECT active_run_id FROM document_versions").fetchone()[0]
    from rag_lab.ingest.transaction import IngestBatchStore, IngestDocumentStore
    batch = IngestBatchStore(c).create()
    row = IngestDocumentStore(c).create(batch, "audit-three", str(e.b), compute_md5(e.b))
    IngestDocumentStore(c).update(row, status="COMMITTED", run_id=run)
    c.execute("DELETE FROM document_versions")
    c.execute("DELETE FROM documents")
    c.commit()
    before = doc_state(e)
    res = ingest(e, e.b)
    assert res["committed"] == 0 and res["failed"] == 1, res
    assert doc_state(e) == before


def test_legacy_failed_foreign_attempt_never_owner(env):
    e = env
    seed(e)
    c = e.d._conn
    original = e.a.read_bytes()
    e.a.write_bytes(e.b.read_bytes())  # owner changes content
    assert ingest(e, e.a)["committed"] == 1
    active = c.execute("SELECT active_run_id FROM document_versions").fetchone()[0]
    from rag_lab.ingest import versioning
    assert versioning.admin_rollback(active, c, e.vs)["status"] == "rolled_back"
    e.a.write_bytes(original)  # A matches the restored version again
    before = doc_state(e)
    # A foreign FAILED run must not grant ownership of B.
    c.execute("INSERT INTO ingest_runs (run_id,doc_id,source_path,started_at,status) "
              "VALUES ('foreign-failed','audit-three',?,datetime('now'),'FAILED')", (str(e.b),))
    c.commit()
    res = ingest(e, e.b)
    assert res["committed"] == 0 and res["failed"] == 1, res
    assert doc_state(e) == before


def test_incoherent_fts_text_denied(env):
    """P0C-SOL61-R1-F1: same chunk ids but stale FTS text must DENY."""
    e = env
    seed(e)
    row = e.d._conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id='audit-three' ORDER BY chunk_id LIMIT 1"
    ).fetchone()
    e.d._conn.execute(
        "UPDATE chunks_fts SET text='FTS_STALE_SENTINEL' WHERE chunk_id=? AND doc_id='audit-three'",
        (row[0],),
    )
    e.d._conn.commit()
    e.a.write_text(body("UPDATED"))
    before = doc_state(e)
    res = ingest(e, e.a)
    assert res["committed"] == 0 and res["failed"] == 1, res
    assert doc_state(e) == before


# ---------------------------------------------------------------------------
# R2-F1 — persisted-hash contradiction with no batch rows
# ---------------------------------------------------------------------------

def _seed_retry_contradiction(e):
    """Seed through ``retry`` (no ``ingest_documents`` rows) and corrupt
    only ``documents.content_hash`` so the two persisted hashes disagree.
    """
    assert ingest(e, e.a, "retry") > 0
    c = e.d._conn
    assert c.execute(
        "SELECT COUNT(*) FROM ingest_documents WHERE doc_id='audit-three'"
    ).fetchone()[0] == 0
    c.execute("UPDATE documents SET content_hash=? WHERE doc_id='audit-three'",
              (compute_md5(e.b),))
    c.commit()
    active = c.execute(
        "SELECT content_hash FROM document_versions WHERE doc_id='audit-three'"
    ).fetchone()[0]
    docs = c.execute(
        "SELECT content_hash FROM documents WHERE doc_id='audit-three'"
    ).fetchone()[0]
    assert active and docs and active != docs
    assert generation(e) == 1


@pytest.mark.parametrize("route", ["batch", "retry"])
@pytest.mark.parametrize("force", [False, True])
def test_contradictory_persisted_hashes_denied_without_batch_rows(env, route, force):
    """P0C-SOL61-R2-F1: version hash vs documents hash must be contrasted even
    when ``committed_hashes`` is empty (the normal state after ``retry``).
    """
    e = env
    _seed_retry_contradiction(e)
    before = doc_state(e)
    gen = generation(e)
    e.a.write_text(body("UPDATED"))
    res = ingest(e, e.a, route, force=force)
    if route == "batch":
        assert res["committed"] == 0 and res["failed"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before
    assert generation(e) == gen


# ---------------------------------------------------------------------------
# R2-F2 — CREATE must not run over orphan index entries
# ---------------------------------------------------------------------------

def _add_index_ghost(e, variant):
    if variant == "fts":
        e.d._conn.execute(
            "INSERT INTO chunks_fts(chunk_id, doc_id, text) "
            "VALUES ('ghost','audit-three','ghost')")
        e.d._conn.commit()
    else:
        e.vs.add(ids=["ghost"],
                 embeddings=np.ones((1, 1024), dtype=np.float32),
                 documents=["ghost"],
                 metadatas=[{"doc_id": "audit-three"}])
    state = ownership.OwnershipStore(e.d._conn, e.vs).load("audit-three")
    assert not state.has_any_chunks and not state.has_documents
    assert not state.fts_matches or not state.chroma_matches


@pytest.mark.parametrize("variant", ["fts", "chroma"])
@pytest.mark.parametrize("route", ["batch", "retry"])
@pytest.mark.parametrize("force", [False, True])
def test_orphan_index_entry_blocks_create(env, variant, route, force):
    """P0C-SOL61-R2-F2: a document is new only if nothing of it exists in any
    store, not just in SQLite.
    """
    e = env
    _add_index_ghost(e, variant)
    before = doc_state(e)
    assert generation(e) is None
    res = ingest(e, e.a, route, force=force)
    if route == "batch":
        assert res["committed"] == 0 and res["failed"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before
    assert generation(e) is None


@pytest.mark.parametrize("route", ["batch", "retry"])
def test_new_document_in_empty_stores_still_creates(env, route):
    """Positive control: fail-closed CREATE must not break the legitimate case."""
    e = env
    decision = ownership.evaluate(
        e.d._conn, e.vs, doc_id="audit-three", origin_path=e.a,
        content_hash=compute_md5(e.a), chunk_dicts=[],
        classification=ownership.classification_from_fields(
            None, None, None, None, None, []),
        force=False,
    )
    assert decision.kind is ownership.DecisionKind.CREATE, decision
    res = ingest(e, e.a, route)
    committed = res["committed"] if route == "batch" else (1 if res > 0 else 0)
    assert committed == 1
    assert generation(e) == 1


def test_pure_core_r2_guards():
    # R2-F1: contradiction is rejected without batch corroboration.
    assert ownership.decide(
        _state(committed_hashes=(), documents_content_hash="other"),
        _incoming()).kind is ownership.DecisionKind.DENY_INCOHERENT
    # R2-F2: CREATE only when every store is empty/coherent.
    fresh = dict(has_version=False, has_documents=False, chunks=(),
                 chunk_ids_version=(), classification=(None, None, None, None, None, ()))
    assert ownership.decide(_state(**fresh), _incoming()).kind is \
        ownership.DecisionKind.CREATE
    for kw in (dict(fts_matches=False), dict(chroma_matches=False),
               dict(chroma_verifiable=False)):
        assert ownership.decide(_state(**fresh, **kw), _incoming()).kind is \
            ownership.DecisionKind.DENY_INCOHERENT


# ---------------------------------------------------------------------------
# CLOSURE-F1 — legacy branch must corroborate the documentary hash
# ---------------------------------------------------------------------------

def _demote_to_legacy(e):
    """Turn a versioned document into a supported legacy state.

    Drops only the ``document_versions`` row: the document, its chunks, the
    COMMITTED run and its batch rows survive, which is exactly the pre-v6
    shape the legacy branch is meant to handle.
    """
    c = e.d._conn
    c.execute("DELETE FROM document_versions WHERE doc_id='audit-three'")
    c.commit()
    assert generation(e) is None
    assert c.execute(
        "SELECT COUNT(*) FROM ingest_documents WHERE doc_id='audit-three' "
        "AND status='COMMITTED'").fetchone()[0] >= 1


@pytest.mark.parametrize("route", ["batch", "retry"])
@pytest.mark.parametrize("force", [False, True])
def test_legacy_contradictory_documents_hash_denied(env, route, force):
    """CLOSURE-F1: in the legacy branch, ``documents.content_hash`` contradicting
    the COMMITTED owner run must DENY, not publish over the contradiction.
    """
    e = env
    seed(e)                       # versioned commit through batch
    _demote_to_legacy(e)
    c = e.d._conn
    c.execute("UPDATE documents SET content_hash=? WHERE doc_id='audit-three'",
              (compute_md5(e.b),))
    c.commit()

    st = ownership.OwnershipStore(c, e.vs).load("audit-three")
    assert not st.has_version
    assert st.committed_hashes, st
    assert st.documents_content_hash not in st.committed_hashes

    before = doc_state(e)
    e.a.write_text(body("UPDATED"))
    res = ingest(e, e.a, route, force=force)
    if route == "batch":
        assert res["committed"] == 0 and res["failed"] == 1, res
    else:
        assert res == 0
    assert doc_state(e) == before
    assert generation(e) is None


def test_legacy_coherent_still_replaces(env):
    """Control positivo: a coherent legacy owner can still update its document."""
    e = env
    seed(e)
    _demote_to_legacy(e)
    c = e.d._conn
    # Keep documents.content_hash consistent with the COMMITTED run.
    run_hash = c.execute(
        "SELECT content_hash FROM ingest_documents WHERE doc_id='audit-three' "
        "AND status='COMMITTED' LIMIT 1").fetchone()[0]
    c.execute("UPDATE documents SET content_hash=? WHERE doc_id='audit-three'",
              (run_hash,))
    c.commit()

    e.a.write_text(body("UPDATED"))
    res = ingest(e, e.a)
    assert res["committed"] == 1, res
    assert generation(e) == 1


def test_pure_core_legacy_hash_guard():
    legacy = dict(has_version=False, owner_path="/ruta/a.md",
                  documents_path="/ruta/a.md", committed_hashes=("h1",),
                  documents_content_hash="h2", fts_matches=True,
                  chroma_matches=True, chroma_verifiable=True,
                  embedding_incoherent=False, pending_ops=False)
    incoming = _incoming(origin_path="/ruta/a.md")
    assert ownership.decide(_state(**legacy), incoming).kind is \
        ownership.DecisionKind.DENY_INCOHERENT
    # Corroborated documentary hash keeps the legitimate path open.
    ok = dict(legacy, documents_content_hash="h1")
    assert ownership.decide(_state(**ok), incoming).kind is \
        ownership.DecisionKind.NOOP_OWNER
