"""P0 regressions for document versioning (RAG-01, RAG-02, RAG-04).

These tests exercise the *real* batch-ingest pipeline with real SQLite/FTS and
a real (temporary) ChromaDB collection.  Only the embedding encoder is faked so
the tests are deterministic and fast.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore
from rag_lab.ingest.transaction import IngestRunStore
from rag_lab.cli_ingest import _run_batch_ingest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_encode(chunk_dicts, batch_size=8, device="cpu"):
    n = len(chunk_dicts)
    dense = np.random.RandomState(0).rand(n, 1024).astype(np.float32)
    return dense, {c["chunk_id"]: {} for c in chunk_dicts}


_SECTION_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega "
)


def _doc_with_sections(*sections):
    parts = ["# Document\n"]
    for name, words in sections:
        parts.append(f"\n## {name}\n\n")
        parts.append((_SECTION_WORDS.strip() + " ") * words)
        parts.append("\n")
    return "".join(parts)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated stores + isolated cleaner/manifest dirs + fake encoder."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr("rag_lab.ingest.cleaner.DATA_DIR", data_dir)
    monkeypatch.setattr("rag_lab.ingest.manifest.DATA_DIR", data_dir)
    monkeypatch.setattr("rag_lab.config.DATA_DIR", data_dir)
    monkeypatch.setattr(
        "rag_lab.embedding.encoder.encode_chunks", _fake_encode
    )

    ds = DocStore(db_path=tmp_path / "docstore.sqlite")
    ds.initialize()
    fts = FTSStore(db_path=tmp_path / "docstore.sqlite")
    fts.initialize()
    vs = VectorStore(storage_path=tmp_path / "chroma")
    vs.initialize()

    yield {"ds": ds, "fts": fts, "vs": vs, "tmp": tmp_path}

    ds.close()
    fts.close()


def _run(env, path, force=False):
    return _run_batch_ingest(
        paths=[path], doc_store=env["ds"], vector_store=env["vs"],
        force=force, device="cpu", strict=False,
    )


def _chunk_ids(ds, doc_id):
    return {
        r[0]
        for r in ds._conn.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))
    }


def _run_ids(ds, doc_id):
    """All ingest run ids for a doc, oldest first (rowid order, second-safe)."""
    return [
        r[0]
        for r in ds._conn.execute(
            "SELECT run_id FROM ingest_runs WHERE doc_id=? ORDER BY rowid ASC", (doc_id,)
        )
    ]


def _chroma_ids(vs, doc_id):
    data = vs.get_by_doc_id(doc_id)
    return set(data["ids"])


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# RAG-01 — failed re-ingest preserves the previous committed version
# ---------------------------------------------------------------------------

class TestFailedReingestPreservesPrevious:
    def test_failed_replacement_restores_previous_version(self, env):
        tmp = env["tmp"]
        ds, vs = env["ds"], env["vs"]

        v1 = _write(tmp, "guia.md", _doc_with_sections(("Alpha", 120), ("Beta", 120)))
        s1 = _run(env, v1)
        assert s1["committed"] == 1
        ids_v1 = _chunk_ids(ds, "guia")
        vec_v1 = _chroma_ids(vs, "guia")
        assert ids_v1 and vec_v1

        # Second ingest with modified content; force metadata failure.
        v2 = _write(tmp, "guia.md", _doc_with_sections(("Alpha", 60), ("Beta", 60), ("Gamma", 120)))
        from rag_lab.storage.metadata_store import MetadataStore

        real_upsert = MetadataStore.upsert_document
        calls = {"n": 0}

        def boom(self, *a, **k):
            calls["n"] += 1
            raise RuntimeError("injected metadata failure")

        with patch.object(MetadataStore, "upsert_document", boom):
            s2 = _run(env, v2, force=True)

        assert s2["committed"] == 0 and s2["failed"] == 1
        # RAG-01: previous version intact in SQLite, FTS and Chroma.
        assert _chunk_ids(ds, "guia") == ids_v1
        assert _chroma_ids(vs, "guia") == vec_v1
        fts_ids = {r[0] for r in ds._conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE doc_id=?", ("guia",))}
        assert fts_ids == ids_v1
        # Document metadata row survives.
        assert ds._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_id=?", ("guia",)
        ).fetchone()[0] == 1


# ---------------------------------------------------------------------------
# RAG-02 — rollback must never destroy a newer version
# ---------------------------------------------------------------------------

class TestRollbackVersionSafety:
    def test_rollback_of_old_run_refused(self, env):
        from rag_lab.ingest import versioning

        tmp = env["tmp"]
        ds, vs = env["ds"], env["vs"]
        f = _write(tmp, "doc.md", _doc_with_sections(("Alpha", 120)))
        _run(env, f)
        run_v1 = _run_ids(ds, "doc")[0]

        f2 = _write(tmp, "doc.md", _doc_with_sections(("Alpha", 60), ("Beta", 120)))
        _run(env, f2, force=True)
        ids_v2 = _chunk_ids(ds, "doc")
        vec_v2 = _chroma_ids(vs, "doc")
        run_v2 = _run_ids(ds, "doc")[-1]

        # Rolling back the old run must be refused and must not change anything.
        result = versioning.admin_rollback(run_v1, ds._conn, vs)
        assert result["status"] == "refused"
        assert _chunk_ids(ds, "doc") == ids_v2
        assert _chroma_ids(vs, "doc") == vec_v2

    def test_rollback_of_active_run_restores_previous(self, env):
        from rag_lab.ingest import versioning

        tmp = env["tmp"]
        ds, vs = env["ds"], env["vs"]
        f = _write(tmp, "doc.md", _doc_with_sections(("Alpha", 120)))
        _run(env, f)
        ids_v1 = _chunk_ids(ds, "doc")
        vec_v1 = _chroma_ids(vs, "doc")

        f2 = _write(tmp, "doc.md", _doc_with_sections(("Alpha", 60), ("Beta", 120)))
        _run(env, f2, force=True)
        run_v2 = _run_ids(ds, "doc")[-1]
        assert _chunk_ids(ds, "doc") != ids_v1

        result = versioning.admin_rollback(run_v2, ds._conn, vs)
        assert result["status"] == "rolled_back"
        assert result["restored_previous"] is True
        assert _chunk_ids(ds, "doc") == ids_v1
        assert _chroma_ids(vs, "doc") == vec_v1


# ---------------------------------------------------------------------------
# RAG-04 — a confirmed replacement drops obsolete chunks
# ---------------------------------------------------------------------------

class TestReplacementDropsObsolete:
    def test_force_reingest_removes_retired_chunks(self, env):
        tmp = env["tmp"]
        ds, vs = env["ds"], env["vs"]

        v1 = _write(tmp, "doc.md", _doc_with_sections(("Alpha", 120), ("Beta", 120)))
        _run(env, v1)
        ids_v1 = _chunk_ids(ds, "doc")
        # Determine which chunks belong to Beta by heading path.
        beta_ids = {r[0] for r in ds._conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id=? AND heading_path LIKE '%Beta%'", ("doc",))}
        assert beta_ids, "expected a Beta chunk"
        assert beta_ids <= ids_v1

        # v2 removes Beta.
        v2 = _write(tmp, "doc.md", _doc_with_sections(("Alpha", 120)))
        _run(env, v2, force=True)
        ids_v2 = _chunk_ids(ds, "doc")

        assert beta_ids - ids_v2 == beta_ids, "obsolete Beta chunks still present in SQLite"
        fts_ids = {r[0] for r in ds._conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE doc_id=?", ("doc",))}
        assert fts_ids == ids_v2
        assert _chroma_ids(vs, "doc") == ids_v2


# ---------------------------------------------------------------------------
# Crash recovery — interrupted version operations are replayed idempotently
# ---------------------------------------------------------------------------

class TestVersionOpRecovery:
    def _seed_chunk(self, ds, doc_id, chunk_id, text):
        ds.add([{
            "chunk_id": chunk_id, "doc_id": doc_id, "text": text,
            "heading_path": "H", "tipo": "texto", "posicion_relativa": 0.0,
            "n_tokens": 3, "line_start": 0, "line_end": 1,
        }])

    def test_recover_interrupted_commit_deletes_obsolete_and_sets_active(self, env):
        import json as _json
        from rag_lab.ingest import versioning

        ds, vs = env["ds"], env["vs"]
        doc = "rec_doc"
        self._seed_chunk(ds, doc, "a", "alpha")
        self._seed_chunk(ds, doc, "b", "obsolete")
        ds._conn.execute(
            "INSERT INTO version_ops(doc_id, op, run_id, state, payload_json) "
            "VALUES (?, 'COMMIT', 'run_new', 'COMMITTING', ?)",
            (doc, _json.dumps({"obsolete": ["b"], "new_ids": ["a"], "new_run": "run_new"})),
        )
        ds._conn.commit()

        n = versioning.recover_pending_ops(ds._conn, vs)
        assert n == 1
        assert _chunk_ids(ds, doc) == {"a"}
        active = ds._conn.execute(
            "SELECT active_run_id, chunk_ids FROM document_versions WHERE doc_id=?", (doc,)
        ).fetchone()
        assert active is not None and active[0] == "run_new"
        assert _json.loads(active[1]) == ["a"]
        assert ds._conn.execute(
            "SELECT COUNT(*) FROM version_ops WHERE state != 'DONE'"
        ).fetchone()[0] == 0

    def test_recover_interrupted_rollback_restores_snapshot_and_active(self, env):
        import json as _json
        from rag_lab.ingest import versioning

        ds, vs = env["ds"], env["vs"]
        doc = "rec_doc2"
        self._seed_chunk(ds, doc, "a", "old")
        self._seed_chunk(ds, doc, "c", "new-partial")
        ds._conn.execute(
            "INSERT OR REPLACE INTO version_snapshot_chunks"
            "(chunk_id, doc_id, text, heading_path, tipo, posicion_relativa, n_tokens, line_start, line_end) "
            "VALUES ('a', ?, 'old', 'H', 'texto', 0.0, 3, 0, 1)",
            (doc,),
        )
        ds._conn.execute(
            "INSERT OR REPLACE INTO version_snapshot_docs(doc_id, run_id, doc_json, tags_json) "
            "VALUES (?, 'run_old', NULL, '[]')",
            (doc,),
        )
        ds._conn.execute(
            "INSERT INTO version_ops(doc_id, op, run_id, state, payload_json) "
            "VALUES (?, 'ROLLBACK', 'run_old', 'RESTORING', ?)",
            (doc, _json.dumps({"old_ids": ["a"], "old_run": "run_old"})),
        )
        ds._conn.commit()

        versioning.recover_pending_ops(ds._conn, vs)
        assert _chunk_ids(ds, doc) == {"a"}
        active = ds._conn.execute(
            "SELECT active_run_id FROM document_versions WHERE doc_id=?", (doc,)
        ).fetchone()
        assert active is not None and active[0] == "run_old"
        assert ds._conn.execute(
            "SELECT COUNT(*) FROM version_ops WHERE state != 'DONE'"
        ).fetchone()[0] == 0
