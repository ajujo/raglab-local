"""P0 regressions for document identity (RAG-05).

Two distinct files sharing a basename must never silently merge/overwrite, and
the declared frontmatter ``doc_id`` (when present) is the stable identity.
"""

import numpy as np
import pytest

from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore
from rag_lab.cli_ingest import _run_batch_ingest
from rag_lab.ingest.cleaner import clean_document


def _fake_encode(chunk_dicts, batch_size=8, device="cpu"):
    n = len(chunk_dicts)
    dense = np.random.RandomState(0).rand(n, 1024).astype(np.float32)
    return dense, {c["chunk_id"]: {} for c in chunk_dicts}


_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega "
)


def _body(tag: str, words: int = 120) -> str:
    return f"# T {tag}\n\n## S {tag}\n\n" + ((_WORDS.strip() + f" {tag} ") * words)


def _front(tag: str, doc_id: str) -> str:
    return f"---\ndoc_id: {doc_id}\ntitle: Title {tag}\n---\n\n" + _body(tag)


@pytest.fixture
def env(tmp_path, monkeypatch):
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


def test_same_basename_distinct_frontmatter_ids(env, monkeypatch):
    """Frontmatter doc_id disambiguates two files sharing a basename."""
    tmp = env["tmp"]
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    p1 = tmp / "a" / "guia.md"
    p2 = tmp / "b" / "guia.md"
    p1.write_text(_front("UNO", "yaml-uno"), encoding="utf-8")
    p2.write_text(_front("DOS", "yaml-dos"), encoding="utf-8")

    summary = _run_batch_ingest(
        paths=[p1, p2], doc_store=env["ds"], vector_store=env["vs"],
        force=False, device="cpu", strict=False,
    )
    assert summary["committed"] == 2
    docs = {r[0] for r in env["ds"]._conn.execute(
        "SELECT doc_id FROM documents")}
    assert {"yaml-uno", "yaml-dos"} <= docs
    # Each identity owns its own chunks.
    assert env["ds"].count_chunks("yaml-uno") > 0
    assert env["ds"].count_chunks("yaml-dos") > 0


def test_same_basename_same_doc_id_is_rejected(env):
    """Two paths claiming the same doc_id must not merge/overwrite."""
    tmp = env["tmp"]
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    p1 = tmp / "a" / "guia.md"
    p2 = tmp / "b" / "guia.md"
    p1.write_text(_front("UNO", "guia"), encoding="utf-8")
    p2.write_text(_front("DOS", "guia"), encoding="utf-8")

    summary = _run_batch_ingest(
        paths=[p1, p2], doc_store=env["ds"], vector_store=env["vs"],
        force=False, device="cpu", strict=False,
    )
    assert summary["committed"] == 1
    assert summary["failed"] == 1
    # Only one document row for the contested id.
    assert env["ds"]._conn.execute(
        "SELECT COUNT(*) FROM documents WHERE doc_id='guia'"
    ).fetchone()[0] == 1


def test_cleaned_paths_do_not_collide(env):
    tmp = env["tmp"]
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    p1 = tmp / "a" / "same.md"
    p2 = tmp / "b" / "same.md"
    p1.write_text("# A\n\ncontent one", encoding="utf-8")
    p2.write_text("# B\n\ncontent two", encoding="utf-8")

    c1 = clean_document(p1)
    c2 = clean_document(p2)
    assert c1 != c2
    assert c1.exists() and c2.exists()
    assert "content one" in c1.read_text()
    assert "content two" in c2.read_text()
