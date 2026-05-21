"""Conftest for integration tests.

Two kinds of integration tests live here:

1. **Read-only regression tests** (test_benchmarks.py):
   Access production stores (DocStore, VectorStore, FTSStore) in READ-ONLY mode
   to validate retrieval quality against the live corpus. They MUST NOT write.

2. **Full-pipeline isolation tests** (test_full_pipeline.py):
   Ingest synthetic data into stores located under tmp_path, which are isolated
   from production. Writes to those stores are intentional.

The `guard_read_only_integration` fixture (non-autouse) can be applied explicitly
to TestBenchmarks to prevent accidental production writes.
"""

import pytest

from rag_lab.config import DOCDSTORE_SQLITE_PATH


def pytest_collection_modifyitems(config, items):
    integration_marker = pytest.mark.integration
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(integration_marker)


@pytest.fixture
def guard_read_only_integration(monkeypatch):
    """Raise AssertionError if the test writes to a production store.

    Checks only production-path stores (those whose db_path matches
    DOCDSTORE_SQLITE_PATH or the default ChromaDB path). Writes to tmp_path
    stores are not affected.
    """
    from rag_lab.storage import docstore as _ds_module
    from rag_lab.storage import vector_store as _vs_module

    _prod_sqlite = str(DOCDSTORE_SQLITE_PATH)
    _orig_ds_add = _ds_module.DocStore.add
    _orig_vs_add = _vs_module.VectorStore.add
    _orig_ds_del = _ds_module.DocStore.delete_by_doc_id
    _orig_ds_del_all = _ds_module.DocStore.delete_all

    def _guarded_ds_add(self, chunks, *args, **kwargs):
        if str(getattr(self, "db_path", "")) == _prod_sqlite:
            raise AssertionError(
                "Integration test attempted to write chunks to the PRODUCTION DocStore. "
                "This test must be read-only."
            )
        return _orig_ds_add(self, chunks, *args, **kwargs)

    def _guarded_vs_add(self, chunks, *args, **kwargs):
        from rag_lab.config import STORAGE_DIR
        if "chroma_db" in str(getattr(self, "_persist_dir", "")) and \
                str(STORAGE_DIR) in str(getattr(self, "_persist_dir", "")):
            raise AssertionError(
                "Integration test attempted to write vectors to the PRODUCTION VectorStore. "
                "This test must be read-only."
            )
        return _orig_vs_add(self, chunks, *args, **kwargs)

    def _guarded_ds_del(self, doc_id, *args, **kwargs):
        if str(getattr(self, "db_path", "")) == _prod_sqlite:
            raise AssertionError(
                "Integration test attempted to delete from the PRODUCTION DocStore."
            )
        return _orig_ds_del(self, doc_id, *args, **kwargs)

    monkeypatch.setattr(_ds_module.DocStore, "add", _guarded_ds_add)
    monkeypatch.setattr(_vs_module.VectorStore, "add", _guarded_vs_add)
    monkeypatch.setattr(_ds_module.DocStore, "delete_by_doc_id", _guarded_ds_del)
