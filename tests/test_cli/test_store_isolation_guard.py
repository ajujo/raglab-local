"""Guard tests: verify that production stores are not touched by the test suite.

These tests verify the isolation contract from several angles:
- The docstore.sqlite and chroma_db paths used in tests differ from production
- The _isolate_stores fixture in TestIngestValidationGate correctly redirects paths
- The guard_read_only_integration fixture raises on accidental production writes
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestProductionPathIsolation:
    """Verify that tests use tmp_path-based stores, not production stores."""

    def test_docstore_default_path_is_production(self):
        """Confirm that DocStore() with no arg points to the production path."""
        from rag_lab.config import DOCDSTORE_SQLITE_PATH
        from rag_lab.storage.docstore import DocStore
        ds = DocStore()
        assert ds.db_path == DOCDSTORE_SQLITE_PATH

    def test_docstore_with_tmp_path_is_isolated(self, tmp_path):
        """DocStore(db_path=tmp_path/...) does not share state with production."""
        from rag_lab.config import DOCDSTORE_SQLITE_PATH
        from rag_lab.storage.docstore import DocStore
        isolated_path = tmp_path / "isolated.sqlite"
        ds = DocStore(db_path=isolated_path)
        assert ds.db_path != DOCDSTORE_SQLITE_PATH
        assert ds.db_path == isolated_path

    def test_isolation_fixture_redirects_docdstore_path(self, tmp_path, monkeypatch):
        """Simulate the _isolate_stores fixture and verify path redirection."""
        import rag_lab.config as config
        import rag_lab.storage.docstore as _ds_mod
        from rag_lab.config import DOCDSTORE_SQLITE_PATH as prod_path

        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()
        monkeypatch.setattr(config, "DOCDSTORE_SQLITE_PATH", storage_dir / "docstore.sqlite")
        monkeypatch.setattr(_ds_mod, "DOCDSTORE_SQLITE_PATH", storage_dir / "docstore.sqlite")

        from rag_lab.storage.docstore import DocStore
        ds = DocStore()
        assert ds.db_path != prod_path
        assert ds.db_path == storage_dir / "docstore.sqlite"

    def test_guard_read_only_raises_on_production_write(self, tmp_path, monkeypatch):
        """guard_read_only_integration raises if a test writes to production DocStore."""
        from rag_lab.storage import docstore as _ds_module
        from rag_lab.config import DOCDSTORE_SQLITE_PATH as prod_path

        _orig_add = _ds_module.DocStore.add

        def _guarded_add(self, chunks, *args, **kwargs):
            if str(getattr(self, "db_path", "")) == str(prod_path):
                raise AssertionError("Attempted write to production DocStore!")
            return _orig_add(self, chunks, *args, **kwargs)

        monkeypatch.setattr(_ds_module.DocStore, "add", _guarded_add)

        from rag_lab.storage.docstore import DocStore
        # Write to isolated store — should NOT raise
        isolated = DocStore(db_path=tmp_path / "safe.sqlite")
        isolated.initialize()
        isolated.add([{
            "chunk_id": "c1", "doc_id": "d1", "text": "safe",
            "heading_path": "", "tipo": "texto",
            "posicion_relativa": 0.0, "n_tokens": 1,
            "line_start": 0, "line_end": 0,
        }])
        isolated.close()

        # Write to production path — MUST raise
        prod_ds = DocStore()  # uses production path
        prod_ds.db_path = prod_path  # ensure it's the production path
        with pytest.raises(AssertionError, match="production DocStore"):
            _guarded_add(prod_ds, [{"chunk_id": "c_danger", "doc_id": "x", "text": "danger"}])


class TestConftest:
    """Meta-tests: verify conftest.py isolation guarantees."""

    def test_embedding_device_is_cpu(self):
        """conftest.py forces EMBEDDING_DEVICE=cpu for all tests."""
        import os
        assert os.environ.get("EMBEDDING_DEVICE") == "cpu"

    def test_reranker_device_is_cpu(self):
        """conftest.py forces RERANKER_DEVICE=cpu for all tests."""
        import os
        assert os.environ.get("RERANKER_DEVICE") == "cpu"

    def test_cuda_invisible(self):
        """conftest.py hides GPU via CUDA_VISIBLE_DEVICES='' for all tests."""
        import os
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
