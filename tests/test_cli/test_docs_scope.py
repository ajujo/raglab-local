"""Scope guard: verify dataset-related commands and options were removed."""

import typer
import pytest

from rag_lab.cli_docs import docs_app
from rag_lab.retrieval.filters import FilterSpec
from rag_lab.storage.metadata_store import MetadataStore


def _docs_command_names():
    return {cmd.name for cmd in docs_app.registered_commands}


class TestDatasetScopeRemoved:
    def test_no_set_dataset_command(self):
        assert "set-dataset" not in _docs_command_names()

    def test_no_dataset_option_in_list(self):
        import inspect
        from rag_lab.cli_docs import docs_list
        sig = inspect.signature(docs_list)
        assert "dataset" not in sig.parameters

    def test_filter_spec_has_no_dataset_id(self):
        spec = FilterSpec()
        assert not hasattr(spec, "dataset_id")

    def test_metadata_store_has_no_upsert_dataset(self):
        assert not hasattr(MetadataStore, "upsert_dataset")

    def test_metadata_store_has_no_list_datasets(self):
        assert not hasattr(MetadataStore, "list_datasets")

    def test_list_documents_has_no_dataset_id_param(self):
        import inspect
        sig = inspect.signature(MetadataStore.list_documents)
        assert "dataset_id" not in sig.parameters
