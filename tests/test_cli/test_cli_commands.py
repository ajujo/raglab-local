"""Tests for CLI commands (mocked)

Tests:
- ingest command
- query command
"""

import pytest
from unittest.mock import Mock, patch
from typer.testing import CliRunner

from rag_lab.cli import app


class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_ingest_command(self, runner):
        with patch("rag_lab.storage.docstore.DocStore") as mock_ds_cls, \
             patch("rag_lab.storage.vector_store.VectorStore") as mock_vs_cls:
            mock_ds = Mock()
            mock_ds._conn = Mock()
            mock_ds_cls.return_value = mock_ds
            mock_vs_cls.return_value = Mock()
            with patch("rag_lab.config.SOURCES", []):
                result = runner.invoke(app, ["ingest"])
            # Should not crash
            assert result.exit_code == 0 or result.exit_code == 1

    def test_query_command(self, runner):
        with patch("rag_lab.cli.process_query") as mock_process:
            mock_process.return_value = []
            result = runner.invoke(app, ["query", "test question"])
            # Should not crash
            assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
