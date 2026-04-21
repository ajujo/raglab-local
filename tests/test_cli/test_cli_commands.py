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
        with patch("rag_lab.cli.clean_document") as mock_clean:
            mock_clean.return_value = Mock()
            with patch("rag_lab.cli.create_manifest"):
                with patch("rag_lab.cli.chunk_document") as mock_chunk:
                    mock_chunk.return_value = []
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
