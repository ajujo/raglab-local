"""Tests for Markdown validation (validation.py and markdown_contract.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from rag_lab.cli import app
from rag_lab.ingest.validation import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    count_tokens_approx,
)
from rag_lab.ingest.markdown_contract import (
    MarkdownValidationConfig,
    validate_markdown,
)


# ---------------------------------------------------------------------------
# ValidationReport data structures
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_empty_report_is_valid(self, tmp_path):
        report = ValidationReport(path=tmp_path / "f.md")
        assert report.is_valid
        assert not report.has_errors
        assert not report.has_warnings
        assert report.summary() == "OK"

    def test_errors_property(self, tmp_path):
        p = tmp_path / "f.md"
        report = ValidationReport(path=p, issues=[
            ValidationIssue(ValidationSeverity.ERROR, "e1", "msg"),
            ValidationIssue(ValidationSeverity.WARN, "w1", "msg"),
        ])
        assert len(report.errors) == 1
        assert len(report.warnings) == 1
        assert not report.is_valid
        assert report.has_errors
        assert report.has_warnings

    def test_summary_counts(self, tmp_path):
        p = tmp_path / "f.md"
        report = ValidationReport(path=p, issues=[
            ValidationIssue(ValidationSeverity.ERROR, "e1", "msg"),
            ValidationIssue(ValidationSeverity.ERROR, "e2", "msg"),
            ValidationIssue(ValidationSeverity.WARN, "w1", "msg"),
            ValidationIssue(ValidationSeverity.INFO, "i1", "msg"),
        ])
        summary = report.summary()
        assert "2 errors" in summary
        assert "1 warning" in summary
        assert "1 info" in summary

    def test_count_tokens_approx(self):
        # Real tokenizer or ~4 chars/token fallback: result must be positive
        # and proportional to length (longer text → more tokens).
        short = count_tokens_approx("hello world")
        long  = count_tokens_approx("hello world " * 50)
        assert short >= 1
        assert long > short
        assert count_tokens_approx("") == 1  # min 1


# ---------------------------------------------------------------------------
# validate_markdown checks
# ---------------------------------------------------------------------------

class TestMarkdownValidation:
    def _write(self, tmp_path: Path, name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_valid_markdown_has_no_issues(self, tmp_path):
        content = "# My Doc\n\n" + "Some content about the topic. " * 10 + "\n"
        doc = self._write(tmp_path, "doc.md", content)
        report = validate_markdown(doc)
        assert report.is_valid
        assert not report.issues

    def test_empty_file_is_error(self, tmp_path):
        doc = self._write(tmp_path, "empty.md", "   \n\n")
        report = validate_markdown(doc)
        assert report.has_errors
        codes = [i.code for i in report.errors]
        assert "empty_file" in codes

    def test_missing_title_warns(self, tmp_path):
        doc = self._write(tmp_path, "notitle.md", "## Section\n\nContent.\n")
        report = validate_markdown(doc)
        warn_codes = [i.code for i in report.warnings]
        assert "missing_title" in warn_codes

    def test_title_present_no_warn(self, tmp_path):
        doc = self._write(tmp_path, "ok.md", "# Title\n\n## Section\n\nContent.\n")
        report = validate_markdown(doc)
        warn_codes = [i.code for i in report.warnings]
        assert "missing_title" not in warn_codes

    def test_heading_hierarchy_skip_warns(self, tmp_path):
        doc = self._write(tmp_path, "skip.md", "# Title\n\n### Skipped H2\n\nContent.\n")
        report = validate_markdown(doc)
        warn_codes = [i.code for i in report.warnings]
        assert "heading_hierarchy_skip" in warn_codes

    def test_proper_hierarchy_no_warn(self, tmp_path):
        doc = self._write(tmp_path, "hier.md", "# Title\n\n## H2\n\n### H3\n\nContent.\n")
        report = validate_markdown(doc)
        warn_codes = [i.code for i in report.warnings]
        assert "heading_hierarchy_skip" not in warn_codes

    def test_section_too_long_warns(self, tmp_path):
        long_content = "word " * 2000  # ~2500 tokens
        doc = self._write(tmp_path, "long.md", f"# Title\n\n## Big Section\n\n{long_content}\n")
        config = MarkdownValidationConfig(max_section_tokens=500)
        report = validate_markdown(doc, config)
        warn_codes = [i.code for i in report.warnings]
        assert "section_too_long" in warn_codes

    def test_section_length_ok(self, tmp_path):
        short_content = "word " * 50
        doc = self._write(tmp_path, "short.md", f"# Title\n\n## Section\n\n{short_content}\n")
        report = validate_markdown(doc)
        warn_codes = [i.code for i in report.warnings]
        assert "section_too_long" not in warn_codes

    def test_large_table_warns(self, tmp_path):
        rows = ["| A | B |", "|---|---|"] + [f"| r{i} | v{i} |" for i in range(50)]
        doc = self._write(tmp_path, "bigtbl.md", "# Title\n\n" + "\n".join(rows) + "\n")
        config = MarkdownValidationConfig(max_table_rows=20)
        report = validate_markdown(doc, config)
        warn_codes = [i.code for i in report.warnings]
        assert "large_table" in warn_codes

    def test_small_table_ok(self, tmp_path):
        rows = ["| A | B |", "|---|---|", "| r1 | v1 |", "| r2 | v2 |"]
        doc = self._write(tmp_path, "smalltbl.md", "# Title\n\n" + "\n".join(rows) + "\n")
        report = validate_markdown(doc)
        warn_codes = [i.code for i in report.warnings]
        assert "large_table" not in warn_codes

    def test_long_line_gives_info(self, tmp_path):
        long_line = "x" * 600
        doc = self._write(tmp_path, "longline.md", f"# Title\n\n{long_line}\n")
        config = MarkdownValidationConfig(max_line_length=500)
        report = validate_markdown(doc, config)
        info_codes = [i.code for i in report.issues if i.severity == ValidationSeverity.INFO]
        assert "long_line" in info_codes

    def test_min_content_warns(self, tmp_path):
        doc = self._write(tmp_path, "tiny.md", "# T\n\nHi.\n")
        config = MarkdownValidationConfig(min_content_tokens=1000)
        report = validate_markdown(doc, config)
        warn_codes = [i.code for i in report.warnings]
        assert "min_content" in warn_codes

    def test_invalid_frontmatter_is_error(self, tmp_path):
        doc = self._write(tmp_path, "badfm.md", "---\ninvalid: [unclosed\n---\n# Title\n\nContent.\n")
        report = validate_markdown(doc)
        # Only fails if yaml is available
        try:
            import yaml
            assert report.has_errors
            assert any(i.code == "frontmatter_invalid_yaml" for i in report.errors)
        except ImportError:
            pass  # yaml not installed; skip the check

    def test_valid_frontmatter_ok(self, tmp_path):
        doc = self._write(tmp_path, "fm.md", "---\ntitle: My Doc\nauthor: Test\n---\n# Title\n\nContent.\n")
        report = validate_markdown(doc)
        assert not any(i.code in ("frontmatter_invalid_yaml", "frontmatter_unclosed") for i in report.issues)

    def test_unclosed_frontmatter_warns(self, tmp_path):
        doc = self._write(tmp_path, "unclosed.md", "---\ntitle: My Doc\n# Title\n\nContent.\n")
        report = validate_markdown(doc)
        warn_codes = [i.code for i in report.warnings]
        assert "frontmatter_unclosed" in warn_codes

    def test_estimated_chunks_high_warns(self, tmp_path):
        large_text = "word " * 5000
        doc = self._write(tmp_path, "big.md", f"# Title\n\n{large_text}\n")
        config = MarkdownValidationConfig(max_estimated_chunks=5)
        report = validate_markdown(doc, config)
        warn_codes = [i.code for i in report.warnings]
        assert "estimated_chunks_high" in warn_codes

    def test_no_issues_for_normal_doc(self, tmp_path):
        content = "\n\n".join([
            "# SDMX Reference",
            "## Introduction",
            "SDMX is a standard for the exchange of statistical data.",
            "## Methods",
            "Various methods are used in this context.",
        ])
        doc = self._write(tmp_path, "normal.md", content + "\n")
        report = validate_markdown(doc)
        assert not report.has_errors


# ---------------------------------------------------------------------------
# CLI: docs validate
# ---------------------------------------------------------------------------

class TestCLIDocsValidate:
    runner = CliRunner()

    def _write(self, tmp_path: Path, name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_valid_doc_exits_0(self, tmp_path):
        doc = self._write(tmp_path, "ok.md", "# Title\n\nSome content here.\n")
        result = self.runner.invoke(app, ["docs", "validate", str(doc)])
        assert result.exit_code == 0, result.output

    def test_empty_doc_exits_1(self, tmp_path):
        doc = self._write(tmp_path, "empty.md", "   ")
        result = self.runner.invoke(app, ["docs", "validate", str(doc)])
        assert result.exit_code == 1

    def test_warn_exits_0_in_normal_mode(self, tmp_path):
        # Missing H1 = WARN, not ERROR
        doc = self._write(tmp_path, "warn.md", "## Section\n\nContent here.\n")
        result = self.runner.invoke(app, ["docs", "validate", str(doc)])
        assert result.exit_code == 0
        assert "WARN" in result.output or "warning" in result.output.lower()

    def test_warn_exits_1_in_strict_mode(self, tmp_path):
        doc = self._write(tmp_path, "warn.md", "## Section\n\nContent here.\n")
        result = self.runner.invoke(app, ["docs", "validate", "--strict", str(doc)])
        assert result.exit_code == 1

    def test_nonexistent_file_exits_1(self, tmp_path):
        result = self.runner.invoke(app, ["docs", "validate", str(tmp_path / "ghost.md")])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Ingest validation gate
# ---------------------------------------------------------------------------

class TestIngestValidationGate:
    """Tests that _ingest_one blocks or passes based on validation results."""

    runner = CliRunner()

    @pytest.fixture(autouse=True)
    def _isolate_stores(self, tmp_path):
        """Redirect all store paths to tmp_path."""
        import rag_lab.config as config
        import rag_lab.storage.docstore as _ds_mod
        import rag_lab.storage.vector_store as _vs_mod

        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()
        orig = {
            "c.DATA_DIR": config.DATA_DIR,
            "c.STORAGE_DIR": config.STORAGE_DIR,
            "c.DOCDSTORE_SQLITE_PATH": config.DOCDSTORE_SQLITE_PATH,
            "c.VECTOR_STORE_PATH": config.VECTOR_STORE_PATH,
            "c.SPARSE_INDEX_PATH": config.SPARSE_INDEX_PATH,
            "ds.DOCDSTORE_SQLITE_PATH": _ds_mod.DOCDSTORE_SQLITE_PATH,
            "vs.VECTOR_STORE_PATH": _vs_mod.VECTOR_STORE_PATH,
        }
        config.DATA_DIR = tmp_path / "data"
        config.STORAGE_DIR = storage_dir
        config.DOCDSTORE_SQLITE_PATH = storage_dir / "docstore.sqlite"
        config.VECTOR_STORE_PATH = storage_dir / "chroma_db"
        config.SPARSE_INDEX_PATH = storage_dir / "sparse_index.json"
        _ds_mod.DOCDSTORE_SQLITE_PATH = storage_dir / "docstore.sqlite"
        _vs_mod.VECTOR_STORE_PATH = storage_dir / "chroma_db"
        yield
        config.DATA_DIR = orig["c.DATA_DIR"]
        config.STORAGE_DIR = orig["c.STORAGE_DIR"]
        config.DOCDSTORE_SQLITE_PATH = orig["c.DOCDSTORE_SQLITE_PATH"]
        config.VECTOR_STORE_PATH = orig["c.VECTOR_STORE_PATH"]
        config.SPARSE_INDEX_PATH = orig["c.SPARSE_INDEX_PATH"]
        _ds_mod.DOCDSTORE_SQLITE_PATH = orig["ds.DOCDSTORE_SQLITE_PATH"]
        _vs_mod.VECTOR_STORE_PATH = orig["vs.VECTOR_STORE_PATH"]

    def test_ingest_blocks_on_error_and_writes_nothing(self, tmp_path):
        """Empty file → ERROR → no store writes, returns 0."""
        from rag_lab.cli_ingest import _ingest_one
        from rag_lab.storage.docstore import DocStore
        from rag_lab.storage.vector_store import VectorStore

        doc = tmp_path / "empty.md"
        doc.write_text("   ", encoding="utf-8")

        ds = DocStore()
        ds.initialize()
        vs = VectorStore()
        vs.initialize()

        with patch.object(ds, "add") as mock_add:
            n = _ingest_one(
                source_path=doc,
                doc_store=ds,
                vector_store=vs,
                force=True,
                device="cpu",
            )

        assert n == 0
        mock_add.assert_not_called()
        ds.close()

    def test_ingest_continues_on_warn_normal_mode(self, tmp_path):
        """Missing H1 → WARN → ingest proceeds past the validation gate."""
        from rag_lab.cli_ingest import _ingest_one
        from rag_lab.storage.docstore import DocStore
        from rag_lab.storage.vector_store import VectorStore
        from rag_lab.ingest.validation import ValidationReport, ValidationIssue, ValidationSeverity
        import numpy as np

        doc = tmp_path / "nowarn.md"
        doc.write_text("## Section\n\nThis is sufficient content for ingestion.\n", encoding="utf-8")

        ds = DocStore()
        ds.initialize()
        vs = VectorStore()
        vs.initialize()

        warn_report = ValidationReport(path=doc, issues=[
            ValidationIssue(ValidationSeverity.WARN, "missing_title", "No H1 heading found.")
        ])
        fake_dense = np.zeros((1, 1024), dtype=np.float32)

        # encode_chunks is imported locally inside _ingest_one via
        # `from rag_lab.embedding.encoder import encode_chunks`, so we patch it there.
        with (
            patch("rag_lab.ingest.markdown_contract.validate_markdown", return_value=warn_report),
            patch("rag_lab.embedding.encoder.encode_chunks", return_value=(fake_dense, {})),
            patch.object(vs, "add"),
        ):
            n = _ingest_one(
                source_path=doc,
                doc_store=ds,
                vector_store=vs,
                force=True,
                device="cpu",
            )

        # Should NOT return 0 — warns don't block in normal mode
        assert n > 0
        ds.close()

    def test_ingest_blocks_on_warn_strict_mode(self, tmp_path):
        """Missing H1 → WARN → blocks when --strict is set."""
        from rag_lab.cli_ingest import _ingest_one
        from rag_lab.storage.docstore import DocStore
        from rag_lab.storage.vector_store import VectorStore

        doc = tmp_path / "strict.md"
        doc.write_text("## Section\n\nContent.\n", encoding="utf-8")

        ds = DocStore()
        ds.initialize()
        vs = VectorStore()
        vs.initialize()

        with patch.object(ds, "add") as mock_add:
            n = _ingest_one(
                source_path=doc,
                doc_store=ds,
                vector_store=vs,
                force=True,
                device="cpu",
                strict=True,
            )

        assert n == 0
        mock_add.assert_not_called()
        ds.close()

    def test_ingest_cli_blocks_on_error(self, tmp_path):
        """CLI `ingest --doc <empty>` shows validation error message."""
        doc = tmp_path / "bad.md"
        doc.write_text("  ", encoding="utf-8")

        result = self.runner.invoke(app, ["ingest", "--doc", str(doc)])
        # v1.16+: batch pipeline reports FAILED + validation error message
        output_lower = result.output.lower()
        assert "failed" in output_lower or "error" in output_lower
        assert result.exit_code == 0  # exits gracefully
