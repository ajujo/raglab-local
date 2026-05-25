"""Tests for the v1.19 frontmatter metadata contract.

Covers:
- FrontmatterData parser (parse_frontmatter, extract_h1_title)
- markdown_contract field-level validation
- MetadataStore new columns (domain, source_type, language, version)
- Tag auto-import during ingest simulation
- FilterSpec classification filters
- Scope guards (dataset/dataset_id, CSV/Parquet/DuckDB/AutoML absent)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_lab.ingest.frontmatter import (
    FrontmatterData,
    extract_h1_title,
    parse_frontmatter,
)
from rag_lab.ingest.markdown_contract import validate_markdown
from rag_lab.ingest.validation import ValidationSeverity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_md(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _ms_in_memory():
    from rag_lab.storage.metadata_store import MetadataStore
    conn = sqlite3.connect(":memory:")
    ms = MetadataStore(conn=conn)
    ms.initialize()
    return ms, conn


# ---------------------------------------------------------------------------
# 1. Frontmatter parser
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_full_frontmatter(self):
        text = """\
---
doc_id: sdmx_guide
title: SDMX User Guide 2.1
domain: sdmx
source_type: manual
language: en
version: "2.1"
tags:
  - sdmx
  - technical_notes
---
# Heading
"""
        fm = parse_frontmatter(text)
        assert fm.doc_id == "sdmx_guide"
        assert fm.title == "SDMX User Guide 2.1"
        assert fm.domain == "sdmx"
        assert fm.source_type == "manual"
        assert fm.language == "en"
        assert fm.version == "2.1"
        assert fm.tags == ["sdmx", "technical_notes"]

    def test_minimal_frontmatter(self):
        text = "---\ndoc_id: my_doc\n---\n# Title\n"
        fm = parse_frontmatter(text)
        assert fm.doc_id == "my_doc"
        assert fm.title is None
        assert fm.domain is None
        assert fm.tags == []

    def test_no_frontmatter_returns_empty(self):
        text = "# Just a heading\nSome content."
        fm = parse_frontmatter(text)
        assert fm.doc_id is None
        assert fm.tags == []
        assert fm.raw == {}

    def test_frontmatter_without_tags(self):
        text = "---\ndoc_id: x\ntitle: X\n---\n# X\n"
        fm = parse_frontmatter(text)
        assert fm.tags == []

    def test_derived_tags_full(self):
        text = """\
---
doc_id: d
domain: SDMX
source_type: Manual
language: EN
version: "2.1"
---
"""
        fm = parse_frontmatter(text)
        assert "domain:sdmx" in fm.derived_tags
        assert "source_type:manual" in fm.derived_tags
        assert "lang:en" in fm.derived_tags
        assert "version:2.1" in fm.derived_tags

    def test_all_tags_union(self):
        text = """\
---
doc_id: d
domain: sdmx
tags:
  - explicit_tag
---
"""
        fm = parse_frontmatter(text)
        all_tags = fm.all_tags
        assert "explicit_tag" in all_tags
        assert "domain:sdmx" in all_tags

    def test_duplicate_tags_deduplicated(self):
        text = "---\ndoc_id: d\ntags:\n  - sdmx\n  - sdmx\n---\n"
        fm = parse_frontmatter(text)
        assert fm.tags.count("sdmx") == 1

    def test_unclosed_frontmatter_returns_empty(self):
        text = "---\ndoc_id: d\ntitle: T\n# H1\n"
        fm = parse_frontmatter(text)
        assert fm.doc_id is None

    def test_version_as_number_converted_to_string(self):
        text = "---\ndoc_id: d\nversion: 2.1\n---\n"
        fm = parse_frontmatter(text)
        # YAML parses 2.1 as float; _str_or_none converts to string
        assert fm.version is not None
        assert "2" in fm.version

    def test_no_derived_tags_when_fields_absent(self):
        text = "---\ndoc_id: d\ntitle: T\n---\n"
        fm = parse_frontmatter(text)
        assert fm.derived_tags == []


class TestExtractH1Title:
    def test_finds_first_h1(self):
        text = "# My Title\n## Sub\nContent."
        assert extract_h1_title(text) == "My Title"

    def test_ignores_h2_plus(self):
        text = "## Section\n### Sub\nContent."
        assert extract_h1_title(text) is None

    def test_skips_frontmatter_block(self):
        text = "---\ntitle: FM title\n---\n# Real H1\n"
        assert extract_h1_title(text) == "Real H1"

    def test_no_heading_returns_none(self):
        text = "Just content, no heading."
        assert extract_h1_title(text) is None


# ---------------------------------------------------------------------------
# 2. Frontmatter validation contract
# ---------------------------------------------------------------------------

class TestFrontmatterValidation:
    def test_full_valid_frontmatter(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", """\
---
doc_id: test_doc
title: Test Document
domain: sdmx
source_type: manual
language: en
version: "1.0"
tags:
  - sdmx
  - test
---
# Test Document
Content here.
""")
        report = validate_markdown(doc)
        assert not report.has_errors
        # domain/source_type/language are set, so no missing field warnings for them
        field_warn_codes = {
            i.code for i in report.warnings
            if i.code in ("frontmatter_missing_domain", "frontmatter_missing_source_type",
                          "frontmatter_missing_language")
        }
        assert not field_warn_codes

    def test_minimal_valid_frontmatter(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", """\
---
doc_id: minimal_doc
title: Minimal
---
# Minimal
Content.
""")
        report = validate_markdown(doc)
        assert not report.has_errors
        # Missing domain/source_type/language generate WARNs — but no ERRORs
        warn_codes = {i.code for i in report.warnings}
        assert "frontmatter_missing_doc_id" not in warn_codes

    def test_missing_doc_id_is_error(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", """\
---
title: No doc_id here
domain: sdmx
---
# Title
Content.
""")
        report = validate_markdown(doc)
        error_codes = {i.code for i in report.errors}
        assert "frontmatter_missing_doc_id" in error_codes

    def test_no_frontmatter_warns(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", "# Title\nContent.")
        report = validate_markdown(doc)
        warn_codes = {i.code for i in report.warnings}
        assert "frontmatter_missing" in warn_codes

    def test_tags_not_list_is_error(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", """\
---
doc_id: d
tags: "not a list"
---
# D
Content.
""")
        report = validate_markdown(doc)
        error_codes = {i.code for i in report.errors}
        assert "frontmatter_tags_not_list" in error_codes

    def test_non_string_tag_is_error(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", """\
---
doc_id: d
tags:
  - valid_tag
  - 123
---
# D
Content.
""")
        report = validate_markdown(doc)
        error_codes = {i.code for i in report.errors}
        assert "frontmatter_tag_not_string" in error_codes

    def test_duplicate_tags_warn(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", """\
---
doc_id: d
tags:
  - sdmx
  - sdmx
---
# D
Content.
""")
        report = validate_markdown(doc)
        warn_codes = {i.code for i in report.warnings}
        assert "frontmatter_tag_duplicate" in warn_codes

    def test_dataset_field_is_scope_error(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", """\
---
doc_id: d
dataset: my_dataset
---
# D
Content.
""")
        report = validate_markdown(doc)
        error_codes = {i.code for i in report.errors}
        assert "frontmatter_scope_violation" in error_codes

    def test_dataset_id_field_is_scope_error(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", """\
---
doc_id: d
dataset_id: my_id
---
# D
Content.
""")
        report = validate_markdown(doc)
        error_codes = {i.code for i in report.errors}
        assert "frontmatter_scope_violation" in error_codes

    def test_missing_domain_source_type_language_warn(self, tmp_path):
        doc = _make_md(tmp_path / "doc.md", """\
---
doc_id: minimal
title: Minimal
---
# Minimal
Content.
""")
        report = validate_markdown(doc)
        warn_codes = {i.code for i in report.warnings}
        assert "frontmatter_missing_domain" in warn_codes
        assert "frontmatter_missing_source_type" in warn_codes
        assert "frontmatter_missing_language" in warn_codes

    def test_old_doc_no_frontmatter_does_not_hard_fail(self, tmp_path):
        doc = _make_md(tmp_path / "old.md", "# Old Document\nLegacy content.")
        report = validate_markdown(doc)
        # Should produce warnings (no frontmatter, missing title possible), but no errors
        assert not report.has_errors


# ---------------------------------------------------------------------------
# 3. MetadataStore — new columns persisted
# ---------------------------------------------------------------------------

class TestMetadataStoreNewColumns:
    def test_upsert_and_get_all_fields(self):
        ms, _ = _ms_in_memory()
        ms.upsert_document(
            "sdmx_guide",
            title="SDMX Guide",
            domain="sdmx",
            source_type="manual",
            language="en",
            version="2.1",
        )
        doc = ms.get_document("sdmx_guide")
        assert doc["title"] == "SDMX Guide"
        assert doc["domain"] == "sdmx"
        assert doc["source_type"] == "manual"
        assert doc["language"] == "en"
        assert doc["version"] == "2.1"

    def test_old_doc_without_classification_does_not_break(self):
        ms, _ = _ms_in_memory()
        ms.upsert_document("old_doc", path="/some/path.md")
        doc = ms.get_document("old_doc")
        assert doc is not None
        assert doc["domain"] is None
        assert doc["source_type"] is None
        assert doc["language"] is None
        assert doc["version"] is None

    def test_tags_imported_from_frontmatter(self):
        ms, _ = _ms_in_memory()
        ms.upsert_document("d", title="D")
        ms.assign_tag("d", "sdmx")
        ms.assign_tag("d", "domain:sdmx")
        ms.assign_tag("d", "lang:en")
        tags = ms.get_tags_for_doc("d")
        assert "sdmx" in tags
        assert "domain:sdmx" in tags
        assert "lang:en" in tags

    def test_derived_tags_present_after_import(self):
        ms, _ = _ms_in_memory()
        ms.upsert_document("d", domain="sdmx", language="en")
        for tag in ["domain:sdmx", "lang:en"]:
            ms.assign_tag("d", tag)
        tags = ms.get_tags_for_doc("d")
        assert "domain:sdmx" in tags
        assert "lang:en" in tags

    def test_list_documents_includes_new_fields(self):
        ms, _ = _ms_in_memory()
        ms.upsert_document("d1", domain="sdmx", source_type="manual")
        docs = ms.list_documents(status=None)
        doc = next(d for d in docs if d["doc_id"] == "d1")
        assert doc["domain"] == "sdmx"
        assert doc["source_type"] == "manual"

    def test_migration_idempotent_on_existing_db(self, tmp_path):
        from rag_lab.storage.metadata_store import MetadataStore
        db = tmp_path / "test.db"
        ms1 = MetadataStore(db_path=db)
        ms1.initialize()
        ms1.upsert_document("d", title="First")
        ms1.close()
        # Re-initialize — should not fail or lose data
        ms2 = MetadataStore(db_path=db)
        ms2.initialize()
        doc = ms2.get_document("d")
        assert doc["title"] == "First"
        ms2.close()


# ---------------------------------------------------------------------------
# 4. Tags: explicit + derived auto-import
# ---------------------------------------------------------------------------

class TestTagAutoImport:
    def test_explicit_tags_imported(self):
        ms, _ = _ms_in_memory()
        ms.upsert_document("d")
        for tag in ["sdmx", "technical_notes", "metadata"]:
            ms.assign_tag("d", tag)
        tags = ms.get_tags_for_doc("d")
        assert set(tags) >= {"sdmx", "technical_notes", "metadata"}

    def test_derived_tags_created(self):
        from rag_lab.ingest.frontmatter import parse_frontmatter
        text = """\
---
doc_id: d
domain: sdmx
source_type: manual
language: en
version: "2.1"
---
"""
        fm = parse_frontmatter(text)
        assert "domain:sdmx" in fm.derived_tags
        assert "source_type:manual" in fm.derived_tags
        assert "lang:en" in fm.derived_tags
        assert "version:2.1" in fm.derived_tags


# ---------------------------------------------------------------------------
# 5. FilterSpec classification filters
# ---------------------------------------------------------------------------

class TestFilterSpecClassification:
    def test_domain_resolves_to_tag(self):
        from rag_lab.retrieval.filters import FilterSpec
        spec = FilterSpec(domain="sdmx")
        assert "domain:sdmx" in spec._effective_tags_include()

    def test_source_type_resolves_to_tag(self):
        from rag_lab.retrieval.filters import FilterSpec
        spec = FilterSpec(source_type="manual")
        assert "source_type:manual" in spec._effective_tags_include()

    def test_language_resolves_to_tag(self):
        from rag_lab.retrieval.filters import FilterSpec
        spec = FilterSpec(language="en")
        assert "lang:en" in spec._effective_tags_include()

    def test_version_resolves_to_tag(self):
        from rag_lab.retrieval.filters import FilterSpec
        spec = FilterSpec(version="2.1")
        assert "version:2.1" in spec._effective_tags_include()

    def test_combined_with_explicit_tags_include(self):
        from rag_lab.retrieval.filters import FilterSpec
        spec = FilterSpec(tags_include=["explicit"], domain="sdmx")
        effective = spec._effective_tags_include()
        assert "explicit" in effective
        assert "domain:sdmx" in effective

    def test_no_classification_filters_returns_none(self):
        from rag_lab.retrieval.filters import FilterSpec
        spec = FilterSpec()
        assert spec._effective_tags_include() is None

    def test_is_empty_false_when_domain_set(self):
        from rag_lab.retrieval.filters import FilterSpec
        spec = FilterSpec(domain="sdmx", status=None)
        assert not spec.is_empty()

    def test_is_empty_true_when_all_none(self):
        from rag_lab.retrieval.filters import FilterSpec
        spec = FilterSpec(status=None)
        assert spec.is_empty()

    def test_domain_filter_resolves_docs(self):
        from rag_lab.retrieval.filters import FilterSpec, resolve_filter
        ms, conn = _ms_in_memory()
        ms.upsert_document("d_sdmx", domain="sdmx")
        ms.assign_tag("d_sdmx", "domain:sdmx")
        ms.upsert_document("d_other", domain="finance")
        ms.assign_tag("d_other", "domain:finance")

        spec = FilterSpec(domain="sdmx", status=None)
        result = resolve_filter(conn, spec)
        assert result == ["d_sdmx"]

    def test_dataset_id_not_in_filter_spec(self):
        from rag_lab.retrieval.filters import FilterSpec
        assert not hasattr(FilterSpec(), "dataset_id")


# ---------------------------------------------------------------------------
# 6. Scope guards
# ---------------------------------------------------------------------------

class TestScopeGuards:
    def test_dataset_not_in_metadata_store(self):
        from rag_lab.storage.metadata_store import MetadataStore
        assert not hasattr(MetadataStore, "upsert_dataset")
        assert not hasattr(MetadataStore, "list_datasets")

    def test_no_csv_parquet_duckdb_loader(self):
        import importlib
        for mod_name in (
            "rag_lab.loaders.csv_loader",
            "rag_lab.loaders.parquet_loader",
            "rag_lab.loaders.duckdb_loader",
            "rag_lab.loaders.automl_loader",
        ):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(mod_name)

    def test_no_pdf_docx_html_loader(self):
        import importlib
        for mod_name in (
            "rag_lab.loaders.pdf_loader",
            "rag_lab.loaders.docx_loader",
            "rag_lab.loaders.html_loader",
        ):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(mod_name)

    def test_no_tabular_references_in_ingest(self):
        import inspect
        import rag_lab.cli_ingest as _ingest
        src = inspect.getsource(_ingest)
        for banned in ("csv", "parquet", "duckdb", "automl", "dataset_id"):
            assert banned.lower() not in src.lower(), f"Found '{banned}' in cli_ingest.py"

    def test_dataset_validation_error_in_frontmatter(self, tmp_path):
        doc = _make_md(tmp_path / "bad.md", """\
---
doc_id: d
dataset: my_data
---
# D
Content.
""")
        report = validate_markdown(doc)
        assert any(
            i.code == "frontmatter_scope_violation" and i.severity == ValidationSeverity.ERROR
            for i in report.issues
        )


# ---------------------------------------------------------------------------
# 7. docs inspect shows frontmatter fields (CLI output check)
# ---------------------------------------------------------------------------

class TestDocsInspectFrontmatter:
    def test_inspect_shows_domain(self, tmp_path):
        from typer.testing import CliRunner
        from rag_lab.cli_docs import docs_app

        doc = _make_md(tmp_path / "guide.md", """\
---
doc_id: guide
title: Guide
domain: sdmx
source_type: manual
language: en
---
# Guide
Content.
""")
        runner = CliRunner()
        result = runner.invoke(docs_app, ["inspect", str(doc)])
        assert result.exit_code == 0
        assert "sdmx" in result.output
        assert "manual" in result.output
        assert "en" in result.output

    def test_inspect_shows_no_frontmatter_doc_id(self, tmp_path):
        from typer.testing import CliRunner
        from rag_lab.cli_docs import docs_app

        doc = _make_md(tmp_path / "legacy.md", "# Legacy\nOld content.")
        runner = CliRunner()
        result = runner.invoke(docs_app, ["inspect", str(doc)])
        assert result.exit_code == 0
        # Should show derived doc_id from filename
        assert "legacy" in result.output


# ---------------------------------------------------------------------------
# 8. Reconcile tag consistency check
# ---------------------------------------------------------------------------

class TestReconcileTagConsistency:
    @staticmethod
    def _patch_db_path(tmp_path):
        """Context manager that redirects all store paths to tmp_path."""
        import rag_lab.config as config
        import rag_lab.storage.docstore as _ds_mod
        import rag_lab.storage.vector_store as _vs_mod
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            storage_dir = tmp_path / "storage"
            storage_dir.mkdir(exist_ok=True)
            db = storage_dir / "docstore.sqlite"
            orig = {
                "c.DOCDSTORE_SQLITE_PATH": config.DOCDSTORE_SQLITE_PATH,
                "c.VECTOR_STORE_PATH": config.VECTOR_STORE_PATH,
                "c.STORAGE_DIR": config.STORAGE_DIR,
                "ds.DOCDSTORE_SQLITE_PATH": _ds_mod.DOCDSTORE_SQLITE_PATH,
                "vs.VECTOR_STORE_PATH": _vs_mod.VECTOR_STORE_PATH,
            }
            config.DOCDSTORE_SQLITE_PATH = db
            config.VECTOR_STORE_PATH = storage_dir / "chroma_db"
            config.STORAGE_DIR = storage_dir
            _ds_mod.DOCDSTORE_SQLITE_PATH = db
            _vs_mod.VECTOR_STORE_PATH = storage_dir / "chroma_db"
            try:
                yield db
            finally:
                config.DOCDSTORE_SQLITE_PATH = orig["c.DOCDSTORE_SQLITE_PATH"]
                config.VECTOR_STORE_PATH = orig["c.VECTOR_STORE_PATH"]
                config.STORAGE_DIR = orig["c.STORAGE_DIR"]
                _ds_mod.DOCDSTORE_SQLITE_PATH = orig["ds.DOCDSTORE_SQLITE_PATH"]
                _vs_mod.VECTOR_STORE_PATH = orig["vs.VECTOR_STORE_PATH"]

        return _ctx()

    def test_detects_missing_derived_tags(self, tmp_path):
        from rag_lab.storage.docstore import DocStore
        from rag_lab.storage.metadata_store import MetadataStore
        from rag_lab.maintenance.reconcile import reconcile

        with self._patch_db_path(tmp_path) as db:
            ds = DocStore(db_path=db)
            ds.initialize()
            ms = MetadataStore(conn=ds._conn)
            ms.upsert_document("d", domain="sdmx")
            # Intentionally do NOT assign tag "domain:sdmx"
            ds._conn.commit()
            ds.close()

            result = reconcile(quiet=True)
            missing = result.get("docs_missing_derived_tags", [])
            entry = next((e for e in missing if e["doc_id"] == "d"), None)
            assert entry is not None
            assert "domain:sdmx" in entry["missing_tags"]

    def test_no_inconsistency_when_tags_sync(self, tmp_path):
        from rag_lab.storage.docstore import DocStore
        from rag_lab.storage.metadata_store import MetadataStore
        from rag_lab.maintenance.reconcile import reconcile

        with self._patch_db_path(tmp_path) as db:
            ds = DocStore(db_path=db)
            ds.initialize()
            ms = MetadataStore(conn=ds._conn)
            ms.upsert_document("d", domain="sdmx")
            ms.assign_tag("d", "domain:sdmx")
            ds._conn.commit()
            ds.close()

            result = reconcile(quiet=True)
            missing = result.get("docs_missing_derived_tags", [])
            assert not any(e["doc_id"] == "d" for e in missing)
