# RAG-Lab — Markdown Frontmatter Contract (v1.19)

This document describes the canonical YAML metadata contract for Markdown documents in RAG-Lab.

---

## Purpose

YAML frontmatter is the official mechanism for associating classification metadata with a document before ingesting it. It enables:

- Identifying a document with a stable, reproducible key (`doc_id`).
- Classifying the document by domain, source type, language, and version.
- Assigning explicit tags that persist in the database.
- Filtering the corpus during retrieval without changing the ranking algorithm.

Frontmatter is **exclusive to Markdown documents**. JSON is reserved for internal system artefacts (benchmarks, audits, configurations). Tabular datasets (CSV, Parquet, DuckDB) and automatic PDF/DOCX/HTML loaders are not supported.

---

## Full recommended example

```yaml
---
doc_id: sdmx_user_guide_2_1
title: SDMX User Guide 2.1
domain: sdmx
source_type: manual
language: en
version: "2.1"
tags:
  - sdmx
  - technical_notes
  - metadata
---

# SDMX User Guide 2.1

Document content...
```

---

## Minimal accepted example

Only `doc_id` is strictly required. All other fields emit WARNs if absent but do not block ingestion:

```yaml
---
doc_id: sdmx_glossary
---

# SDMX Glossary

Document content...
```

This document passes validation without errors but emits WARNs for the absent `title`, `domain`, `source_type`, and `language` fields. With `--strict`, ingestion would be blocked.

---

## Legacy document (compatible)

A document with no frontmatter is **valid** (not an ERROR), but emits WARN `frontmatter_missing`. It is compatible with all existing systems: it is ingested, indexed, and retrieved normally.

```markdown
# SDMX Glossary (legacy)

Document content without frontmatter...
```

In this case `doc_id` is derived from the filename (legacy behaviour). The `domain`, `source_type`, `language`, and `version` fields remain empty in the database.

---

## Contract fields

### `doc_id` — required

Unique identifier for the document in the system. Must be stable, reproducible, and contain no spaces or special characters. Used as the primary key in the `documents` table.

```yaml
doc_id: sdmx_user_guide_2_1
```

- Absent → ERROR `frontmatter_missing_doc_id`. Ingestion does not proceed.
- Persisted in `documents.doc_id` (PRIMARY KEY).

### `title` — recommended

Human-readable title. If absent, the first H1 found in the content is used.

```yaml
title: SDMX User Guide 2.1
```

- Absent → WARN `frontmatter_missing_title`.
- Persisted in `documents.title`.
- Visible in `docs show` (Classification section) and `diagnose --doc-id`.

### `domain` — recommended

Thematic domain of the document. Lowercase value.

```yaml
domain: sdmx
```

- Absent → WARN `frontmatter_missing_domain`.
- Persisted in `documents.domain`.
- Automatically generates derived tag `domain:sdmx`.
- Filterable in retrieval with `FilterSpec(domain="sdmx")`.

### `source_type` — recommended

Document source type. Typical values: `manual`, `spec`, `training`, `glossary`, `notes`.

```yaml
source_type: manual
```

- Absent → WARN `frontmatter_missing_source_type`.
- Persisted in `documents.source_type`.
- Automatically generates derived tag `source_type:manual`.
- Filterable in retrieval with `FilterSpec(source_type="manual")`.

### `language` — recommended

ISO 639-1 language code for the document (`en`, `es`, `fr`, …).

```yaml
language: en
```

- Absent → WARN `frontmatter_missing_language`.
- Persisted in `documents.language`.
- Automatically generates derived tag `lang:en` (prefix `lang:`, not `language:`).
- Filterable in retrieval with `FilterSpec(language="en")`.

### `version` — optional

Version of the standard or document. Always quote values that contain dots to prevent YAML from interpreting them as floats.

```yaml
version: "2.1"
```

- Absent → no WARN.
- Persisted in `documents.version`.
- Automatically generates derived tag `version:2.1`.
- Filterable in retrieval with `FilterSpec(version="2.1")`.

### `tags` — optional

List of explicit tags assigned to the document. Must be non-empty strings.

```yaml
tags:
  - sdmx
  - technical_notes
  - metadata
```

- `tags` is not a list → ERROR `frontmatter_tags_not_list`.
- Element is not a string → ERROR `frontmatter_tag_not_string`.
- Empty element → WARN `frontmatter_tag_empty`.
- Whitespace-only element → WARN `frontmatter_tag_whitespace`.
- Duplicate element → WARN `frontmatter_tag_duplicate`.
- Persisted in the `document_tags` table.
- Filterable in retrieval with `FilterSpec(tags_include=["sdmx"])`.

---

## Explicit tags vs derived tags

RAG-Lab distinguishes two types of tags on an ingested document:

### Explicit tags

Defined in the frontmatter `tags:` field. Stored literally in `document_tags`.

```yaml
tags:
  - sdmx
  - metadata
  - smoke
```

### Derived tags

Generated automatically during ingestion from classification fields. The mapping is:

| Field | Derived tag | Example |
|-------|-------------|---------|
| `domain: sdmx` | `domain:sdmx` | `domain:sdmx` |
| `source_type: manual` | `source_type:manual` | `source_type:manual` |
| `language: en` | `lang:en` | `lang:en` |
| `version: "2.1"` | `version:2.1` | `version:2.1` |

Derived tags are auto-imported into `document_tags` alongside explicit tags during ingestion. They enable filtering by classification using the same existing tag infrastructure.

To view both types after ingestion:

```bash
rag-lab docs show <doc_id>
# Classification section:
#   tags (explicit)   metadata, sdmx, smoke
#   tags (derived)    domain:sdmx, lang:en, source_type:manual, version:2.1
```

---

## Prohibited fields

The `dataset` and `dataset_id` fields are **explicitly prohibited** in frontmatter:

```yaml
# WRONG — this will produce an ERROR:
dataset: sdmx_codelist
dataset_id: cl_freq_001
```

Error generated: `frontmatter_scope_violation` (ERROR). Ingestion does not proceed.

**Reason:** RAG-Lab is a RAG system over Markdown documents. There is no support for tabular data, datasets, CSV, Parquet, or DuckDB. The `dataset` field has no semantics in this system and its presence indicates a document classification error. If finer provenance than `source_type` is ever needed, a separate `source` field will be evaluated — but that is a future decision, not an escape hatch for tabular data.

---

## How to validate a document

### Normal validation

```bash
rag-lab docs validate path/to/doc.md
```

- Exits with code 0 and `✓ OK` if no errors.
- Exits with code 0 and a list of WARNs if there are warnings.
- Exits with code 1 if there are ERRORs (blocks ingestion).

### Strict validation (warnings block)

```bash
rag-lab docs validate --strict path/to/doc.md
```

Useful for CI pipelines or before bulk ingestion. Treats WARNs as ERRORs.

### Inspect structure and frontmatter

```bash
rag-lab docs inspect path/to/doc.md
```

Shows the parsed frontmatter with all fields (including `derived_tags`), heading structure, token estimates, chunk count estimate, and validation result.

Example output:

```
Inspect: doc.md

  Frontmatter
  doc_id                   sdmx_user_guide_2_1
  title                    SDMX User Guide 2.1
  domain                   sdmx
  source_type              manual
  language                 en
  version                  2.1
  tags                     sdmx, technical_notes, metadata
  derived_tags             domain:sdmx, source_type:manual, lang:en, version:2.1

  Structure
  file_size                42.3 KB
  total_lines              1847
  total_tokens (~)         9823
  estimated_chunks (~)     14
  validation               OK
```

---

## How to verify metadata after ingestion

### View classification in the database

```bash
rag-lab docs show <doc_id>
```

Shows Classification section (title, domain, source_type, language, version, explicit and derived tags) and Technical section (path, hash, timestamps, chunks).

### View metadata in diagnose

```bash
rag-lab diagnose --doc-id <doc_id> --query "your question" --explain
```

Shows the `Document metadata for '<doc_id>'` block with all classification fields and tags, and runs a test query filtered to that document.

### Verify store consistency

```bash
rag-lab reconcile --check
```

Confirms DocStore, ChromaDB, FTS5, and Sparse BLOBs are in sync. From v1.19, also verifies derived tag consistency: if a document has `domain=sdmx` in `documents.domain` but lacks the `domain:sdmx` tag in `document_tags`, it is reported as an inconsistency.

---

## Complete workflow for a new document

```bash
# 1. Validate the Markdown and its frontmatter
rag-lab docs validate path/to/doc.md

# 2. Inspect structure (frontmatter, tokens, estimated chunks)
rag-lab docs inspect path/to/doc.md

# 3. Preview chunks without writing anything
rag-lab docs preview-chunks path/to/doc.md

# 4. Ingest
rag-lab ingest --doc path/to/doc.md

# 5. Verify that metadata was persisted correctly
rag-lab docs show <doc_id>
rag-lab diagnose --doc-id <doc_id> --query "test question" --explain

# 6. Confirm store integrity
rag-lab reconcile --check
rag-lab doctor
```

---

## Filtering by metadata in retrieval

Classification fields from frontmatter are resolved internally as derived tags before the search. The ranking algorithm (RRF, MMR, reranker) does not change — only the candidate pool is restricted.

Programmatic usage examples:

```python
from rag_lab.retrieval.filters import FilterSpec

# Only documents from the sdmx domain
FilterSpec(domain="sdmx")

# Only English-language manuals
FilterSpec(source_type="manual", language="en")

# Specific version
FilterSpec(version="2.1")

# Combined with explicit tags
FilterSpec(domain="sdmx", tags_include=["technical_notes"])
```

Internally, `FilterSpec(domain="sdmx")` translates to `tags_include=["domain:sdmx"]` before doc_id resolution. There is no special code path per classification field — everything goes through the same tag system.

---

## Notes on future evolution

- **`source` field:** If finer document provenance is ever needed (origin URL, repository, issuing organisation), a separate `source` field will be evaluated. This decision is pending real use cases; it does not exist and is not planned in v1.19.

- **`source_type` values:** Typical current values are `manual`, `spec`, `training`, `glossary`, `notes`. There is no closed list — any lowercase string is accepted.

- **Additional languages:** `language` accepts any two-character ISO 639-1 code. The current corpus primarily uses `en` and `es`.
