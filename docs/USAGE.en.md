# RAG-Lab Usage Guide

This guide covers daily use of RAG-Lab: ingesting documents, running queries, managing the system, and recovering from failures.

---

## 1. Basic daily workflow

```bash
# Activate the environment
conda activate rag-lab

# Run a query
rag-lab query "What is SDMX?"

# Interactive chat
rag-lab chat
```

That covers the most common use case. The sections below describe each area in detail.

---

## 2. Ingesting documents

The ingest pipeline has several phases. Following this order helps catch problems early:

### Full ingest workflow (recommended)

```bash
# 1. Validate frontmatter and document structure
rag-lab docs validate path/to/doc.md

# 2. Inspect metadata that will be extracted
rag-lab docs inspect path/to/doc.md

# 3. Preview how the document will be split into chunks
rag-lab docs preview-chunks path/to/doc.md

# 4. Ingest the document
rag-lab ingest --doc path/to/doc.md

# 5. Verify it was registered correctly
rag-lab docs show <doc_id>

# 6. Check consistency across stores
rag-lab reconcile --check

# 7. General health check
rag-lab doctor
```

### Ingest options

**Ingest all documents configured under `SOURCES`:**

```bash
rag-lab ingest
```

Processes all files listed in `rag_lab/config.py` under `SOURCES`.

**Ingest a single document:**

```bash
rag-lab ingest --doc path/to/doc.md
```

**Ingest all `.md` files in a directory:**

```bash
rag-lab ingest --doc path/to/directory/
```

**Force re-ingest even if the document is already ingested:**

```bash
rag-lab ingest --doc path/to/doc.md --force
```

By default, ingest is incremental: it skips documents whose hash has not changed. With `--force` the document is always reprocessed.

**Enable strict mode (warnings block ingest):**

```bash
rag-lab ingest --doc path/to/doc.md --strict
```

Without `--strict`, warnings (such as incomplete frontmatter) are logged but do not stop processing. With `--strict`, any warning or error aborts the ingest.

**Parallel ingest with multiple workers:**

```bash
rag-lab ingest --workers 4
```

Useful when ingesting large directories. The optimal number depends on available CPU and memory.

**Resume an interrupted batch:**

```bash
rag-lab ingest --resume
```

If a bulk ingest was interrupted halfway through, `--resume` picks up where it left off.

**Retry documents that failed in the last batch:**

```bash
rag-lab ingest --retry-failed
```

### Inspecting ingest history

```bash
# List recent batches
rag-lab ingest batches

# List recent runs
rag-lab ingest runs

# Filter runs by status
rag-lab ingest runs --status FAILED

# View runs for a specific document
rag-lab ingest runs --doc <doc_id>

# View full details of a specific run
rag-lab ingest show <run_id>
```

---

## 3. Querying

### Basic usage

```bash
rag-lab query "What is the difference between a DataSet and a DataStructureDefinition?"
```

The query goes through the full pipeline: embedding, hybrid search (dense + sparse), reranking, and generation with the LLM. The answer is printed with citations to the source chunks.

### Available flags

| Flag | Description |
|---|---|
| `--hyde` | Generates a hypothetical answer to expand the query before retrieval. Experimental; benchmarks show no consistent improvement. |
| `--rewrite` | Rewrites the query with domain terminology before retrieval. Experimental. |
| `--fast` | Skips the reranker. Faster, lower precision. |
| `--top-k N` | Number of chunks passed to the LLM after reranking (default: 8). |
| `--no-cache` | Bypasses the result cache. Use this to measure real latency. |
| `--profile` | Prints the time spent in each pipeline phase. |
| `--cpu-embedding` | Forces embedding to run on CPU for this session. |
| `--cpu-reranker` | Forces reranking to run on CPU for this session. |

### When to use each flag

**`--hyde`:** HyDE (Hypothetical Document Embeddings) can help with highly abstract queries or when documents use different terminology than the question. However, benchmarks for this project do not show consistent improvement, so it is disabled by default. Try it if results without it are poor.

**`--rewrite`:** Useful in technical domains where the user's query may not use the exact nomenclature of the documents. Adds latency due to an extra LLM call.

**`--fast`:** Good when speed matters more than precision — for example in demos or when doing initial corpus exploration. Without the reranker, chunks sent to the LLM are simply the top-N results from RRF fusion.

**`--no-cache`:** Essential when you need to measure real latency or when you suspect the cache has stale results.

**`--profile`:** Useful for diagnosing bottlenecks (slow embedding, slow reranker, slow LLM).

### Examples

```bash
# Simple query
rag-lab query "What is a Code List in SDMX?"

# Without reranking (fast)
rag-lab query "Explain aggregation levels" --fast

# With timing information
rag-lab query "How is a concept defined?" --profile

# Pass 12 chunks to the LLM
rag-lab query "Differences between SDMX 2.1 and 3.0" --top-k 12

# No cache, measuring real latency
rag-lab query "What is a dataflow?" --no-cache
```

---

## 4. Interactive chat

```bash
rag-lab chat
```

Launches an interactive chat where you can ask multiple questions in sequence. The chat maintains a document selection context for the duration of the session.

Inside the chat you can use special commands with the `/` prefix:

- `/docs` — view and select which documents to filter by
- `/mode` — change the query mode (for example, toggle HyDE)
- `/quit` or `/exit` — exit the chat

---

## 5. Document management

### List documents

```bash
# All documents
rag-lab docs list

# Filter by tag
rag-lab docs list --tag sdmx

# Filter by status
rag-lab docs list --status active
```

### View document details

```bash
rag-lab docs show my_doc_id
```

Shows metadata, tags, chunk count, ingest dates, and more.

### Validate a document before ingesting

```bash
rag-lab docs validate path/to/doc.md

# Strict mode (warnings are also errors)
rag-lab docs validate path/to/doc.md --strict
```

Checks the YAML frontmatter, required fields, prohibited fields, and general Markdown structure.

### Inspect extracted metadata

```bash
rag-lab docs inspect path/to/doc.md
```

Shows the metadata that will be extracted from the frontmatter without actually ingesting the document.

### Preview chunks

```bash
rag-lab docs preview-chunks path/to/doc.md

# Limit the number of chunks shown
rag-lab docs preview-chunks path/to/doc.md --limit 10
```

Very useful for verifying that the document will be split correctly before ingesting.

### Add and remove tags

```bash
rag-lab docs tag my_doc_id sdmx
rag-lab docs untag my_doc_id sdmx
```

### Delete a document

```bash
# With interactive confirmation
rag-lab docs delete my_doc_id

# Without confirmation
rag-lab docs delete my_doc_id --force
```

Removes the document from all stores (ChromaDB, SQLite, sparse index).

### Associate with a source

```bash
rag-lab docs set-source my_doc_id my_source_id
```

---

## 6. Tags

Tags let you organize documents and filter queries to a subset of the corpus.

### List all tags

```bash
rag-lab tags list
```

### Rename a tag

```bash
rag-lab tags rename sdmx-v2 sdmx
```

Renames the tag across all documents that have it.

### Delete a tag

```bash
# With confirmation
rag-lab tags delete obsolete

# Without confirmation
rag-lab tags delete obsolete --force
```

### Tags generated automatically from frontmatter

When a document has frontmatter with the recommended fields, tags are generated automatically:

| Frontmatter field | Generated tag |
|---|---|
| `domain: sdmx` | `domain:sdmx` |
| `source_type: manual` | `source_type:manual` |
| `language: en` | `lang:en` |
| `version: "2.1"` | `version:2.1` |

---

## 7. Query cache

RAG-Lab caches query results to avoid running the full pipeline again for identical questions.

### What gets cached

- Query embedding vectors
- Retrieval results (before reranking)
- Final LLM responses

### What does not get cached

- Document ingest
- `doctor` and `reconcile` results
- Feedback

### Commands

```bash
# View cache statistics (size, entries, hit rate)
rag-lab cache stats

# Inspect a specific cache entry
rag-lab cache inspect <key>

# Clear the entire cache
rag-lab cache clear

# Compact the cache (free disk space without deleting valid entries)
rag-lab cache vacuum
```

### When to invalidate the cache

After ingesting new documents or deleting existing ones, the cache may return stale results. Use `rag-lab cache clear` or pass `--no-cache` to subsequent queries until you are confident the cache reflects the current corpus state.

---

## 8. Feedback

Feedback lets you record the usefulness of results for later analysis. **It does not affect ranking or the behavior of the system in real time.**

### Adding feedback

```bash
rag-lab feedback add --query "What is a dataflow?" --chunk-id "chunk_abc123" --feedback relevant
```

### Available feedback types

| Type | Meaning |
|---|---|
| `relevant` | The retrieved chunk is relevant to the query |
| `irrelevant` | The retrieved chunk is not relevant |
| `useful` | The generated answer is useful |
| `not_useful` | The generated answer is not useful |
| `wrong_doc` | The wrong document was retrieved |
| `outdated` | The chunk's content is outdated |
| `duplicate` | The chunk is a duplicate of another already retrieved |
| `bad_citation` | The citation to the chunk in the answer is incorrect |

### Viewing and exporting feedback

```bash
# List the last 20 entries
rag-lab feedback list

# List more entries
rag-lab feedback list --limit 50

# Filter by type
rag-lab feedback list --feedback irrelevant

# View aggregate statistics
rag-lab feedback stats

# Export to JSON
rag-lab feedback export

# Export to a specific file
rag-lab feedback export --output feedback_export.json
```

### Clearing feedback

```bash
rag-lab feedback clear --yes
```

---

## 9. Doctor

`rag-lab doctor` is the system health check. It verifies that all components are in a good state.

### When to run it

- After installing or updating RAG-Lab
- After a bulk `rag-lab ingest`
- When a query returns unexpected results
- Periodically as preventive maintenance

### What it checks

| Check | Description |
|---|---|
| `config` | Environment variables and configuration parameters |
| `docstore` | SQLite integrity (chunks, metadata) |
| `chromadb` | Vector collection status |
| `fts5` | Full-text search index |
| `sparse_coverage` | Sparse index coverage |
| `reconcile` | Inconsistencies between stores |
| `ingest_health` | Status of recent ingest runs |
| `test_query` | Runs an end-to-end test query |

### Running selective checks

```bash
# Only check configuration and stores
rag-lab doctor --checks config,docstore,chromadb

# Run a test query with a specific question
rag-lab doctor --query "What is SDMX?"
```

### Interpreting the output

- `PASS` — component is healthy
- `WARN` — possible non-critical issue
- `FAIL` — issue that requires attention

---

## 10. Reconcile

`rag-lab reconcile` detects and repairs inconsistencies between stores (ChromaDB, SQLite, sparse index, FTS5).

### When to run it

- After any bulk `rag-lab ingest`
- If `doctor` reports inconsistencies between stores
- If you delete documents manually (outside the CLI)
- As a periodic maintenance step

### Modes

```bash
# Detect problems only, do not repair
rag-lab reconcile --check

# Remove orphans in ChromaDB (chunks with no docstore entry)
rag-lab reconcile --repair

# Repair FTS5 duplicates
rag-lab reconcile --repair-fts

# Back-fill NULL model metadata
rag-lab reconcile --repair-metadata

# Save a JSON report
rag-lab reconcile --check --report-json reconcile_report.json
```

Multiple flags can be combined:

```bash
rag-lab reconcile --repair --repair-fts --repair-metadata
```

---

## 11. Diagnose

`rag-lab diagnose` goes beyond `doctor`: it lets you analyze in detail what the pipeline is retrieving for a specific query.

### When to use diagnose vs doctor

- **`doctor`** — checks the general health of the system (stores, config, connectivity).
- **`diagnose`** — debugs why a specific query returns good or bad results.

### Usage

```bash
# Diagnose a query
rag-lab diagnose --query "What is an SDMX dataflow?"

# With detailed retrieval explanation
rag-lab diagnose --query "What is a dataflow?" --explain

# Limit to a specific document
rag-lab diagnose --query "Explain the structure" --doc-id my_doc_id

# Filter by tag
rag-lab diagnose --query "Basic concepts" --tag sdmx

# Exclude documents with a tag
rag-lab diagnose --query "Basic concepts" --exclude-tag draft
```

`--explain` shows the score for each retrieved chunk, why it ranked where it did, and which pipeline phase selected it.

---

## 12. Benchmark

`rag-lab benchmark` evaluates pipeline performance against a set of reference queries and expected answers.

### When to run it

- When changing retrieval parameters (`RETRIEVAL_TOP_K`, `RRF_K`, `RERANK_TOP_K`)
- When changing the embedding or reranker model
- When enabling or disabling HyDE or rewrite
- To compare pipeline variants before a significant change

### Usage

```bash
# Official suite with all variants
rag-lab benchmark --suite official --variants full

# Dense retrieval only
rag-lab benchmark --suite official --variants dense

# No cache (reproducible results)
rag-lab benchmark --suite official --variants full --no-cache

# Save results to a file
rag-lab benchmark --suite official --variants full --output results.json
```

### Available suites

| Suite | Description |
|---|---|
| `official` | Project reference query set |
| `candidates` | Candidate queries under evaluation |
| `all` | All suites combined |

### Pipeline variants

| Variant | Description |
|---|---|
| `full` | Full pipeline: dense + sparse embeddings + reranker |
| `dense` | Dense embedding only |
| `bm25` | Sparse index only (BM25-like) |
| `hybrid` | Dense + sparse without reranker |

### Interpreting results

Benchmarks report metrics such as nDCG@10 and R@5. Higher values are better. Compare the variant you are evaluating against the reference baseline (currently v1.11) to determine whether the change is a genuine improvement.

---

## 13. YAML frontmatter

YAML frontmatter is the primary mechanism for associating metadata with a document before ingesting it. Well-structured frontmatter improves search quality and tag-based filtering.

### Complete structure

```yaml
---
doc_id: my_doc_id          # REQUIRED — unique document identifier
title: Document Title       # Recommended
domain: sdmx               # Recommended — generates tag domain:sdmx
source_type: manual        # Recommended — generates tag source_type:manual
language: en               # Recommended — generates tag lang:en
version: "2.1"             # Optional — generates tag version:2.1
tags:
  - sdmx
  - technical_notes
---
```

### Required fields

| Field | Type | Description |
|---|---|---|
| `doc_id` | string | Unique identifier. No spaces. Should be stable across updates. |

### Recommended fields

| Field | Type | Description |
|---|---|---|
| `title` | string | Human-readable document title |
| `domain` | string | Subject domain. Generates tag `domain:<value>` |
| `source_type` | string | Source type (`manual`, `spec`, `guide`, etc.). Generates tag `source_type:<value>` |
| `language` | string | ISO 639-1 language code (`en`, `es`, etc.). Generates tag `lang:<value>` |
| `version` | string | Document version. Generates tag `version:<value>` |
| `tags` | list | Additional free-form tags |

### Prohibited fields

| Field | Reason |
|---|---|
| `dataset` | Causes a validation ERROR |
| `dataset_id` | Causes a validation ERROR |

RAG-Lab is for narrative Markdown documents, not tabular datasets.

### Validate before ingesting

```bash
rag-lab docs validate path/to/doc.md
```

If the frontmatter is incorrect, validation describes exactly which field is missing or wrong.

---

## 14. Recovery workflow after a failed ingest

If an ingest fails partway through or produces incorrect results, follow these steps:

### 1. Identify the failed run

```bash
rag-lab ingest runs --status FAILED
rag-lab ingest show <run_id>
```

### 2. Roll back the run

```bash
rag-lab ingest rollback <run_id>
```

The rollback removes the partially ingested chunks and vectors for that run from all stores.

### 3. Verify the rollback was clean

```bash
rag-lab reconcile --check
```

If `reconcile` reports orphans, repair them:

```bash
rag-lab reconcile --repair
```

### 4. Fix the document and re-ingest

```bash
# Validate first to confirm the problem is resolved
rag-lab docs validate path/to/doc.md

# Re-ingest
rag-lab ingest --doc path/to/doc.md --force
```

### 5. Retry a failed run directly

If the failure was transient (for example, the LLM server was unavailable) and the document itself is correct:

```bash
rag-lab ingest retry <run_id>
```

### 6. Verify the final state

```bash
rag-lab docs show <doc_id>
rag-lab reconcile --check
rag-lab doctor
```

---

## Unsupported document types

RAG-Lab **only processes Markdown files (`.md`)**. The following formats are not supported:

- PDF (no text extraction)
- DOCX / ODT
- HTML
- CSV, JSON, Excel, or other tabular data formats

To use a document in another format, convert it to Markdown first, add the appropriate frontmatter, and then ingest it.
