# RAG-Lab — CLI Reference (v1.21)

Complete reference for all commands available in `rag-lab`. All commands assume the `rag-lab`
conda environment is active and the local LLM server is available for commands that generate
answers.

---

## Table of contents

- [query](#query)
- [chat](#chat)
- [ingest](#ingest)
- [docs](#docs)
- [tags](#tags)
- [cache](#cache)
- [feedback](#feedback)
- [doctor](#doctor)
- [reconcile](#reconcile)
- [diagnose](#diagnose)
- [benchmark](#benchmark)
- [eval](#eval)

---

## query

Run a query over the ingested corpus and return a generated answer with context.

```
rag-lab query "question" [options]
```

### Options

| Option | Description |
|---|---|
| `--hyde` | Enable HyDE (Hypothetical Document Embeddings). **Disabled by default.** Benchmarked: R@5 -3.8pp, latency ×12.5. Available for experimentation. |
| `--rewrite` | Enable query rewriting with domain terminology. **Disabled by default.** No official benchmark. |
| `--fast` | Fast mode: disable the reranker and verification. Lower quality, lower latency. |
| `--top-k N` | Number of chunks to retrieve before reranking. Default: value of `RETRIEVAL_TOP_K` in config (30). |
| `--no-cache` | Ignore the query cache and force a full pipeline re-execution. |
| `--profile` | Show a detailed pipeline trace: retrieval scores, verification trace, broken-down trust score. |
| `--cpu-embedding` | Force the embedding model to use CPU regardless of the environment variable. |
| `--cpu-reranker` | Force the reranker to use CPU regardless of the environment variable. |

### Examples

```bash
# Basic query
rag-lab query "What is SDMX?"

# With full pipeline trace
rag-lab query "How does RRF work?" --profile

# No cache, no reranker, fast
rag-lab query "Definition of dimension in SDMX" --no-cache --fast

# Retrieve more candidates before reranking
rag-lab query "structure of a Data Flow" --top-k 50
```

---

## chat

Start an interactive chat session with active document selection.

```
rag-lab chat
```

No command-line options. The interactive mode allows:
- Selecting which corpus documents are active in the session
- Asking questions conversationally
- Viewing the answer with context from retrieved chunks

### Example

```bash
rag-lab chat
```

---

## ingest

Ingest documents into the corpus. Supports individual ingestion, batch ingestion, and resuming.

### Basic ingestion

```
rag-lab ingest [options]
```

Without `--doc`, ingests all documents configured in `SOURCES` inside `config.py`.

| Option | Description |
|---|---|
| `--doc PATH` | Ingest a single document at the specified path. |
| `--force` | Re-ingest the document even if it already exists in the corpus (replaces it). |
| `--resume` | Continue an interrupted ingestion from the last checkpoint. |
| `--retry-failed` | Automatically retry documents with `failed` status from the most recent run. |
| `--workers N` | Number of parallel workers for batch ingestion. Default: 1 (sequential). |
| `--strict` | Apply strict validation: Markdown WARNs block ingestion (treated as ERRORs). |

### Examples

```bash
# Ingest all configured documents
rag-lab ingest

# Ingest a specific document
rag-lab ingest --doc data/docs/sdmx_user_guide.md

# Force re-ingest (replace existing)
rag-lab ingest --doc data/docs/sdmx_user_guide.md --force

# Parallel ingestion with 4 workers and strict validation
rag-lab ingest --workers 4 --strict

# Retry failed documents from the last run
rag-lab ingest --retry-failed
```

---

### ingest batches

List ingestion batches with optional filters.

```
rag-lab ingest batches [--status STATUS] [--doc DOC]
```

| Option | Description |
|---|---|
| `--status S` | Filter by status: `success`, `failed`, `partial`, `running`. |
| `--doc D` | Filter by document name or path. |

```bash
# View all batches
rag-lab ingest batches

# View only failed batches
rag-lab ingest batches --status failed
```

---

### ingest runs

List ingestion runs with optional filters.

```
rag-lab ingest runs [--status STATUS] [--doc DOC]
```

| Option | Description |
|---|---|
| `--status S` | Filter by status: `success`, `failed`, `partial`, `running`. |
| `--doc D` | Filter by document name or path. |

```bash
# View all runs
rag-lab ingest runs

# View failed runs
rag-lab ingest runs --status failed
```

---

### ingest show

Show the details of a specific run.

```
rag-lab ingest show <run_id>
```

```bash
rag-lab ingest show abc123def456
```

---

### ingest rollback

Undo a complete ingestion, removing all chunks, vectors, and records for that run.

```
rag-lab ingest rollback <run_id>
```

```bash
rag-lab ingest rollback abc123def456
```

**Note:** Rollback is atomic. It removes the document from DocStore, ChromaDB, FTS5, and
Sparse BLOBs simultaneously. It cannot be undone.

---

### ingest retry

Retry the failed documents from a specific run.

```
rag-lab ingest retry <run_id>
```

```bash
rag-lab ingest retry abc123def456
```

---

## docs

Document corpus management.

### docs list

List all ingested documents.

```
rag-lab docs list [--tag TAG]
```

| Option | Description |
|---|---|
| `--tag TAG` | Filter documents that have the specified tag. |

```bash
# List all documents
rag-lab docs list

# List only documents tagged 'sdmx'
rag-lab docs list --tag sdmx
```

---

### docs show

Show complete information for a document: classification metadata, explicit and derived tags,
path, hash, timestamps, and chunk count.

```
rag-lab docs show <id>
```

`<id>` is the document's `doc_id` (the value of the `doc_id` field in the frontmatter).

```bash
rag-lab docs show sdmx_user_guide_2_1
```

Output includes sections:
- **Classification:** title, domain, source_type, language, version, explicit tags, derived tags
- **Technical:** path, hash, ingest timestamp, chunk count

---

### docs tag

Add a tag to a document.

```
rag-lab docs tag <id> <tag>
```

```bash
rag-lab docs tag sdmx_user_guide_2_1 reviewed
```

---

### docs untag

Remove a tag from a document.

```
rag-lab docs untag <id> <tag>
```

```bash
rag-lab docs untag sdmx_user_guide_2_1 draft
```

---

### docs delete

Delete a document from the corpus (DocStore, ChromaDB, FTS5, Sparse BLOBs).

```
rag-lab docs delete <id> [--force]
```

| Option | Description |
|---|---|
| `--force` | Delete without asking for confirmation. |

```bash
# Delete with confirmation
rag-lab docs delete sdmx_glossary_legacy

# Delete without confirmation
rag-lab docs delete sdmx_glossary_legacy --force
```

---

### docs set-source

Associate a document with a data source.

```
rag-lab docs set-source <id> <src_id>
```

```bash
rag-lab docs set-source sdmx_user_guide_2_1 sdmx_official
```

---

### docs validate

Validate a Markdown file before ingesting it. Checks frontmatter, structure, headings,
tables, and other quality criteria.

```
rag-lab docs validate <path> [--strict]
```

| Option | Description |
|---|---|
| `--strict` | Treat WARNs as ERRORs. Exits with code 1 if there are any issues. |

Exit codes:
- `0`: OK (no ERRORs; WARNs may be present in normal mode)
- `1`: ERRORs present, or WARNs present in `--strict` mode

```bash
# Normal validation
rag-lab docs validate data/docs/sdmx_user_guide.md

# Strict validation (WARNs block)
rag-lab docs validate data/docs/sdmx_user_guide.md --strict
```

---

### docs inspect

Show the complete structure of a document without ingesting it: parsed frontmatter (including
derived tags), heading tree, token and chunk estimates, and validation result.

```
rag-lab docs inspect <path>
```

```bash
rag-lab docs inspect data/docs/sdmx_user_guide.md
```

---

### docs preview-chunks

Generate the chunks that would be created when ingesting the document, without writing anything
to the stores. Useful for auditing the chunking result before committing.

```
rag-lab docs preview-chunks <path> [--limit N]
```

| Option | Description |
|---|---|
| `--limit N` | Show only the first N chunks. Default: all. |

```bash
# See all chunks that would be created
rag-lab docs preview-chunks data/docs/sdmx_user_guide.md

# See only the first 10
rag-lab docs preview-chunks data/docs/sdmx_user_guide.md --limit 10
```

---

## tags

Tag catalog management.

### tags list

List all tags in the corpus with the number of associated documents.

```
rag-lab tags list
```

```bash
rag-lab tags list
```

---

### tags rename

Rename a tag across all documents that have it.

```
rag-lab tags rename <old> <new>
```

```bash
rag-lab tags rename draft reviewed
```

---

### tags delete

Remove a tag from all documents that have it.

```
rag-lab tags delete <name> [--force]
```

| Option | Description |
|---|---|
| `--force` | Delete without asking for confirmation. |

```bash
rag-lab tags delete obsolete --force
```

---

## cache

Query cache management.

### cache stats

Show cache statistics: entry count, size, TTL, session hit rate.

```
rag-lab cache stats
```

```bash
rag-lab cache stats
```

---

### cache clear

Remove all cache entries.

```
rag-lab cache clear
```

```bash
rag-lab cache clear
```

---

### cache vacuum

Remove expired cache entries (older than 7 days by default) and compact the SQLite database.

```
rag-lab cache vacuum
```

```bash
rag-lab cache vacuum
```

---

### cache inspect

Show the content of a cache entry by its key (fingerprint).

```
rag-lab cache inspect <key>
```

```bash
rag-lab cache inspect abc123
```

---

## feedback

User feedback store management.

### feedback add

Add a feedback event for a specific query and chunk.

```
rag-lab feedback add --query "QUERY" --chunk-id "CHUNK_ID" --feedback TYPE
```

| Option | Description |
|---|---|
| `--query TEXT` | Original query text. Required. |
| `--chunk-id ID` | ID of the chunk the feedback applies to. Required. |
| `--feedback TYPE` | Feedback type. Values: `relevant`, `irrelevant`, `useful`, `not_useful`, `wrong_doc`, `outdated`, `duplicate`, `bad_citation`. |

```bash
rag-lab feedback add \
  --query "What is a Data Flow in SDMX?" \
  --chunk-id "sdmx_user_guide_2_1:chunk_042" \
  --feedback useful
```

---

### feedback list

List recorded feedback events.

```
rag-lab feedback list [--limit N] [--feedback TYPE]
```

| Option | Description |
|---|---|
| `--limit N` | Maximum number of events to show. Default: 20. |
| `--feedback TYPE` | Filter by feedback type. |

```bash
# Last 20 events
rag-lab feedback list

# Only negative feedback
rag-lab feedback list --feedback not_useful

# Last 100 bad citation events
rag-lab feedback list --limit 100 --feedback bad_citation
```

---

### feedback stats

Show aggregated feedback statistics: distribution by type, documents with the most negative
feedback, queries with the most failures.

```
rag-lab feedback stats
```

```bash
rag-lab feedback stats
```

---

### feedback export

Export all feedback events to a JSON or CSV file.

```
rag-lab feedback export [--output PATH]
```

| Option | Description |
|---|---|
| `--output PATH` | Output path. Default: stdout in JSON format. |

```bash
# Export to file
rag-lab feedback export --output feedback_export.json
```

---

### feedback clear

Delete all feedback events. This operation is irreversible.

```
rag-lab feedback clear --yes
```

`--yes` is required to prevent accidental deletion.

```bash
rag-lab feedback clear --yes
```

---

## doctor

Run a complete system health check and report the status of each component.

```
rag-lab doctor [--checks CHECKS] [--query TEXT]
```

| Option | Description |
|---|---|
| `--checks CHECKS` | Comma-separated list of checks to run. Default: all. |
| `--query TEXT` | Test query to use in the `test_query` check. Default: predefined query. |

### Available checks

| Check | What it verifies |
|---|---|
| `config` | Environment variables, configuration paths, LLM connectivity. |
| `docstore` | SQLite database integrity: tables, indices, record counts. |
| `chromadb` | ChromaDB collection: existence, vector count, consistency with DocStore. |
| `fts5` | FTS5 table: existence, row count, consistency with DocStore. |
| `sparse_coverage` | Sparse BLOB coverage: what percentage of chunks have a sparse vector. |
| `reconcile` | Runs a quick reconciliation between stores and reports inconsistencies. |
| `ingest_health` | Status of recent ingestion runs: failed, partial, orphaned. |
| `test_query` | Runs a complete test query and verifies it produces a result. |

### Examples

```bash
# Full health check
rag-lab doctor

# Only storage checks
rag-lab doctor --checks docstore,chromadb,fts5,sparse_coverage

# With a custom test query
rag-lab doctor --query "What is SDMX?"
```

---

## reconcile

Verify and repair inconsistencies between system stores (DocStore, ChromaDB, FTS5,
Sparse BLOBs).

```
rag-lab reconcile [options]
```

| Option | Description |
|---|---|
| `--check` | Only report inconsistencies without modifying anything. Exit code 1 if issues found. |
| `--repair` | Remove ChromaDB orphans (chunks in ChromaDB without a DocStore entry). |
| `--repair-fts` | Fix duplicates in the FTS5 table. |
| `--repair-metadata` | Backfill NULL model metadata fields. |
| `--report-json PATH` | Write the reconciliation report in JSON format to the specified path. |

### Examples

```bash
# Check only (no modifications)
rag-lab reconcile --check

# Repair ChromaDB orphans
rag-lab reconcile --repair

# Repair FTS5 duplicates
rag-lab reconcile --repair-fts

# Backfill NULL metadata
rag-lab reconcile --repair-metadata

# Check and save report
rag-lab reconcile --check --report-json reconcile_report.json
```

---

## diagnose

Run a diagnostic query with a detailed pipeline trace to investigate retrieval problems or
verify the state of a specific document.

```
rag-lab diagnose [options]
```

| Option | Description |
|---|---|
| `--query TEXT` | Query to run in the diagnostic. |
| `--explain` | Show detailed explanation of each pipeline phase. |
| `--doc-id ID` | Restrict retrieval to a specific document. |
| `--tag TAG` | Restrict retrieval to documents with this tag. |
| `--exclude-tag TAG` | Exclude documents with this tag from retrieval. |

### Examples

```bash
# Diagnose a query with full explanation
rag-lab diagnose --query "What is a Data Flow?" --explain

# Diagnose on a specific document
rag-lab diagnose --query "Data Flow structure" --doc-id sdmx_user_guide_2_1 --explain

# Diagnose with tag filter
rag-lab diagnose --query "basic concepts" --tag sdmx --explain

# Diagnose excluding obsolete documents
rag-lab diagnose --query "SDMX syntax" --exclude-tag obsolete --explain
```

---

## benchmark

Run the retrieval benchmark over the SDMX corpus with the evaluation query set.

```
rag-lab benchmark [options]
rag-lab benchmark run [options]
```

`run` is an alias for `benchmark` — both forms are equivalent.

| Option | Description |
|---|---|
| `--suite SUITE` | Suite to run: `official`, `candidates`, `all`. Default: `official`. |
| `--variants V [V...]` | Variants to benchmark. Default: `full`. |
| `--no-cache` | Disable the query cache during the benchmark. |
| `--output PATH` | Write results to the specified path (JSON). |
| `--top-k N` | Number of chunks to retrieve in the benchmark. |
| `--rrf-k N` | K constant for the RRF algorithm. |

### Available suites

| Suite | Description |
|---|---|
| `official` | Official evaluation query set over the SDMX corpus. Reference for comparisons. |
| `candidates` | Candidate query set under evaluation. |
| `all` | All suites. |

### Reported metrics

| Metric | Current value (v1.11 baseline) |
|---|---|
| R@5 | 0.821 |
| R@10 | 0.896 |
| R@30 | 0.978 |
| MRR | 0.939 |
| nDCG@10 | 0.837 |

### Examples

```bash
# Full benchmark with official suite
rag-lab benchmark --suite official --variants full --no-cache

# Benchmark with 'run' alias
rag-lab benchmark run --suite official --variants full --no-cache

# Benchmark with custom parameters and saved results
rag-lab benchmark --suite official --top-k 50 --rrf-k 30 --output results.json

# Benchmark all suites
rag-lab benchmark --suite all --no-cache
```

---

## eval

Runs the full pipeline end-to-end over a query set and saves all output to JSONL.
Each line captures `question → contexts → answer → citations → trust_score` — a format
consumable by any external evaluator (RAGAS, TruLens, human review).

```
rag-lab eval run   [options]
rag-lab eval list  [options]
rag-lab eval show  <run_id>
```

### eval run

```
rag-lab eval run [options]
```

| Option | Description |
|---|---|
| `--suite SUITE` | Query suite to evaluate. Default: `official`. |
| `--output PATH` | Output JSONL file. Default: `data/eval_runs/<suite>_<timestamp>.jsonl`. |
| `--limit N` | Evaluate only the first N queries in the suite. |
| `--queries q001,q002` | Specific query IDs to evaluate (comma-separated). |
| `--top-k N` | Retrieval pool size before reranking. Default: 50. |
| `--rerank-top-k N` | Top-K chunks passed to the LLM. Default: 8. |
| `--temperature F` | LLM temperature. Default: 0.0 (deterministic). |

JSONL output is written **incrementally**: if the run is interrupted, results produced
before the failure are preserved.

### eval list

```
rag-lab eval list [--limit N]
```

Lists previous runs in `data/eval_runs/`, sorted by most recent. Shows query count,
error count, mean trust score, and modification date.

### eval show

```
rag-lab eval show <run_id>
```

Prints a summary of a run: mean score, mean latency, trust level distribution, and
individual errors if any. `<run_id>` can be the full file name or a prefix.

### JSONL schema

Each line in the output file has this schema:

```json
{
  "query_id": "q001",
  "question": "What is SDMX?",
  "language": "en",
  "category": "glossary_definition",
  "answer": "SDMX is...",
  "contexts": ["chunk text 1", "chunk text 2"],
  "context_metadata": [
    {"chunk_id": "...", "doc_id": "SDMX_Glossary", "heading_path": "...", "rerank_score": 0.87}
  ],
  "citations": [{"chunk_id": "...", "doc_id": "...", "lines": "10-25", "status": "valid"}],
  "trust_score": 0.87,
  "trust_level": "HIGH",
  "latency_ms": 245,
  "expected_answer": null,
  "expected_doc_ids": ["SDMX_Glossary"],
  "doc_relevance": {"SDMX_Glossary": 3},
  "error": null
}
```

`error` is `null` on successful runs; it contains the exception message if a query fails.

### Examples

```bash
# Quick smoke test: 5 queries
rag-lab eval run --suite official --limit 5 --output /tmp/smoke.jsonl

# Full suite (v1.21 baseline)
rag-lab eval run --suite official --output data/eval_runs/v1.21_baseline.jsonl

# Review results
rag-lab eval list
rag-lab eval show v1.21_baseline
```

---

## Common workflows

### Adding a new document

```bash
# 1. Validate Markdown quality
rag-lab docs validate path/to/doc.md

# 2. Inspect frontmatter and structure
rag-lab docs inspect path/to/doc.md

# 3. Preview chunks that would be created
rag-lab docs preview-chunks path/to/doc.md

# 4. Ingest
rag-lab ingest --doc path/to/doc.md

# 5. Verify metadata
rag-lab docs show <doc_id>

# 6. Confirm integrity
rag-lab reconcile --check
rag-lab doctor --checks docstore,chromadb,fts5
```

### Investigating a query that is not working well

```bash
# Full pipeline trace with document filter
rag-lab diagnose --query "your question" --explain

# With a specific document
rag-lab diagnose --query "your question" --doc-id <doc_id> --explain

# Normal query with profile
rag-lab query "your question" --profile --no-cache
```

### Periodic health check

```bash
rag-lab doctor
rag-lab reconcile --check
```

### Verifying a retrieval improvement

```bash
# Benchmark without cache to measure the real change
rag-lab benchmark --suite official --variants full --no-cache

# Compare against saved baseline
rag-lab benchmark --suite official --no-cache --output new_result.json
```

---

*Version: v1.21*
