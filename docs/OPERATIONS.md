# RAG-Lab Operations Guide

Operational reference for diagnosing, maintaining, and protecting the RAG-Lab system.

---

## Quick-reference commands

| Goal | Command |
|------|---------|
| Full health check | `python -m rag_lab.doctor` |
| Validate a document | `rag-lab docs validate path/to/doc.md` |
| Validate (strict — warns block) | `rag-lab docs validate --strict path/to/doc.md` |
| Inspect document structure | `rag-lab docs inspect path/to/doc.md` |
| Preview chunks without ingesting | `rag-lab docs preview-chunks path/to/doc.md` |
| Ingest a document (transactional) | `rag-lab ingest --doc path/to/doc.md` |
| Ingest with strict validation | `rag-lab ingest --strict --doc path/to/doc.md` |
| Ingest all sources | `rag-lab ingest` |
| Resume crashed ingest runs | `rag-lab ingest --resume` |
| Retry all failed ingest runs | `rag-lab ingest --retry-failed` |
| List ingest run history | `rag-lab ingest runs` |
| Show a specific ingest run | `rag-lab ingest show <run_id>` |
| Roll back a failed run manually | `rag-lab ingest rollback <run_id>` |
| Retry a specific failed run | `rag-lab ingest retry <run_id>` |
| Health check with test query | `python -m rag_lab.doctor --query "What is SDMX?"` |
| Run specific checks only | `python -m rag_lab.doctor --checks config,docstore,chromadb` |
| Store consistency report | `python -m rag_lab.maintenance.reconcile` |
| Store consistency (CI mode) | `python -m rag_lab.maintenance.reconcile --check` |
| Remove ChromaDB orphans | `python -m rag_lab.maintenance.reconcile --repair` |
| Save reconcile report | `python -m rag_lab.maintenance.reconcile --report-json out.json` |
| Full system diagnostic | `python -m rag_lab.maintenance.diagnose` |
| Diagnostic with test query | `python -m rag_lab.maintenance.diagnose --query "What is SDMX?"` |
| Diagnostic with signal breakdown | `python -m rag_lab.maintenance.diagnose --query "..." --explain` |
| Diagnostic with filter | `python -m rag_lab.maintenance.diagnose --query "..." --tag glossary --explain` |
| Benchmark regression check | `python -m rag_lab.benchmark.compare --baseline data/benchmark_v1_1_mmr_20260521.json --current data/benchmark_latest.json` |
| Run benchmark | `python -m rag_lab.benchmark --variants hybrid hybrid_mmr --output data/benchmark_latest.json` |
| Ingest all documents | `python -m rag_lab.cli ingest` |
| Backfill sparse BLOBs | `python -m rag_lab.maintenance.backfill_sparse` |
| Migrate to schema v2 | `python -m rag_lab.maintenance.migrate_to_v2` |
| Migrate to schema v3 (metadata) | `python -m rag_lab.maintenance.migrate_to_v3` |
| List documents | `python -m rag_lab.cli docs list` |
| List documents by tag | `python -m rag_lab.cli docs list --tag glossary` |
| Show document details | `python -m rag_lab.cli docs show SDMX_Glossary` |
| Tag a document | `python -m rag_lab.cli docs tag SDMX_Glossary glossary` |
| Untag a document | `python -m rag_lab.cli docs untag SDMX_Glossary glossary` |
| Delete document (all stores) | `python -m rag_lab.cli docs delete SDMX_Glossary` |
| List all tags | `python -m rag_lab.cli tags list` |
| Rename a tag | `python -m rag_lab.cli tags rename old-name new-name` |

---

## Installation

Install the `rag-lab` CLI wrapper with editable mode:

```bash
pip install -e .
which rag-lab      # → /path/to/envs/rag-lab/bin/rag-lab
rag-lab --help
```

After this, `rag-lab ingest`, `rag-lab docs`, `rag-lab tags`, etc. all work from PATH.

---

## Doctor command

`python -m rag_lab.doctor` runs 8 sequential health checks and exits with a clear status.

### Checks

| Check | What it verifies |
|-------|-----------------|
| `config` | Required config constants exist and have valid values |
| `docstore` | SQLite DocStore opens and contains chunks |
| `chromadb` | ChromaDB collection is reachable and non-empty |
| `fts5` | FTS5 index is in sync — uses real ID comparison, not COUNT(*). Reports missing/orphan chunks, not cosmetic counter inflation. |
| `sparse_coverage` | Fraction of chunks with sparse BLOBs meets `SPARSE_COVERAGE_THRESHOLD` |
| `reconcile` | Cross-store consistency (DocStore vs ChromaDB); calls reconcile internally |
| `ingest_health` | No stale IN_PROGRESS runs (>30 min) or FAILED runs without rollback |
| `test_query` | End-to-end retrieval returns at least one result for the test query. Falls back to CPU if GPU is OOM (WARN, not FAIL). |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All checks OK |
| `1` | At least one WARN, no FAIL |
| `2` | At least one FAIL |

### Options

```
--checks NAME[,NAME,...]   Run only the specified checks (comma-separated)
--query TEXT               Query used for test_query check (default: "What is SDMX?")
```

### Example output

```
───────────────────────────────────────────────────
RAG-Lab Doctor
───────────────────────────────────────────────────
  ✓ config               OK  — dim=1024, model_ver=2024-09, top_k=50
  ✓ docstore             OK  — 610 chunks
  ✓ chromadb             OK  — 610 vectors
  ✓ fts5                 OK  — 610/610 chunks indexed
  ✓ sparse_coverage      OK  — 610/610 (100%)
  ✓ reconcile            OK  — DocStore=610, ChromaDB=610
  ✓ test_query           OK  — 3 results — top: SDMX_Glossary rrf=0.0412

───────────────────────────────────────────────────
  ✓ Overall: OK
───────────────────────────────────────────────────
```

---

## Reconcile command

`python -m rag_lab.maintenance.reconcile` checks consistency between DocStore and ChromaDB.

### Modes

| Flag | Behaviour |
|------|-----------|
| _(none)_ | Print report, exit 0 if consistent, exit 1 if issues |
| `--check` | CI mode — same as default (explicit) |
| `--repair` | Remove orphaned entries from ChromaDB (destructive) |
| `--fix` | Alias for `--repair` (backward compatibility) |
| `--report-json PATH` | Save full JSON report to PATH |

### Extended checks (v1.2+)

Beyond orphan detection, reconcile now reports:

- **Duplicate chunk IDs** — should never occur (PK constraint), but reported if found
- **Model version mismatches** — chunks ingested with an older `embedding_model_version`
- **Embedding dim mismatches** — chunks with a different `embedding_dim` than current config
- **Sparse format version mismatches** — chunks with a stale `sparse_format_version`

Any of these conditions causes exit code 1 and is flagged in the report.

### JSON report format

```json
{
  "docstore_count": 610,
  "chroma_count": 610,
  "fts_count": 610,
  "sparse_blob_count": 610,
  "chroma_orphans": [],
  "missing_from_chroma": [],
  "duplicate_chunk_ids": [],
  "model_version_mismatches": [],
  "embedding_dim_mismatches": [],
  "sparse_format_version_mismatches": [],
  "repaired": false
}
```

### Recovery actions

| Issue | Command |
|-------|---------|
| ChromaDB orphans | `python -m rag_lab.maintenance.reconcile --repair` |
| Missing from ChromaDB | `python -m rag_lab.cli ingest --force` |
| FTS5 incomplete | `python -m rag_lab.maintenance.migrate_to_v2` |
| Sparse BLOBs missing | `python -m rag_lab.maintenance.backfill_sparse` |
| Model version mismatch | Re-ingest affected documents |

---

## Diagnose command

`python -m rag_lab.maintenance.diagnose` gives a detailed view of store counts and coverage.

### Options

```
--query TEXT     Run a test retrieval query
--explain        Show per-signal rank breakdown for each result (requires --query)
```

### Explain mode

With `--explain`, each result shows:

```
  [1] SDMX_Glossary | lines 142-160
       rrf=0.0412  dense=0.0231  bm25=12.50  sparse=0.0871
       in_dense=True  in_bm25=True  in_sparse=True
       ┌─ signal ranks: dense[rank=2]  bm25[rank=1]  sparse[rank=3]
       │  rrf_rank=1  chunk_id=abc123de…
       └─ mmr_score=0.7231 ← MMR reordered
```

Reranked chunks also carry `heading_path_used: bool` indicating whether structural
context was prepended to the text sent to the cross-encoder (v1.10+).

This is the primary tool for diagnosing why a specific chunk appears or doesn't appear in results.

---

## Query expansion variants (v1.11+)

`process_query()` can generate additional query variants to broaden the candidate pool.
Both variants are **disabled by default** — A/B testing showed zero retrieval quality benefit
at 2-2.4× per-query latency cost.

### Config flags

```python
QUERY_VARIANT_STOPWORD_ENABLED: bool = False   # key-terms only (stop-words removed)
QUERY_VARIANT_LAST_TERMS_ENABLED: bool = False  # last 5 key terms of the query
```

Set in `rag_lab/config.py` or `.env`.

### When to enable

Re-enable only if empirical A/B testing on your specific query distribution shows
a measurable improvement. Each variant roughly doubles or triples embedding + search cost.

### Variant types

| Variant | Type tag | Example input | Example output |
|---------|----------|--------------|----------------|
| Stopword | `variant_stopword` | "What is the role of SDMX?" | "role sdmx" |
| Tail terms | `variant_last_terms` | "What are the specs for DSD key families?" | "specs dsd key families" |

### Legacy note

Before v1.11, this was controlled by `VARIANTS_COUNT = 2`. That parameter has been removed.

---

## HyDE (Hypothetical Document Embeddings) — v1.12+, opt-in

HyDE generates a short hypothetical answer via LLM, encodes it with BGE-M3, and uses
that embedding as the dense retrieval query. The theory: hypothetical vocabulary is closer
to the target documents than the bare question.

**Status:** Disabled by default. A/B benchmark (65 queries, 2026-05-22) shows net negative
on this corpus: R@5 −3.8pp, nDCG@10 −1.9pp, latency ×12.5. See `docs/BENCHMARKS.md`.

### Config flags

```python
HYDE_ENABLED: bool = False        # main on/off switch
HYDE_MAX_TOKENS: int = 300        # token budget for the hypothetical answer
HYDE_TEMPERATURE: float = 0.1     # low temperature → factual density
HYDE_FORCE_NO_THINKING: bool = True  # skip 4× multiplier (thinking suppressed)
HYDE_TIMEOUT_SECONDS: int = 15    # timeout for LLM call; 0 = no timeout
HYDE_USE_FOR_DENSE: bool = True   # hypothetical → dense retrieval
HYDE_USE_FOR_BM25: bool = False   # original text → BM25 (not generated)
HYDE_USE_FOR_SPARSE: bool = False  # original sparse weights preserved
```

### Fallback behaviour

If the LLM call fails (server down, timeout, empty response), HyDE silently falls back
to the original query. Search quality is the same as without HyDE. No query fails.

### Benchmark experiment

```bash
python -m rag_lab.benchmark --suite official --variants full_hyde \
    --top-k 50 --rrf-k 20 --output /tmp/hyde_experiment.json
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current /tmp/hyde_experiment.json --variant full_hyde
```

### Dependency

Requires a live LLM server. `full_hyde` is NOT included in default benchmark runs.

---

## Query rewriting (LLM-based) — opt-in

Rewrites the user's question before processing to expand acronyms and add domain
terminology. Unlike HyDE, rewriting replaces the original query (not additive).

**Status:** Disabled by default. Not yet benchmarked on official suite.

### Config flags

```python
QUERY_REWRITING_ENABLED: bool = False
QUERY_REWRITING_MAX_TOKENS: int = 200
QUERY_REWRITING_TEMPERATURE: float = 0.0   # deterministic
QUERY_REWRITING_TIMEOUT_SECONDS: int = 10
```

### CLI

```bash
rag-lab query "What is DSD?" --rewrite
```

---

## Reranker heading context (v1.10+)

The cross-encoder receives enriched text with structural context:

```
Document: SDMX_Technical_Notes
Section: ## 4. Data Structure Definition > ### 4.2 Key Families

<chunk text>
```

### Config flag

`RERANKER_USE_HEADING_CONTEXT = True` (default: on)

Set to `False` in `.env` or `rag_lab/config.py` to restore v1.9 text-only behaviour.

### When to disable

- If you observe regressions on multilingual (ES/EN) queries in your domain.
- During A/B testing when you need a clean text-only reranker baseline.

### Diagnose field

Each chunk returned by `rerank()` carries `heading_path_used: bool`. Query with
`--explain` to see this per-result.

---

## Benchmark regression guard

`python -m rag_lab.benchmark.compare` compares a current benchmark run against a saved baseline.

### Usage

```bash
python -m rag_lab.benchmark.compare \
    --baseline data/benchmark_v1_1_mmr_20260521.json \
    --current  data/benchmark_latest.json \
    --variant  hybrid_mmr \
    --output   data/regression_report.json
```

### Default thresholds

| Metric | Threshold | Severity |
|--------|-----------|----------|
| R@5 | drop > 2 pp | FAIL |
| nDCG@10 | drop > 2 pp | FAIL |
| MRR | drop > 3 pp | FAIL |
| P95 latency | increase > 25% (relative) | WARN |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | No regressions |
| `1` | At least one WARN |
| `2` | At least one FAIL |

### Canonical baseline (activo: v1.11)

**`data/baselines/v1.11_official_full_eval.json`** — baseline activo desde v1.11.

65 queries (suite `official`), variante `full`, `top_k=50`, `rrf_k=20`,
`RERANKER_USE_HEADING_CONTEXT=True`, `QUERY_VARIANT_STOPWORD_ENABLED=False`,
`QUERY_VARIANT_LAST_TERMS_ENABLED=False`.

| Metric  | Value  |
|---------|--------|
| R@5     | 0.8205 |
| R@10    | 0.8962 |
| MRR     | 0.9385 |
| nDCG@10 | 0.8373 |

Métricas idénticas a v1.10 (Δ+0.0000). v1.11 reduce la latencia de candidate generation
~2× al eliminar variantes de query con beneficio nulo (A/B evidencia sobre 65 queries).

Comando estándar de regression guard:

```bash
python -m rag_lab.benchmark --suite official --variants full --output /tmp/current.json
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current  /tmp/current.json
```

**Regresión conocida (heredada de v1.10):** q070 (`cross_lingual_es_en`) MRR 1.000→0.500.
Pre-reranker MRR=1.000 — es efecto puro del cross-encoder con heading context en español.
Documentada en `v1.11_official_full_eval.json` meta → `known_regressions`.

**Baseline anterior:** `data/baselines/v1.10_official_full_eval.json` (histórico, activo v1.10).
**Baseline histórico:** `data/baselines/v1.8.1_official_full_eval.json` (v1.9 y anterior).

---

## Markdown quality gate (v1.6+)

Every ingest run validates the source document against the canonical Markdown contract before opening any transaction.

### Validation checks

| Code | Severity | Description |
|------|----------|-------------|
| `encoding_error` | ERROR | File is not valid UTF-8 |
| `empty_file` | ERROR | File is empty or whitespace-only |
| `frontmatter_invalid_yaml` | ERROR | YAML frontmatter fails to parse |
| `frontmatter_unclosed` | WARN | `---` block opened but never closed |
| `min_content` | WARN | Content below `min_content_tokens` threshold (default 50) |
| `missing_title` | WARN | No H1 heading found |
| `heading_hierarchy_skip` | WARN | Heading levels skip (e.g. H1→H3) |
| `section_too_long` | WARN | Section exceeds `max_section_tokens` (default 1600) |
| `large_table` | WARN | Table has more than `max_table_rows` rows (default 200) |
| `estimated_chunks_high` | WARN | Document will exceed `max_estimated_chunks` (default 200) |
| `long_line` | INFO | Line length exceeds `max_line_length` (default 500 chars) |

### Blocking behaviour

| Mode | Blocks on |
|------|-----------|
| Normal (default) | ERROR only |
| `--strict` | ERROR + WARN |

On block: no stores are written and no `IngestTransaction` is opened. Exit 0 is still returned (the run is just skipped).

### CLI tools

```bash
# Validate before ingesting
rag-lab docs validate path/to/doc.md
rag-lab docs validate --strict path/to/doc.md  # treats warnings as errors

# Inspect document structure (headings, tokens, chunks estimate, issues)
rag-lab docs inspect path/to/doc.md

# Preview chunking result without writing to stores
rag-lab docs preview-chunks path/to/doc.md
rag-lab docs preview-chunks path/to/doc.md --limit 10  # first 10 chunks only
```

### Ingest with strict validation

```bash
# Block on any warning (useful in CI / before bulk ingests)
rag-lab ingest --strict --doc path/to/doc.md
rag-lab ingest --strict  # validate all SOURCES
```

---

## Ingest transactions (v1.4+)

Every document ingest is now wrapped in a logical transaction that tracks
progress and performs rollback compensation on failure.

### Status transitions

```
IN_PROGRESS → COMMITTED     (success)
IN_PROGRESS → FAILED        (exception raised mid-ingest)
FAILED      → ROLLED_BACK   (after compensation: delete from ChromaDB + DocStore + FTS5 + documents)
```

### Checking ingest history

```bash
rag-lab ingest runs                      # last 20 runs
rag-lab ingest runs --status FAILED      # only failed
rag-lab ingest runs --doc SDMX_Glossary  # runs for one doc
rag-lab ingest show abc123def456         # full details
```

### Recovering from failures

After a crash or partial ingest:

```bash
# Automatic: roll back stale IN_PROGRESS + FAILED, then re-ingest
rag-lab ingest --resume          # stale IN_PROGRESS only
rag-lab ingest --retry-failed    # FAILED only

# Manual: roll back a specific run, then re-ingest separately
rag-lab ingest rollback abc123def456
rag-lab ingest --doc path/to/doc.md --force
```

`rollback` is **idempotent** — safe to run multiple times.

### What rollback does

1. Delete from ChromaDB: `VectorStore.delete_by_doc_id(doc_id)`
2. Delete from SQLite chunks + FTS5 + documents table
3. Mark run status as `ROLLED_BACK`

### Stale IN_PROGRESS detection

Runs that have been `IN_PROGRESS` for more than 30 minutes are considered
stale (process likely crashed). Both `reconcile` and `doctor ingest_health`
detect and report them.

---

## Routine maintenance checklist

Run after any bulk ingest or schema change:

1. `python -m rag_lab.doctor` — quick health gate
2. `python -m rag_lab.maintenance.reconcile` — cross-store consistency
3. `python -m rag_lab.benchmark.compare --baseline data/benchmark_v1_1_mmr_20260521.json --current <new_run>` — regression guard
4. `pytest tests/ -v` — full test suite

---

## Troubleshooting

### "Sparse scoring disabled: coverage X% < threshold 95%"

Sparse BLOBs are below the configured threshold. Run:

```bash
python -m rag_lab.maintenance.backfill_sparse
```

### "FTS5 table is empty — run migrate_to_v2"

The FTS5 virtual table was not created or is empty. Run:

```bash
python -m rag_lab.maintenance.migrate_to_v2
```

### ChromaDB orphans after integration tests

Some integration tests insert temporary chunks that may not be cleaned up. Run:

```bash
python -m rag_lab.maintenance.reconcile --repair
```

### Stale model version / embedding dim mismatches

Chunks were ingested with an older config. Re-ingest the affected documents:

```bash
python -m rag_lab.cli ingest --doc path/to/document.md --force
```

---

## Document metadata and tags (v1.3+)

### Schema

v1.3 adds five metadata tables to docstore.sqlite (same file as chunks):

| Table | Purpose |
|-------|---------|
| `documents` | One row per doc: path, content_hash, source_id, status, timestamps |
| `tags` | Normalized tag names (tag_id, name UNIQUE) |
| `document_tags` | Many-to-many join, ON DELETE CASCADE |
| `sources` | Optional source catalogue |

### Initial migration

After upgrading to v1.3, run once to populate the documents table from chunks
and migrate any tags from the legacy doc_manager.db:

```bash
python -m rag_lab.maintenance.migrate_to_v3
```

Safe to re-run — fully idempotent.

### Tagging documents

```bash
rag-lab docs tag SDMX_Glossary glossary
rag-lab docs tag SDMX_Glossary sdmx-core
rag-lab docs tag SDMX_2-1_User_Guide_6 user-guide
rag-lab tags rename sdmx-core sdmx
```

### Filtering retrieval by tags

Tags are resolved to doc_ids before any retrieval call. The ranking pipeline
(RRF, MMR, etc.) is unchanged — tags only restrict the candidate pool.

Via diagnose (for debugging):
```bash
python -m rag_lab.maintenance.diagnose --query "code list" --tag glossary
python -m rag_lab.maintenance.diagnose --query "REST API" --exclude-tag test --explain
```

Via Python (`hybrid_search`):
```python
from rag_lab.retrieval.filters import FilterSpec
results = hybrid_search(
    query, vs, ds, fts,
    query_dense=emb,
    query_sparse=sparse,
    filter_spec=FilterSpec(tags_include=["glossary"]),
)
```

### tag include logic (AND)

`tags_include=["glossary", "sdmx-core"]` returns only documents that have
**all** listed tags. Use multiple `--tag` flags in diagnose for AND logic.

### Deleting a document consistently

```bash
rag-lab docs delete SDMX_Glossary_Test
```

Removes from: SQLite chunks + FTS5 + documents table + ChromaDB vectors.

### Reconcile metadata checks

After v3 migration, reconcile also checks:
- Documents in documents table with no chunks (possible after manual deletion)
- Chunk doc_ids with no documents row (migration not run yet)
- document_tags pointing to deleted documents (should be zero due to CASCADE)

```bash
python -m rag_lab.maintenance.reconcile
```

---

## Test isolation

Unit tests always use `tmp_path` fixtures and never touch production stores.

Integration tests in `tests/integration/` fall into two categories:

1. **Read-only regression tests** (`test_benchmarks.py`): Access production stores
   to validate retrieval quality on the live corpus. Protected by
   `guard_read_only_integration` fixture which raises `AssertionError` on any write.

2. **Full-pipeline isolation tests** (`test_full_pipeline.py`): Write to stores
   located in `tmp_path` by patching both `rag_lab.config` AND the module-level
   bindings in `docstore.py` / `vector_store.py`. After the test, bindings are
   restored to production paths.

To run only integration tests:
```bash
pytest tests/integration/ -v -m integration
```

To run only unit tests (skip integration):
```bash
pytest tests/ -v -m "not integration"
```

---

## Operational audit checklist

Before starting a new version branch, verify:

```bash
# 1. Tests
pytest tests/ -q

# 2. Store consistency
python -m rag_lab.maintenance.reconcile --check

# 3. System health
python -m rag_lab.doctor

# 4. Benchmark regression
python -m rag_lab.benchmark.compare \
  --baseline data/benchmark_v1_1_mmr_20260521.json \
  --current data/benchmark_results_latest.json \
  --variant hybrid

# 5. Scope guard (no tabular/dataset references)
rg -n -i "dataset|csv|parquet|duckdb|forecast|automl" rag_lab tests

# 6. Validate production documents
rag-lab docs validate docs/SDMX_Glossary.md
rag-lab docs validate docs/SDMX_2-1_User_Guide_6.md
```

All commands must exit 0 (or WARN-only for doctor with justified reason).
