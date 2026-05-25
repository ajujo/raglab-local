# RAG-Lab Operations Guide

Operational reference for diagnosing, maintaining, and protecting the RAG-Lab system.

---

## Quick-reference commands

| Goal | Command |
|------|---------|
| Full health check | `rag-lab doctor` |
| Validate a document | `rag-lab docs validate path/to/doc.md` |
| Validate (strict — warnings block) | `rag-lab docs validate --strict path/to/doc.md` |
| Inspect document structure | `rag-lab docs inspect path/to/doc.md` |
| Preview chunks without ingesting | `rag-lab docs preview-chunks path/to/doc.md` |
| Ingest a document (transactional) | `rag-lab ingest --doc path/to/doc.md` |
| Ingest a directory of .md files | `rag-lab ingest --doc path/to/dir/` |
| Ingest with strict validation | `rag-lab ingest --strict --doc path/to/doc.md` |
| Ingest with parallel workers | `rag-lab ingest --workers 4` |
| Ingest all sources | `rag-lab ingest` |
| Resume the last incomplete batch | `rag-lab ingest --resume` |
| Retry all failed documents | `rag-lab ingest --retry-failed` |
| List batch history | `rag-lab ingest batches` |
| List ingest run history | `rag-lab ingest runs` |
| Show a specific ingest run | `rag-lab ingest show <run_id>` |
| Roll back a failed run manually | `rag-lab ingest rollback <run_id>` |
| Retry a specific failed run | `rag-lab ingest retry <run_id>` |
| Health check with test query | `rag-lab doctor --query "What is SDMX?"` |
| Run specific checks only | `rag-lab doctor --checks config,docstore,chromadb` |
| Store consistency report | `rag-lab reconcile` |
| Store consistency (CI mode) | `rag-lab reconcile --check` |
| Remove ChromaDB orphans | `rag-lab reconcile --repair` |
| Repair FTS5 duplicates | `rag-lab reconcile --repair-fts` |
| Save reconcile report | `rag-lab reconcile --report-json out.json` |
| Full system diagnostic | `rag-lab diagnose` |
| Diagnostic with test query | `rag-lab diagnose --query "What is SDMX?"` |
| Diagnostic with signal breakdown | `rag-lab diagnose --query "..." --explain` |
| Run benchmark | `rag-lab benchmark --suite official --variants full --no-cache` |
| Benchmark regression check | `python -m rag_lab.benchmark.compare --baseline data/benchmark_v1_1_mmr_20260521.json --current data/benchmark_latest.json` |
| Backfill sparse BLOBs | `python -m rag_lab.maintenance.backfill_sparse` |
| Migrate to schema v2 | `python -m rag_lab.maintenance.migrate_to_v2` |
| Migrate to schema v3 (metadata) | `python -m rag_lab.maintenance.migrate_to_v3` |
| List documents | `rag-lab docs list` |
| List documents by tag | `rag-lab docs list --tag glossary` |
| Show document details | `rag-lab docs show SDMX_Glossary` |
| Tag a document | `rag-lab docs tag SDMX_Glossary glossary` |
| Untag a document | `rag-lab docs untag SDMX_Glossary glossary` |
| Delete document (all stores) | `rag-lab docs delete SDMX_Glossary` |
| List all tags | `rag-lab tags list` |
| Rename a tag | `rag-lab tags rename old-name new-name` |
| Cache stats | `rag-lab cache stats` |
| Clear cache | `rag-lab cache clear` |
| Vacuum cache (remove expired) | `rag-lab cache vacuum` |
| Inspect cache entry | `rag-lab cache inspect <key>` |
| Benchmark without cache (default) | `rag-lab benchmark --suite official --variants full` |
| Benchmark with cache | `rag-lab benchmark --suite official --variants full --cache` |
| Add chunk feedback | `rag-lab feedback add --query "..." --chunk-id "..." --feedback relevant` |
| List feedback events | `rag-lab feedback list` |
| Feedback statistics | `rag-lab feedback stats` |
| Export feedback JSONL | `rag-lab feedback export --output path.jsonl` |
| Clear all feedback | `rag-lab feedback clear --yes` |

---

## Installation

Install the `rag-lab` CLI wrapper in editable mode:

```bash
pip install -e .
which rag-lab      # → /path/to/envs/rag-lab/bin/rag-lab
rag-lab --help
```

After this, `rag-lab ingest`, `rag-lab docs`, `rag-lab tags`, etc. all work from PATH.

---

## Doctor command

`rag-lab doctor` runs 8 sequential health checks and exits with a clear status.

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

`rag-lab reconcile` checks consistency between DocStore and ChromaDB.

### Modes

| Flag | Behaviour |
|------|-----------|
| _(none)_ | Print report, exit 0 if consistent, exit 1 if issues |
| `--check` | CI mode — same as default (explicit) |
| `--repair` | Remove orphaned entries from ChromaDB (destructive) |
| `--fix` | Alias for `--repair` (backward compatibility) |
| `--repair-fts` | Remove FTS5 duplicate rows (v1.16.1+) |
| `--repair-metadata` | Back-fill NULL embedding_model_name/version for pre-v2 chunks (v1.16.3+) |
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
| ChromaDB orphans | `rag-lab reconcile --repair` |
| FTS5 duplicates | `rag-lab reconcile --repair-fts` |
| NULL model metadata | `rag-lab reconcile --repair-metadata` |
| Missing from ChromaDB | `rag-lab ingest --force` |
| FTS5 incomplete | `python -m rag_lab.maintenance.migrate_to_v2` |
| Sparse BLOBs missing | `python -m rag_lab.maintenance.backfill_sparse` |
| Model version mismatch | Re-ingest affected documents |

---

## Diagnose command

`rag-lab diagnose` gives a detailed view of store counts and coverage.

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
at 2–2.4× per-query latency cost.

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

## HNSW vector store — v1.13+

ChromaDB uses HNSW (Hierarchical Navigable Small World) for dense search.

### Configurable parameters (`rag_lab/config.py`, section 6.5)

```python
VECTOR_HNSW_SPACE = "cosine"       # distance metric: "cosine", "l2", "ip"
VECTOR_HNSW_M = 16                 # connections per node (quality vs memory)
VECTOR_HNSW_CONSTRUCTION_EF = 100  # graph quality during indexing
VECTOR_HNSW_SEARCH_EF = 100        # candidate pool during search
```

### Important ChromaDB 1.x constraints

**All parameters are build-time.** There is no mutable query-time mechanism.
Changing `hnsw:space` on an existing collection raises an explicit error.
Calling `col.modify(metadata={"hnsw:search_ef": N})` persists the value in metadata
but **does not modify the in-memory hnswlib index** — the index continues using the
value it was built with.

### Source of truth: `configuration_json` vs `metadata`

ChromaDB 1.5+ exposes two sources of HNSW information:

| Source | What it represents |
|--------|-------------------|
| `col.configuration_json['hnsw']` | Actual index parameters as built (**authoritative**) |
| `col.metadata` | Optional annotations; may be stale after `modify()` |

`VectorStore.initialize()` uses `configuration_json` as the authoritative source for
mismatch detection. This avoids false warnings from vestigial metadata annotations
left by past experiments.

### Official production collection (accepted baseline)

```
collection: sdmx_rag
configuration_json.hnsw:
  space:           cosine
  max_neighbors:   16     (= hnsw:M)
  ef_construction: 100
  ef_search:       100
metadata (vestigial): {hnsw:search_ef: 500}  <- stale, no effect
```

The `hnsw:search_ef=500` in `metadata` is a leftover from an earlier experiment
(a `modify()` call). The actual index uses `ef_search=100`. The collection should
not be modified to correct this — it is harmless and produces no warnings.

### When parameters take effect

| Moment | Effect |
|--------|--------|
| First ingest (new collection) | Yes — applied when collection is created |
| Additional ingest (existing collection) | No — existing collection is reused |
| Config change without rebuild | No — mismatch WARNING is emitted |
| Rebuild (delete chroma_db + re-ingest) | Yes — new collection with new params |

### How to inspect the active configuration

```python
import chromadb
c = chromadb.PersistentClient("storage/chroma_db")
col = c.get_collection("sdmx_rag")
print(col.configuration_json['hnsw'])  # real index parameters (authoritative)
print(col.metadata)                    # annotations (may be stale)
```

Or with the doctor:
```bash
rag-lab doctor
```

### How to rebuild

```bash
rm -rf storage/chroma_db/
rag-lab ingest
```

### Mismatch warning

If the actual index parameters (`configuration_json`) differ from the values in
`config.py`, `VectorStore.initialize()` emits a WARNING with the differences and the
rebuild command. The collection is **never destroyed or modified** automatically.

Metadata annotations that differ from the config but do not reflect the actual index
parameters **do not produce a warning** (they are not real mismatches).

### Profile benchmark (2026-05-23, 610 chunks)

| Profile  |  M | ef_c | ef_s | p50 (ms) | recall vs prod |
|----------|----|------|------|----------|----------------|
| current  | 16 |  100 |  100 |     1.87 | 0.9547         |
| fast     |  8 |   64 |   50 |     1.87 | **0.8313** ❌  |
| balanced | 16 |  128 |  100 |     1.91 | 0.9553         |
| recall   | 32 |  200 |  200 |     2.09 | 0.9533         |

**Recommendation:** keep `current` (M=16). HNSW latency (~2ms) is negligible compared
to the reranker (~250ms). `fast` degrades recall. `balanced`/`recall` provide less than
0.001 improvement at 610 chunks. A real benefit from `recall` only emerges above ~10k chunks.

### Profile tool

```bash
python -m rag_lab.maintenance.hnsw_profiles
```

Creates temporary collections (without touching production), copies embeddings, and
measures latency and recall.

---

## HyDE (Hypothetical Document Embeddings) — v1.12+, opt-in

HyDE generates a short hypothetical answer via LLM, encodes it with BGE-M3, and uses
that embedding as the dense retrieval query. The theory: hypothetical vocabulary is closer
to the target documents than the bare question.

**Status:** Disabled by default. A/B benchmark (65 queries, 2026-05-22) shows net negative
on this corpus: R@5 −3.8pp, nDCG@10 −1.9pp, latency ×12.5. See `docs/BENCHMARKS.en.md`.

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

## Query cache — v1.14+

### What is cached and what is not

| Element | Cached? |
|---------|---------|
| `hybrid_search` + reranker results (chunk list with scores) | **Yes** |
| Final LLM response | **No** (intentional: LLM is stochastic) |
| Verification / trust score | **No** |
| Query embeddings | **No** (always recomputed; efficient) |

### Configuration

```python
QUERY_CACHE_ENABLED: bool = True          # global on/off switch
QUERY_CACHE_PATH = DATA_DIR / "query_cache.sqlite"
QUERY_CACHE_TTL_SECONDS: int = 604800     # 7 days; 0 = no TTL
```

### Cache key

The key includes: normalized query, filters, top_k, rrf_k, dense/bm25/sparse weights,
MMR, HyDE, reranker_heading_context, embedding model version, sparse format version,
and **corpus fingerprint** (`n_chunks:max_ingest_run_id:revision`).

The corpus fingerprint changes automatically when the corpus or its metadata changes.

### Automatic invalidation

The fingerprint has three components:

| Component | Changes when |
|-----------|-------------|
| `n_chunks` | A document is ingested or deleted |
| `max_ingest_run_id` | Any ingest/delete operation |
| `revision` | A tag is assigned, removed, renamed, or deleted; a document is deleted |

The `revision` counter lives in `MetadataStore.cache_revision` and is incremented on
every tag or document mutation. This ensures tag-filtered queries never return stale
results after tagging changes.

Old entries are not returned even if they exist physically (the fingerprint won't match).
`rag-lab cache vacuum` removes entries expired by TTL or with TTL set to 0.

### CLI commands

```bash
rag-lab cache stats            # stats (entries, hits, DB size)
rag-lab cache clear            # delete all entries
rag-lab cache vacuum           # remove expired entries + VACUUM SQLite
rag-lab cache inspect <key>    # inspect an entry by its key
```

### Temporary bypass

```bash
rag-lab query "..." --no-cache  # ignore the cache for this query
```

### Benchmark behaviour

The benchmark ignores the cache by default to measure real quality and latency:
```bash
rag-lab benchmark --suite official --variants full           # cache disabled (default)
rag-lab benchmark --suite official --variants full --cache   # cache enabled (measures benefit)
```

The cache **does not change quality metrics** (R@5, nDCG, MRR). It only reduces latency
on hits. Official comparisons against baseline must always be run without cache.

### Why LLM responses are not cached

LLM responses are stochastic (temperature > 0) and depend on the full system context.
Caching them would introduce potentially stale responses when corpus, config, or prompt
changes. Retrieval is the bottleneck to reduce; the LLM is called once per query anyway.

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

### Canonical baseline (active: v1.11)

**`data/baselines/v1.11_official_full_eval.json`** — active baseline since v1.11.

65 queries (suite `official`), variant `full`, `top_k=50`, `rrf_k=20`,
`RERANKER_USE_HEADING_CONTEXT=True`, `QUERY_VARIANT_STOPWORD_ENABLED=False`,
`QUERY_VARIANT_LAST_TERMS_ENABLED=False`.

| Metric  | Value  |
|---------|--------|
| R@5     | 0.8205 |
| R@10    | 0.8962 |
| MRR     | 0.9385 |
| nDCG@10 | 0.8373 |

Metrics are identical to v1.10 (Δ+0.0000). v1.11 reduces candidate generation latency
~2× by removing query variants that showed no benefit in A/B evidence over 65 queries.

Standard regression guard command:

```bash
rag-lab benchmark --suite official --variants full --output /tmp/current.json
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current  /tmp/current.json
```

**Known regression (inherited from v1.10):** q070 (`cross_lingual_es_en`) MRR 1.000→0.500.
Pre-reranker MRR=1.000 — this is a pure effect of the cross-encoder with heading context
on Spanish text. Documented in `v1.11_official_full_eval.json` meta → `known_regressions`.

**Previous baseline:** `data/baselines/v1.10_official_full_eval.json` (historical, active during v1.10).
**Historical baseline:** `data/baselines/v1.8.1_official_full_eval.json` (v1.9 and earlier).

---

## Markdown quality gate (v1.6+)

Every ingest run validates the source document against the canonical Markdown contract
before opening any transaction.

For full documentation of the frontmatter contract (fields, derived tags, filter
integration), see **[docs/FRONTMATTER.en.md](FRONTMATTER.en.md)**.

### Validation checks

#### General document checks

| Code | Severity | Description |
|------|----------|-------------|
| `encoding_error` | ERROR | File is not valid UTF-8 |
| `empty_file` | ERROR | File is empty or whitespace-only |
| `min_content` | WARN | Content below `min_content_tokens` threshold (default 50) |
| `missing_title` | WARN | No H1 heading found |
| `heading_hierarchy_skip` | WARN | Heading levels skip (e.g. H1→H3) |
| `section_too_long` | WARN | Section exceeds `max_section_tokens` (default 1600) |
| `large_table` | WARN | Table has more than `max_table_rows` rows (default 200) |
| `estimated_chunks_high` | WARN | Document will exceed `max_estimated_chunks` (default 200) |
| `long_line` | INFO | Line length exceeds `max_line_length` (default 500 chars) |

#### Frontmatter checks (v1.19+)

| Code | Severity | Description |
|------|----------|-------------|
| `frontmatter_missing` | WARN | No `---` frontmatter block found at start of file |
| `frontmatter_unclosed` | WARN | `---` block opened but never closed |
| `frontmatter_invalid_yaml` | ERROR | YAML frontmatter fails to parse |
| `frontmatter_not_mapping` | ERROR | Frontmatter parses but is not a key-value mapping |
| `frontmatter_scope_violation` | ERROR | Prohibited field present (`dataset` or `dataset_id`) |
| `frontmatter_missing_doc_id` | ERROR | `doc_id` field is absent — required for ingestion |
| `frontmatter_missing_title` | WARN | `title` field is absent (H1 will be used as fallback) |
| `frontmatter_missing_domain` | WARN | `domain` classification field is absent |
| `frontmatter_missing_source_type` | WARN | `source_type` classification field is absent |
| `frontmatter_missing_language` | WARN | `language` classification field is absent |
| `frontmatter_tags_not_list` | ERROR | `tags` field is present but is not a YAML list |
| `frontmatter_tag_not_string` | ERROR | A tag element is not a string |
| `frontmatter_tag_empty` | WARN | A tag element is an empty string |
| `frontmatter_tag_whitespace` | WARN | A tag element contains only whitespace |
| `frontmatter_tag_duplicate` | WARN | A tag element appears more than once |
| `yaml_unavailable` | INFO | `pyyaml` is not installed; frontmatter validation skipped |

### Blocking behaviour

| Mode | Blocks on |
|------|-----------|
| Normal (default) | ERROR only |
| `--strict` | ERROR + WARN |

On block: no stores are written and no `IngestTransaction` is opened. Exit 0 is still
returned (the run is skipped).

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

## Ingest pipeline (v1.16+)

The ingest command runs a 4-phase pipeline with a **single-writer guarantee**:
worker threads only compute (no DB access); all writes happen on the main thread.

### Phases

1. **Parallel preparation** (`--workers N`) — validate + clean + chunk each doc.
   Thread-safe: no SQLite or ChromaDB access. Default: `INGEST_MAX_WORKERS = 2`.
2. **SKIPPED detection** (main thread) — check `ingest_documents` for a previous
   COMMITTED row with the same content hash. Fallback: scan `data/ingested.jsonl`
   for pre-v1.16 ingests.
3. **Sequential embedding** — GPU/CPU model is not thread-safe; always on main thread.
4. **Sequential write** (`IngestTransaction` per document) — ChromaDB → SQLite chunks
   → FTS5 → metadata. Failure in any phase triggers compensation rollback for that doc
   only; previously committed docs are unaffected.

### Batch tracking (v1.16+)

Each CLI invocation creates one **batch** row (`ingest_batches`) and one
**document** row (`ingest_documents`) per file.

Document status lifecycle:
```
PENDING → VALIDATED → EMBEDDING → WRITING → COMMITTED
                ↓                       ↓
             FAILED               ROLLED_BACK
                                       ↑
                              SKIPPED (hash unchanged)
```

### Checking ingest history

```bash
rag-lab ingest batches                   # last 20 batches (high-level)
rag-lab ingest runs                      # last 20 runs (low-level, per-document)
rag-lab ingest runs --status FAILED      # only failed
rag-lab ingest runs --doc SDMX_Glossary  # runs for one doc
rag-lab ingest show abc123def456         # full details for a run
```

### Ingesting directories

```bash
rag-lab ingest --doc path/to/dir/       # all *.md files, sorted
rag-lab ingest --doc path/to/dir/ --workers 4
```

### Recovering from failures

```bash
# Resume the most recent incomplete batch (PENDING/FAILED docs only)
rag-lab ingest --resume

# Retry all FAILED/ROLLED_BACK documents (creates a new batch)
rag-lab ingest --retry-failed

# Force re-ingest even if content hash is unchanged
rag-lab ingest --doc path/to/doc.md --force

# Manual: roll back a specific run, then re-ingest
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

1. `rag-lab doctor` — quick health gate
2. `rag-lab reconcile --check` — cross-store consistency
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
rag-lab reconcile --repair
```

### Stale model version / embedding dim mismatches

Chunks were ingested with an older config. Re-ingest the affected documents:

```bash
rag-lab ingest --doc path/to/document.md --force
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
rag-lab diagnose --query "code list" --tag glossary
rag-lab diagnose --query "REST API" --exclude-tag test --explain
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

### Tag include logic (AND)

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
rag-lab reconcile --check
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
rag-lab reconcile --check

# 3. System health
rag-lab doctor

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

All commands must exit 0 (or WARN-only for doctor with a justified reason).

---

## Feedback — chunk-level signals — v1.15+

### What it is and what it is not

The feedback module captures relevance judgements per chunk. In v1.15 it is
**purely an observation log**: it does not alter ranking, scores, the cache,
or any index. Accumulated feedback is intended to feed re-ranking signals in v1.16.

### Supported feedback types

| Value | Meaning |
|-------|---------|
| `relevant` | Chunk is relevant to the query |
| `irrelevant` | Chunk is not relevant to the query |
| `useful` | The response based on this chunk was useful |
| `not_useful` | The response based on this chunk was not useful |
| `wrong_doc` | The chunk is from the wrong document |
| `outdated` | The chunk content is outdated |
| `duplicate` | This chunk repeats content from another retrieved chunk |
| `bad_citation` | The citation generated from this chunk is incorrect |

### CLI commands

```bash
# Record feedback for a specific chunk
rag-lab feedback add --query "What is SDMX?" --chunk-id "<chunk_id>" --feedback relevant
rag-lab feedback add --query "What is SDMX?" --chunk-id "<chunk_id>" --feedback irrelevant --reason wrong_doc
rag-lab feedback add --query "..." --chunk-id "..." --feedback bad_citation --note "citation on line 42 is wrong"

# List recent events
rag-lab feedback list                     # last 20
rag-lab feedback list --limit 50
rag-lab feedback list --feedback irrelevant
rag-lab feedback list --chunk-id "<id>"

# Statistics
rag-lab feedback stats

# Export for future evaluation
rag-lab feedback export                   # print JSONL to stdout
rag-lab feedback export --output data/feedback_export.jsonl

# Clear
rag-lab feedback clear --yes
```

### Query integration

After displaying the response, `rag-lab query` prints the rank, chunk_id, doc_id,
and score of each retrieved chunk, along with an example command:

```
── Retrieved chunks (for feedback) ──
   1. chunk=<id>…  doc=sdmx_glossary  score=0.912
   2. chunk=<id>…  doc=sdmx_user_guide  score=0.887
   ...

  To give feedback: rag-lab feedback add --query "..." --chunk-id "..." --feedback relevant
```

### Backend and schema

- **DB:** `storage/docstore.sqlite` (same database as corpus metadata)
- **Table:** `feedback_events`
- **Key fields:** `query_text`, `query_hash`, `chunk_id`, `doc_id`, `rank`,
  `feedback`, `rating`, `reason`, `source`, `pipeline_variant`, `cache_hit`,
  `cache_key`, `corpus_fingerprint`, `retrieval_config_hash`, `user_note`, `created_at`

The table is created automatically on the first `feedback add`.

### Traceability

- `query_hash` — SHA-256 of the normalized query. Allows grouping feedback for the
  same question regardless of case or extra whitespace.
- `corpus_fingerprint` — captures the state of the corpus at the time of feedback
  (`n_chunks:max_ingest_run_id:revision`). Useful for detecting whether feedback
  was collected against a different corpus than the current one.
- `retrieval_config_hash` — SHA-256 of retrieval parameters (top_k, rrf_k, weights,
  MMR, HNSW, etc.). Allows filtering out feedback taken with different configurations.

### Exporting as benchmark queries

The exported JSONL can be converted into curated benchmark queries:

```bash
# Export negative feedback for review
rag-lab feedback export | jq 'select(.feedback == "irrelevant" or .feedback == "wrong_doc")'
```

Recurring negative cases are candidates for new queries in
`data/benchmark_queries.yaml` with `expected_behavior: low_recall_expected` or similar.

### Isolation guarantees (v1.15)

- Reading or writing `feedback_events` does not invoke any retrieval code.
- `make_query_hash()` and `make_retrieval_config_hash()` do not read embeddings or indexes.
- `FeedbackStore.add()` does not call `MetadataStore.bump_revision()` —
  the `corpus_fingerprint` does not change with feedback operations.
- Explicit tests in `tests/test_feedback/test_feedback_events.py` verify that
  retrieval results are identical before and after adding feedback.

### Plan for v1.16

v1.16 will evaluate using accumulated feedback as an additional signal in re-ranking:
- Boost for chunks marked `relevant` for similar queries (query_hash match).
- Penalty for chunks marked `irrelevant` or `wrong_doc`.
- Feedback as a signal will be opt-in and benchmarked before being activated.
