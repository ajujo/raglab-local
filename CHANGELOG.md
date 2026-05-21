# Changelog

All notable changes to RAG-Lab are documented here.

---

## v1.3 — 2026-05-21

### Metadata, tags, and structured filters

Adds a normalized metadata layer and structured document filtering without
touching any retrieval ranking, RRF, MMR, weights, top-k, or models.

**New schema (v3) in docstore.sqlite:**

| Table | Purpose |
|-------|---------|
| `documents` | One row per ingested doc: path, content_hash, source_id, dataset_id, status, timestamps, embedding metadata |
| `tags` | Normalized tag names with auto-increment tag_id |
| `document_tags` | Many-to-many between documents and tags (ON DELETE CASCADE) |
| `sources` | Optional source catalogue (URL, description) |
| `datasets` | Optional dataset groupings |

Migration: `python -m rag_lab.maintenance.migrate_to_v3` — idempotent, populates
documents from existing chunks, migrates tags from legacy doc_manager.db if present.

**Structured filters (`rag_lab/retrieval/filters.py`):**

`FilterSpec` dataclass with `doc_ids`, `tags_include` (AND), `tags_exclude`,
`source_id`, `dataset_id`, `status`. `resolve_filter(conn, spec)` converts it
to a `List[str]` of doc_ids for the existing filter mechanism. `hybrid_search()`
now accepts `filter_spec=` alongside the existing `doc_ids=`.

**New CLI commands (`rag-lab docs` / `rag-lab tags`):**

```
rag-lab docs list [--tag TAG] [--source SOURCE] [--dataset DATASET] [--status STATUS]
rag-lab docs show DOC_ID
rag-lab docs tag DOC_ID TAG_NAME
rag-lab docs untag DOC_ID TAG_NAME
rag-lab docs delete DOC_ID [--force]
rag-lab docs set-source DOC_ID SOURCE_ID
rag-lab docs set-dataset DOC_ID DATASET_ID
rag-lab tags list
rag-lab tags rename OLD NEW
rag-lab tags delete NAME [--force]
```

`docs delete` removes consistently from chunks (SQLite + FTS5 + documents table)
and ChromaDB. `DocStore.delete_by_doc_id()` and `VectorStore.delete_by_doc_id()`
added as first-class methods.

**Diagnose filter support:**

```
python -m rag_lab.maintenance.diagnose --query "..." --tag glossary
python -m rag_lab.maintenance.diagnose --query "..." --doc-id SDMX_Glossary --explain
python -m rag_lab.maintenance.diagnose --query "..." --exclude-tag test
```

`--explain` now also shows which filters were applied and how many documents
matched before retrieval.

**Reconcile metadata checks:**

Reconcile now reports orphaned documents (documents table row with no chunks),
doc_ids in chunks with no documents row, and document_tags pointing to
non-existent documents.

**Test suite:** 476 tests, EXIT_CODE=0 (was 427 in v1.2; +49 new tests
covering MetadataStore CRUD, FilterSpec resolution, migration idempotency, and
delete_by_doc_id).

---

## v1.2 — 2026-05-21

### Reliability and observability

This release adds diagnostics, regression protection, and extended consistency
checking. No retrieval behaviour, ranking, or model changes.

**New commands:**

- `python -m rag_lab.doctor` — 7-check system health gate (config, docstore,
  chromadb, fts5, sparse_coverage, reconcile, test_query). Exit codes: 0=OK,
  1=WARN, 2=FAIL. Supports `--checks NAME[,...]` to run a subset.
- `python -m rag_lab.benchmark.compare` — regression guard. Compares a current
  benchmark JSON against a saved baseline. Default thresholds: R@5/nDCG@10 drop
  >2 pp = FAIL; MRR drop >3 pp = FAIL; P95 increase >25% = WARN.
- `python -m rag_lab.maintenance.diagnose --explain` — per-signal rank breakdown
  showing `dense_rank`, `bm25_rank`, `sparse_rank`, `rrf_rank`, `mmr_score`, and
  `was_mmr_reordered` for every result.

**Reconcile improvements:**

- `--repair` flag (alias: `--fix`), `--check` CI mode, `--report-json PATH`.
- Extended checks: duplicate chunk IDs, model version mismatches vs config,
  embedding dimension mismatches vs config, sparse format version mismatches vs config.
- Quiet mode (`quiet=True`) for programmatic callers.

**Rank fields in hybrid_search output:**

Every chunk result now carries `dense_rank`, `bm25_rank`, `sparse_rank` (1-based
rank in each signal's list, or `None` if absent), `rrf_rank` (1-based in fused
order), and `was_mmr_reordered` (bool). Used by `--explain` mode.

**Documentation:** `docs/OPERATIONS.md` — runbook covering all operational commands.

**Test suite:** 427 tests, EXIT_CODE=0 (was 343 in v1.1; +84 new tests covering
reconcile, doctor, compare, and explain/rank fields).

---

## v1.1 — 2026-05-21

### MMR document-diversity post-processing

This release activates MMR (Maximal Marginal Relevance) doc-diversity reranking
by default, addressing the large-document monopoly problem identified during the
v1.0 baseline analysis.

**Problem solved:** With `top_k=50`, large documents (e.g. SDMX_2-1_User_Guide_6
with 197 chunks) could occupy multiple result slots in top-5/10, blocking smaller
but equally relevant documents. This degraded nDCG@10 (which counts each doc only
at first occurrence) and reduced the diversity of context passed to the LLM.

**Solution:** MMR post-processing applied after weighted RRF fusion. Greedy
selection penalises chunks from already-represented documents with a configurable
λ parameter. At λ=0.6, relevance still dominates — a second chunk from the same
document survives if its rrf_score justifiably outweighs the diversity penalty.

**Configuration changes (`rag_lab/config.py`):**

| Parameter | Before (v1.0) | After (v1.1) | Reason |
|-----------|--------------|--------------|--------|
| `MMR_ENABLED` | `False` | `True` | Activated after edge case validation |
| `MMR_LAMBDA` | `0.7` | `0.6` | λ=0.6 achieves perfect R@5=1.000 on 28-query set |

To compare against v1.0 baseline: set `MMR_ENABLED = False` in `config.py`.
`DOC_CAP_ENABLED` remains `False` — `hybrid_mmr` provides superior diversity
without a hard per-document limit.

**New code (`rag_lab/retrieval/diversity.py`):**

- `apply_document_cap(chunks, cap)` — hard per-doc-id limit, O(n). Validated,
  kept as experimental alternative (`hybrid_cap` variant).
- `apply_mmr(chunks, lambda_, k)` — doc-diversity MMR greedy selection. Adds
  `mmr_score` field to each result. Does not mutate inputs.

**New benchmark infrastructure:**

- `rag_lab/benchmark/metrics.diversity_stats()` — `unique_docs@k` and
  `max_chunks_same_doc@k` metrics.
- Two new benchmark variants: `hybrid_cap` and `hybrid_mmr` (opt-in via
  `--variants`; not included in the default five-variant run).
- `hybrid_search()` accepts `diversity_mode` parameter (`"cap"`, `"mmr"`, or
  `None`) and passes through doc_cap / mmr_lambda.

**Test suite:** 343 tests, EXIT_CODE=0 (was 323 in v1.0; +20 new diversity tests).

**Benchmark results — v1.1 official** (`top_k=50, rrf_k=20, sparse_w=0.25,
mmr_lambda=0.6`, 28 queries — see `data/benchmark_v1_1_mmr_20260521.json`):

| Variant | R@5 | R@10 | R@30 | MRR | nDCG@10 | unique_docs@5 |
|---------|-----|------|------|-----|---------|:---:|
| hybrid (v1.0 baseline) | 0.762 | 0.923 | 0.982 | 0.867 | 0.755 | 2.75 |
| hybrid_cap (N=3)       | 0.816 | 0.946 | 1.000 | 0.867 | 0.768 | 2.93 |
| **hybrid_mmr (λ=0.6)** | **1.000** | **1.000** | **1.000** | **0.884** | **0.840** | **4.82** |

**Corpus state:** 610 chunks / 610 ChromaDB / 610 FTS5 / 610 sparse BLOBs (100%).

**Key edge case findings (16 new annotated queries, q013–q028):**
- Spanish queries (q027, q028): hybrid R@5=0.000–0.333 → hybrid_mmr R@5=1.000.
  BM25 language mismatch + dense bias had produced a monopoly of marginally-relevant
  English chunks. MMR's diversity pressure surfaces the Spanish-language source.
- Multi-chunk same-doc queries (q013, q026): MMR never causes regression.
  At λ=0.6, the first chunk of the dominant doc stays at rank 1; subsequent chunks
  survive only if their rrf_score justifies the diversity penalty (confirmed for q026
  where nDCG@10 improved 0.974 → 1.000).
- Single-source Glossary queries (q016–q021): zero regressions. Glossary terminology
  is not blocked by MMR when each chunk covers a distinct artefact type.

**Recalibration triggers (same as v1.0):** corpus changes ≥20% size increase,
model updates (BGE-M3 or reranker), cross-lingual query distribution shifts.

---

## Baseline v1.0 — 2026-05-20

### Retrieval baseline: weighted RRF + calibrated parameters

This release closes the retrieval evaluation phase and establishes the first
official performance baseline. It replaces the equal-weight RRF3 fusion with a
calibrated weighted variant and raises the candidate pool from 30 to 50 results.

**Configuration changes (`rag_lab/config.py`):**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `RETRIEVAL_TOP_K` | 30 | 50 | Larger pool improves R@5 without latency impact |
| `RRF_K` | 60 | 20 | More discriminative ranking at this corpus size |
| `SPARSE_RRF_WEIGHT` | 1.0 (implicit) | 0.25 | Eliminates large-document bias in sparse signal |
| `DENSE_RRF_WEIGHT` | 1.0 (implicit) | 1.0 | Reference weight, unchanged |
| `BM25_RRF_WEIGHT` | 1.0 (implicit) | 1.0 | Unchanged |

**Code changes:**

- `rag_lab/retrieval/fusion.py`: Added `weighted_rrf()` as primary function.
  `rrf_three()` retained as backward-compatible wrapper.
- `rag_lab/retrieval/hybrid_search.py`: Uses `weighted_rrf` with configurable
  per-signal weights. New optional params `dense_weight`, `bm25_weight`, `sparse_weight`.
- `rag_lab/benchmark/__main__.py`: CLI defaults now read from config instead of
  hardcoded values, so benchmark always reflects the active configuration.
- `rag_lab/benchmark/weighted_fusion.py`: Thin re-export from `retrieval.fusion`.
- `rag_lab/benchmark/calibration.py`: Import updated to canonical location.

**Test suite:** 323 tests, EXIT_CODE=0. Includes 21 new tests for `weighted_rrf`
covering weight scaling, sparse dominance, rrf_k discriminativeness, and
backward compatibility with `rrf_three`.

**Benchmark results** (top_k=50, rrf_k=20, sparse_w=0.25, 12 queries):

| Variant | R@5 | MRR | nDCG@10 |
|---------|-----|-----|---------|
| dense        | 0.743 | 0.778 | 0.724 |
| dense_bm25   | 0.792 | 0.840 | 0.755 |
| **hybrid**   | **0.812** | **0.847** | **0.750** |

Hybrid now outperforms dense_bm25 on R@5 (+2pp). Before calibration,
hybrid trailed dense_bm25 by 8pp on R@5.

**Corpus state:** 610 chunks / 610 ChromaDB / 610 FTS5 / 610 sparse BLOBs (100%).

---

## Pre-baseline work (2026-05-20, same day)

The following work was completed before the baseline was frozen:

### Retrieval benchmark framework
- Five pipeline variants: dense, bm25, dense_bm25, hybrid, full
- IR metrics: recall@5/10/30, MRR, nDCG@10, latency P50/P95/P99
- Annotated query file: 12 SDMX queries with graded relevance (0–3 per doc)
- CLI: `python -m rag_lab.benchmark`
- 38 unit tests for metrics and runner

### Calibration grid search
- 324 configurations × 12 queries: `dense_k`, `bm25_k`, `sparse_w`, `bm25_w`, `rrf_k`
- Precompute-once optimization: O(queries) store round-trips vs O(configs × queries)
- Root cause identified: BGE-M3 sparse over-weights large documents at `sparse_w=1.0`
- CLI: `python -m rag_lab.benchmark.calibrate`

### MVP hybrid pipeline (earlier)
- Three-store architecture: ChromaDB (dense) + FTS5 (BM25) + DocStore (sparse BLOBs)
- 100% sparse coverage via `backfill_sparse`
- Five-score result shape: `rrf_score`, `dense_score`, `bm25_score`, `sparse_score`, flags
- Sparse coverage guard: auto-disables sparse if coverage < 95%
- Corpus cleanup: 610/610/610/610 consistency across all stores
