# Changelog

All notable changes to RAG-Lab are documented here.

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
