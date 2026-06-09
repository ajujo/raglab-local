# RAG-Lab Benchmark Documentation

RAG-Lab uses a retrieval benchmark to measure quality and guard against regressions
in the retrieval pipeline. The benchmark evaluates Recall@k, MRR, and nDCG@10 across
a curated set of SDMX queries with ground-truth relevance grades.

**Important:** these benchmarks measure *retrieval quality*, not full answer quality.
They do not evaluate LLM response accuracy, citation correctness, or user satisfaction.

For answer quality metrics (faithfulness, answer_relevancy), see
[docs/RAGAS_USAGE.md](RAGAS_USAGE.md). Those metrics are computed separately using
RAGAS with an external LLM judge and a dedicated `ragas` conda environment.

The RAGAS evaluation uses `answer_for_eval` — a version of the answer with inline
citations stripped — rather than the raw `answer`. See RAGAS_USAGE.md §"answer vs
answer_for_eval" for why this matters.

See also: [docs/BENCHMARKS.es.md](BENCHMARKS.es.md) — versión en castellano.

---

## Active baseline: v1.11 (CI regression guard)

**File:** `data/baselines/v1.11_official_full_eval.json`  
**Generated:** 2026-05-22  
**Queries:** 65 · **Variant:** `full` · **Corpus:** 610 chunks (tag `v1.11` @ `b2e9594`)  
**Config:** `top_k=50`, `rrf_k=20`, `RERANKER_USE_HEADING_CONTEXT=True`, query variants disabled

### Reference metrics

| Metric     | Value  |
|-----------|--------|
| Recall@5  | 0.8205 |
| Recall@10 | 0.8962 |
| Recall@30 | 0.9782 |
| MRR       | 0.9385 |
| nDCG@10   | 0.8373 |
| P50 (ms)  | 334    |
| P95 (ms)  | 384    |
| P99 (ms)  | 412    |

> **Note:** v1.11 does not change any quality metric vs v1.10 (Δ+0.0000 across all metrics).
> The latency is measured cold-start on an RTX 5090; in practice candidate generation
> is ~2× faster with query variants disabled.

### Run and compare against this baseline

```bash
# Run the official benchmark (no-cache for accurate measurement)
rag-lab benchmark --suite official --variants full --no-cache

# Compare against canonical baseline
rag-lab benchmark --suite official --variants full --no-cache --output /tmp/current.json
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current  /tmp/current.json
```

### Known regression — q070 (`cross_lingual_es_en`) — inherited from v1.10

> **Query:** "¿Cómo se utilizan las restricciones en SDMX para limitar los valores permitidos?"  
> **MRR before (v1.9):** 1.000 · **MRR after (v1.10+):** 0.500 · **Δ:** −0.500

The English structural prefix (doc_id + heading_path) slightly affects cross-encoder attention
on Spanish queries. Pre-reranker MRR=1.000 — the regression is a pure reranker effect.
To disable: set `RERANKER_USE_HEADING_CONTEXT=False` in config.

---

## Historical baseline: v1.10

**File:** `data/baselines/v1.10_official_full_eval.json`  
**Queries:** 65 · **Variant:** `full` · **Corpus:** 610 chunks (tag `v1.10` @ `adb1a5a`)

> Active baseline from v1.10 to v1.11. Kept as historical reference.
> **Use `v1.11_official_full_eval.json` for current regression guards.**

| Metric     | Value  |
|-----------|--------|
| Recall@5  | 0.8205 |
| Recall@10 | 0.8962 |
| Recall@30 | 0.9782 |
| MRR       | 0.9385 |
| nDCG@10   | 0.8373 |

---

## Historical baseline: v1.8.1

**File:** `data/baselines/v1.8.1_official_full_eval.json`  
**Queries:** 65 (28 original + 37 curated in v1.8.1) · **Variant:** `full`  
**Corpus:** 610 chunks (tag `v1.8.1` @ `614a836`)

> Active baseline from v1.8.1 to v1.10. Kept as historical reference.

| Metric     | Value  |
|-----------|--------|
| Recall@5  | 0.8000 |
| Recall@10 | 0.9141 |
| Recall@30 | 0.9731 |
| MRR       | 0.9128 |
| nDCG@10   | 0.8255 |
| P50 (ms)  | 250.92 |
| P95 (ms)  | 257.89 |
| P99 (ms)  | 270.03 |

---

## Historical baseline: v1.7

**File:** `data/baselines/v1.7_official.json`  
**Queries:** 28 (q001–q028) · **Variant:** `full` · **Corpus:** 610 chunks (v1.7 @ `00882e3`)

| Metric     | Value  |
|-----------|--------|
| Recall@5  | 0.7708 |
| Recall@10 | 0.9256 |
| Recall@30 | 0.9613 |
| MRR       | 0.8780 |
| nDCG@10   | 0.8067 |
| P50 (ms)  | 240.18 |

---

## Metrics explained

| Metric | Description |
|--------|-------------|
| `Recall@k` | Doc-level: fraction of annotated relevant docs with ≥1 chunk in top-k results |
| `MRR` | Mean Reciprocal Rank: 1/rank of the first relevant result (0 if not found) |
| `nDCG@10` | Normalised DCG using grades 0–3 per doc; each doc counted at first appearance only |
| `P50/P95/P99 ms` | Retrieval latency percentiles (encoding excluded for fair cross-variant comparison) |
| `Pool` | Mean candidate pool size entering fusion/rerank |
| `dense/bm25/sparse coverage` | Fraction of returned results that carry each retrieval signal |

---

## What `full` represents (and what it doesn't)

`full` is the closest reproducible proxy for the production pipeline. It includes
the BGE cross-encoder reranker, which has the highest impact on final result ordering.

**`full` is NOT an exact replica of the interactive production pipeline.**

| Aspect | Production CLI | Benchmark `full` |
|--------|----------------|------------------|
| Query expansion | 1 query (variants off) | 1 query |
| MMR before rerank | Yes (`MMR_ENABLED=True`) | No (off) |
| top_k | 50 (CLI default) | 50 |
| Reranker | Yes | Yes |

The MMR difference has minimal effect on metrics (the reranker re-orders the full list anyway).
`full` is deterministic and reproducible — which makes it the correct regression guard.

---

## Available variants

| Variant | Description | Default run |
|---------|-------------|:-----------:|
| `dense` | Dense only (ChromaDB cosine) | Yes |
| `bm25` | BM25/FTS5 only | Yes |
| `dense_bm25` | RRF: dense + BM25 | Yes |
| `hybrid` | RRF: dense + BM25 + BGE-M3 sparse | Yes |
| `full` | hybrid + BGE cross-encoder reranker (production) | Yes |
| `hybrid_cap` | hybrid + doc cap (opt-in) | No |
| `hybrid_mmr` | hybrid + MMR diversity (opt-in) | No |
| `full_hyde` | full + HyDE dense augmentation (experimental, requires LLM) | No |

---

## Running the benchmark

```bash
# Standard CI run (65 official queries, full variant, no cache)
rag-lab benchmark --suite official --variants full --no-cache

# Save output for comparison
rag-lab benchmark --suite official --variants full --no-cache \
    --output data/baselines/run_$(date +%Y%m%d).json

# All variants comparison
rag-lab benchmark --variants dense bm25 dense_bm25 hybrid full

# Candidate queries only
rag-lab benchmark --suite candidates --variants full
```

---

## Regression thresholds (compare.py)

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Recall@5 | drop > 2 pp | FAIL |
| nDCG@10 | drop > 2 pp | FAIL |
| MRR | drop > 3 pp | FAIL |
| P95 latency | increase > 25% (relative) | WARN |

Exit codes: `0` = OK · `1` = WARN · `2` = FAIL

---

## Suite and query structure

### `data/benchmark_queries.yaml` format

```yaml
queries:
  - id: q001
    text: "What is SDMX?"
    category: glossary_definition
    language: en
    suite: official
    validated: true
    expected_behavior: "Return definition chunks from Glossary and Training"
    source_of_truth: "SDMX_Glossary, SDMX-Training-introduction-2015"
    doc_relevance:
      SDMX-Training-introduction-2015: 3
      SDMX_Glossary: 3
```

### Suite distribution (v1.8.1+)

| Suite | Validated | Count | Use |
|-------|-----------|-------|-----|
| `official` | `true` | 65 | CI regression guard |
| `candidate` | `true` | 4 | Confirmed negatives — excluded from guard |
| `candidate` | `false` | 3 | Pending review — excluded from guard |

Categories: `glossary_definition`, `technical_standard`, `cross_lingual_es_en`,
`multi_chunk_same_doc`, `multi_doc_synthesis`, `acronym_or_exact_term`,
`table_or_structured_reference`, `negative_no_answer`, `ambiguity_test`, `regression_known_hard`.

---

## HyDE experiment (v1.12) — disabled by default

A/B benchmark over 65 official queries (2026-05-22):

| Metric | full (v1.11) | full_hyde | Δ | Verdict |
|--------|-------------|-----------|---|---------|
| R@5 | 0.8205 | 0.7821 | −0.038 | ❌ FAIL |
| R@10 | 0.8962 | 0.8577 | −0.038 | ❌ |
| MRR | 0.9385 | 0.9385 | 0.000 | ✓ |
| nDCG@10 | 0.8373 | 0.8187 | −0.019 | ⚠ |
| P50 (ms) | 237 | 2966 | +2729 | ❌ 12.5× slower |

**Conclusion:** BGE-M3 is already strong enough on the SDMX corpus. HyDE displaces
the dense query toward slightly different vocabulary, causing more misses than hits.
The LLM call overhead (×12.5 latency) is unacceptable for production use.

HyDE remains available as an opt-in flag (`--hyde`) for experimentation.
Current config: `HYDE_ENABLED = False` in `rag_lab/config.py`.

---

## HNSW experiment (v1.13) — build-time parameters

All HNSW parameters in ChromaDB 1.x are build-time. Changing config without
rebuilding the collection has no effect on the running index.

| Profile | M | ef_c | ef_s | P50 (ms) | Recall vs prod |
|---------|----|------|------|----------|----------------|
| current | 16 | 100 | 100 | 1.87 | 0.9547 |
| fast | 8 | 64 | 50 | 1.87 | **0.8313** ❌ |
| balanced | 16 | 128 | 100 | 1.91 | 0.9553 |
| recall | 32 | 200 | 200 | 2.09 | 0.9533 |

**Recommendation: keep `current` (M=16, ef_c=100, ef_s=100).**
`fast` degrades recall by ~12pp. `balanced`/`recall` give <0.001 improvement at 610 chunks.
HNSW latency (~2ms) is negligible compared to reranker (~250ms) — no E2E impact.
