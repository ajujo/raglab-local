# RAG-Lab — Full English Guide

## 1. What is RAG-Lab

RAG-Lab is a local, CLI-only Retrieval-Augmented Generation system for querying technical Markdown documents. It was built to make a large SDMX standards corpus searchable via natural language, without sending data to any external service. The entire pipeline runs on a single machine with a locally served language model.

The current corpus consists of SDMX (Statistical Data and Metadata eXchange) technical documentation: technical notes, glossary, user guides, and training materials. The system ingests those documents, splits them into semantic chunks, embeds them with BAAI/bge-m3, indexes them in ChromaDB and SQLite, and answers queries by combining dense search, BM25, and a cross-encoder reranker before passing the context to the LLM.

---

## 2. What problem does it solve

Large technical specifications — thousand-page standards, glossaries, appendices — are impractical to search manually. RAG-Lab lets you ask questions in natural language and receive contextualized answers with exact citations (chunk number, source document line range). The system also reports a confidence score for each answer, so you know how much to rely on it.

---

## 3. What it is NOT

- Not a web service or public REST API.
- Not a multi-user system.
- Not a PDF, DOCX, or HTML loader — the corpus must be clean Markdown.
- Not a tabular data tool (no CSV, Parquet, or datasets).
- Not a production SaaS product or cloud deployment.
- No graphical user interface — CLI only.

---

## 4. Current state

| Field | Value |
|-------|-------|
| Version | v1.19.1 |
| Status | Stable, controlled local use |
| Tests | 1031 passing |
| Corpus | 610 chunks in production |
| Python | 3.11 (conda env `rag-lab`) |
| Recommended GPU | RTX 5090 (CPU mode available) |

---

## 5. Quick installation

```bash
# Create and activate the conda environment
conda create -n rag-lab python=3.11
conda activate rag-lab

# Install in editable mode (registers the rag-lab CLI entry point)
pip install -e .

# Configure the environment
cp .env.example .env
# Edit .env with your values:
#   LLM_BASE_URL=http://localhost:8000/v1
#   LLM_MODEL=your-local-model-name
#   EMBEDDING_DEVICE=cuda   # or cpu
#   RERANKER_DEVICE=cuda    # or cpu

# Verify everything is healthy
rag-lab doctor
```

The LLM must be available as an OpenAI-compatible API at the configured URL. If no GPU is available, set `EMBEDDING_DEVICE=cpu` and `RERANKER_DEVICE=cpu` — this works but increases latency significantly.

---

## 6. First use

```bash
# Validate a document before ingesting it
rag-lab docs validate path/to/doc.md

# Ingest all documents configured in SOURCES
rag-lab ingest

# Run a query
rag-lab query "What is a Data Structure Definition in SDMX?"
```

---

## 7. Main commands

### Queries

| Command | Description |
|---------|-------------|
| `rag-lab query "question"` | Standard query |
| `rag-lab query "question" --fast` | Skip the reranker (faster, lower precision) |
| `rag-lab query "question" --top-k N` | Change the number of final chunks |
| `rag-lab query "question" --hyde` | Enable HyDE (off by default — see section 14) |
| `rag-lab query "question" --rewrite` | Enable query rewriting (off by default) |
| `rag-lab query "question" --no-cache` | Bypass the results cache |
| `rag-lab query "question" --profile` | Show per-phase timing |
| `rag-lab chat` | Interactive chat with document selection |

### Ingestion

| Command | Description |
|---------|-------------|
| `rag-lab ingest` | Ingest all configured source documents |
| `rag-lab ingest --doc path/to/doc.md` | Ingest a single document |
| `rag-lab ingest --strict` | Treat frontmatter WARNs as errors |
| `rag-lab ingest --force` | Re-ingest even if the document has not changed |
| `rag-lab ingest --resume` | Resume an incomplete batch |
| `rag-lab ingest --retry-failed` | Retry failed documents from the last batch |
| `rag-lab ingest --workers N` | Parallel workers for large batches |

### Document management

| Command | Description |
|---------|-------------|
| `rag-lab docs list` | List all ingested documents |
| `rag-lab docs show <doc_id>` | Show full metadata for a document |
| `rag-lab docs tag <doc_id> <tag>` | Assign a tag |
| `rag-lab docs untag <doc_id> <tag>` | Remove a tag |
| `rag-lab docs delete <doc_id>` | Delete a document from all stores |
| `rag-lab docs validate path/to/doc.md` | Validate YAML frontmatter |
| `rag-lab docs inspect path/to/doc.md` | Show structure, token count, estimated chunks |
| `rag-lab docs preview-chunks path/to/doc.md` | Preview chunks without writing anything |

### Tags

| Command | Description |
|---------|-------------|
| `rag-lab tags list` | List all tags |
| `rag-lab tags rename <old> <new>` | Rename a tag |
| `rag-lab tags delete <tag>` | Delete a tag |

### Cache

| Command | Description |
|---------|-------------|
| `rag-lab cache stats` | Cache usage statistics |
| `rag-lab cache clear` | Clear all cached entries |
| `rag-lab cache vacuum` | Remove expired entries |
| `rag-lab cache inspect <key>` | Inspect a specific entry |

### Feedback

| Command | Description |
|---------|-------------|
| `rag-lab feedback list` | List feedback records |
| `rag-lab feedback stats` | Aggregated statistics |
| `rag-lab feedback export --output path.jsonl` | Export to JSONL |
| `rag-lab feedback clear --yes` | Delete all feedback records |

### Operations and diagnostics

| Command | Description |
|---------|-------------|
| `rag-lab doctor` | Full health check |
| `rag-lab doctor --checks config,docstore` | Run specific checks only |
| `rag-lab reconcile --check` | Check consistency across stores |
| `rag-lab reconcile --repair` | Remove ChromaDB orphans |
| `rag-lab reconcile --repair-fts` | Fix FTS5 duplicates |
| `rag-lab reconcile --repair-metadata` | Back-fill NULL model metadata |
| `rag-lab diagnose --query "..." --explain` | Full diagnostic with signal breakdown |
| `rag-lab benchmark --suite official --variants full --no-cache` | Run the official benchmark |

---

## 8. Architecture — 9 phases

The pipeline runs 9 sequential phases, each in its own subpackage under `rag_lab/`:

```
ingest -> chunking -> embedding -> storage -> retrieval -> reranking -> generation -> verification -> feedback
```

| Phase | Package | What it does |
|-------|---------|--------------|
| 1. Ingest | `rag_lab/ingest/` | Validates Markdown, strips base64 images, writes `data/ingested.jsonl` manifest |
| 2. Chunking | `rag_lab/chunking/` | Semantic split — never crosses H2+ headings, tables stay intact as one chunk |
| 3. Embedding | `rag_lab/embedding/` | BAAI/bge-m3, dense + sparse vectors simultaneously, max 1024 tokens; model is globally cached |
| 4. Storage | `rag_lab/storage/` | ChromaDB for dense vectors; SQLite DocStore for chunk text, metadata, sparse BLOBs, FTS5 index |
| 5. Retrieval | `rag_lab/retrieval/` | Dense + BM25 + sparse rescore + RRF fusion; optional query expansion and HyDE |
| 6. Reranking | `rag_lab/retrieval/reranker.py` | BAAI/bge-reranker-v2-m3 cross-encoder with heading context |
| 7. Generation | `rag_lab/generation/` | Formats numbered chunk context; calls local OpenAI-compatible LLM endpoint |
| 8. Verification | `rag_lab/verification/` | Citation check + consistency check + trust score (0-1, HIGH/MEDIUM/LOW) |
| 9. Feedback | `rag_lab/feedback/` | Observational only — stores user ratings in SQLite; no effect on ranking |

### Key configuration (`rag_lab/config.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| CHUNK_MAX_TOKENS | 800 | Must not exceed EMBEDDING_MAX_LENGTH (1024) |
| RETRIEVAL_TOP_K | 30 | Candidates before reranking |
| RERANK_TOP_K | 8 | Chunks passed to the LLM |
| RRF_K | 60 | RRF fusion constant |
| ENABLE_CONSISTENCY_CHECK | True | Disabling saves one LLM call per query |

---

## 9. Retrieval — how it works

Retrieval combines four signals before the reranker sees any result:

1. **Dense search** — ChromaDB cosine similarity over BGE-M3 dense vectors.
2. **BM25** — exact-term search over the FTS5 index in SQLite.
3. **Sparse rescore** — BGE-M3 sparse vectors stored as BLOBs in SQLite rescore the candidate pool.
4. **RRF fusion** — Reciprocal Rank Fusion merges the three ranked lists into a single pool.

After fusion, the candidate pool optionally passes through MMR (Maximal Marginal Relevance, enabled by default via `MMR_ENABLED=True`) for diversity before the reranker.

The cross-encoder reranker (BAAI/bge-reranker-v2-m3) evaluates each (query, chunk) pair with heading context to produce the final ranked order. The top `RERANK_TOP_K` chunks are passed to the LLM as context.

---

## 10. Verification layer

After the LLM generates a response, the system automatically runs three verification steps:

**Step 1 — Citation check:** extracts citations from the response text by regex and validates them against the retrieved chunks. Each citation is classified as VALID, PARTIAL, or INVALID. Invalid citations produce warnings.

**Step 2 — Consistency check:** a second LLM call evaluates whether the response is supported by the retrieved chunks, checking for unsupported claims and contradictions. Controlled by `ENABLE_CONSISTENCY_CHECK` in `config.py` (on by default). Disabling it saves one LLM call per query.

**Step 3 — Trust score:** combines four sub-scores into a single 0-1 confidence value:

| Sub-score | Weight |
|-----------|--------|
| Citations (ratio of valid citations) | 35% |
| Retrieval (mean cosine similarity, min-max normalized) | 30% |
| Consistency | 25% |
| Coverage (chunks cited / chunks retrieved) | 10% |

Confidence levels: HIGH (>= 0.75), MEDIUM (>= 0.50), LOW (< 0.50).

---

## 11. Frontmatter contract

Since v1.19, Markdown documents should include a YAML frontmatter block with classification metadata. Only `doc_id` is strictly required. Missing recommended fields produce WARNs at ingest time; with `--strict`, WARNs become blocking errors.

### Recommended full example

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
---
```

### Fields

| Field | Required | Missing behaviour |
|-------|----------|-------------------|
| `doc_id` | Yes | ERROR — ingest blocked |
| `title` | Recommended | WARN — falls back to first H1 |
| `domain` | Recommended | WARN |
| `source_type` | Recommended | WARN |
| `language` | Recommended | WARN |
| `version` | Optional | No warning |
| `tags` | Optional | No warning |

### Derived tags

Classification fields automatically generate tags at ingest time:

| Field | Derived tag |
|-------|-------------|
| `domain: sdmx` | `domain:sdmx` |
| `source_type: manual` | `source_type:manual` |
| `language: en` | `lang:en` |
| `version: "2.1"` | `version:2.1` |

These tags allow filtering the retrieval pool without changing the ranking algorithm. Filtering is done via `FilterSpec` — for example, `FilterSpec(domain="sdmx")` restricts the candidate pool to documents tagged `domain:sdmx`.

The fields `dataset` and `dataset_id` are **explicitly prohibited** — their presence triggers a `frontmatter_scope_violation` ERROR. RAG-Lab does not support tabular data.

Documents without frontmatter are technically valid (WARN `frontmatter_missing`) but adding the full contract to every new document is recommended.

Full documentation: [docs/FRONTMATTER.es.md](docs/FRONTMATTER.es.md)

---

## 12. Benchmark results

The benchmark measures retrieval quality on 65 curated queries with ground-truth relevance grades. **It does not measure the quality of complete LLM answers.**

### Official results — baseline v1.11

Suite: official | 65 queries | variant: full | no cache | corpus: 610 chunks

| Metric | Value |
|--------|-------|
| Recall@5 | 0.821 |
| Recall@10 | 0.896 |
| Recall@30 | 0.978 |
| MRR | 0.939 |
| nDCG@10 | 0.837 |
### Answer quality — RAGAS (v1.21, 65 queries, external judge: DeepSeek v4 Flash)

Reference-free metrics — no annotated ground truth required.

| Metric | Value | What it measures |
|--------|-------|-----------------|
| `faithfulness` | **0.9123** | Fraction of answer statements supported by retrieved contexts (anti-hallucination) |
| `answer_relevancy` | **0.7624** | How directly the answer addresses the question asked |

Full benchmark history: [docs/BENCHMARK_HISTORY.md](docs/BENCHMARK_HISTORY.md)

| P50 latency | 334 ms |
| P95 latency | 384 ms |

The `full` variant includes the cross-encoder reranker (the most impactful step for final ordering) but does not include MMR pre-reranker. It is the correct proxy for regression guards because it is fully reproducible and deterministic.

### Running the benchmark

```bash
rag-lab benchmark --suite official --variants full --no-cache
```

### Comparing against the official baseline

```bash
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current /tmp/current_run.json
```

Exit codes: 0 = OK, 1 = WARN, 2 = FAIL. Regression thresholds: R@5 drop > 2 pp = FAIL, nDCG@10 drop > 2 pp = FAIL, MRR drop > 3 pp = FAIL.

Full documentation: [docs/BENCHMARKS.en.md](docs/BENCHMARKS.en.md)

---

## 13. Feedback — observational only

After each query, the system can collect a user rating (useful / not useful). These records are stored in SQLite (`rag_lab/feedback/feedback.db`).

**Feedback is strictly observational. It does not affect ranking, retrieval scores, or any part of the pipeline.** This is an explicit, deliberate design decision: the current feedback dataset is too small to use as a ranking signal without significant overfitting risk.

```bash
rag-lab feedback stats                           # statistics
rag-lab feedback export --output feedback.jsonl  # export all records
```

---

## 14. Disabled by default

### HyDE (Hypothetical Document Embeddings)

HyDE generates a hypothetical LLM answer and uses its embedding as an additional query vector. Available via `--hyde`, but **off by default** based on benchmark evidence:

A/B benchmark results over 65 official queries (v1.12, 2026-05-22):

| Metric | full (baseline) | full_hyde | Delta |
|--------|----------------|-----------|-------|
| R@5 | 0.821 | 0.782 | -3.8 pp |
| R@10 | 0.896 | 0.858 | -3.8 pp |
| MRR | 0.939 | 0.939 | 0.0 pp |
| nDCG@10 | 0.837 | 0.819 | -1.9 pp |
| P50 latency | 237 ms | 2966 ms | 12.5x slower |

Why it underperforms here: BGE-M3 embedding is already strong on the SDMX corpus. The hypothetical text shifts the dense search toward slightly different vocabulary, causing more misses than hits. The extra LLM call per query (12.5x latency increase) is unacceptable for interactive use.

When to reconsider HyDE: if the corpus grows with documents whose vocabulary is much more specialized than the typical user query phrasing, or if the local LLM improves significantly in speed.

### Query rewriting

Query rewriting reformulates the user's question with domain terminology before retrieval. Available via `--rewrite`. It has not been systematically benchmarked; it remains off by default until there is solid evidence of improvement.

### Feedback as a ranking signal

Feedback collection is active but the ranking signal is explicitly frozen. Feedback data is available for analysis but is not read by any retrieval or scoring component.

---

## 15. Further documentation

| Document | Content |
|----------|---------|
| [README.es.md](README.es.md) | Full guide in Spanish |
| [docs/BENCHMARKS.en.md](docs/BENCHMARKS.en.md) | Detailed benchmarks, baseline history, all variants |
| [docs/FRONTMATTER.es.md](docs/FRONTMATTER.es.md) | Full YAML contract with all fields and examples |
| [docs/OPERATIONS.es.md](docs/OPERATIONS.es.md) | Operations guide: doctor, reconcile, diagnose, maintenance |
| [docs/ANSWER_VERIFICATION.md](docs/ANSWER_VERIFICATION.md) | Answer verification system documentation |

---

## 16. License

MIT
