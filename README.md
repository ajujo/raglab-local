# RAG-Lab

RAG-Lab is a local, CLI-only Retrieval-Augmented Generation system for querying technical Markdown documents. It was built to make a large SDMX standards corpus searchable via natural language, without sending data to external services. The full pipeline runs on a single machine with a locally served LLM.

## What it is NOT

- Not a web service or REST API
- Not a multi-user system
- Not a PDF, DOCX, or HTML loader — corpus must be clean Markdown
- Not a tabular data tool (no CSV, Parquet, or datasets)
- Not a production SaaS product

## Current state

Version **v1.21** — stable, in controlled local use. 1058 tests passing. Production corpus: 610 chunks.

## Quick start

```bash
conda activate rag-lab
# (clone the repo if you have not already, then:)
cp .env.example .env            # set LLM_BASE_URL, LLM_MODEL, EMBEDDING_DEVICE, RERANKER_DEVICE
rag-lab ingest                  # ingest all configured source documents
rag-lab query "What is SDMX?"   # run a query
```

Run `rag-lab doctor` after setup to verify all components are healthy.

## Architecture — 9 phases

| Phase | Description |
|-------|-------------|
| 1. Ingest | Validates Markdown, strips base64 images, writes manifest |
| 2. Chunking | Semantic split; never crosses H2+ headings; tables stay intact |
| 3. Embedding | BAAI/bge-m3, dense + sparse simultaneously, max 1024 tokens |
| 4. Storage | ChromaDB (dense vectors), SQLite DocStore (chunks + sparse BLOBs + FTS5) |
| 5. Retrieval | Dense (ChromaDB) + BM25 (FTS5) + sparse rescore + RRF fusion |
| 6. Reranking | BAAI/bge-reranker-v2-m3 cross-encoder with heading context |
| 7. Generation | Local LLM via OpenAI-compatible API |
| 8. Verification | Citation check + consistency check + trust score (HIGH/MEDIUM/LOW) |
| 9. Feedback | Observational only — stored in SQLite, has no effect on ranking |

## Benchmark results

### Retrieval quality — official suite, 65 queries, variant `full`, no cache

| Metric | Value |
|--------|-------|
| R@5 | 0.821 |
| R@10 | 0.896 |
| R@30 | 0.978 |
| MRR | 0.939 |
| nDCG@10 | 0.837 |

### Answer quality — RAGAS (v1.21, 65 queries, external judge: DeepSeek v4 Flash)

Reference-free metrics — no annotated ground truth required.

| Metric | Value | What it measures |
|--------|-------|-----------------|
| `faithfulness` | **0.9123** | Fraction of answer statements supported by retrieved contexts (anti-hallucination) |
| `answer_relevancy` | **0.7624** | How directly the answer addresses the question asked |

Full benchmark history and reproduction commands: [docs/BENCHMARK_HISTORY.md](docs/BENCHMARK_HISTORY.md)

## Documentation

- [Guía completa en castellano](README.es.md)
- [Full English guide](README.en.md)
- [Installation (en)](docs/INSTALLATION.en.md) · [Instalación (es)](docs/INSTALLATION.es.md)
- [Usage (en)](docs/USAGE.en.md) · [Uso (es)](docs/USAGE.es.md)
- [Architecture (en)](docs/ARCHITECTURE.en.md) · [Arquitectura (es)](docs/ARCHITECTURE.es.md)
- [Operations (es)](docs/OPERATIONS.es.md) · [Operations (en)](docs/OPERATIONS.en.md)
- [Benchmarks (en)](docs/BENCHMARKS.en.md) · [Benchmarks (es)](docs/BENCHMARKS.es.md)
- [Frontmatter contract (es)](docs/FRONTMATTER.es.md) · [Frontmatter contract (en)](docs/FRONTMATTER.en.md)
- [Roadmap (en)](docs/ROADMAP.en.md) · [Hoja de ruta (es)](docs/ROADMAP.es.md)
- [Development history (en)](docs/DEVELOPMENT_HISTORY.en.md)
- [CLI reference (en)](docs/API_REFERENCE.en.md) · [Referencia CLI (es)](docs/API_REFERENCE.es.md)

## License

MIT
