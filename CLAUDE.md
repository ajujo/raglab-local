# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.11 via conda (`conda activate rag-lab`)
- GPU: RTX 5090, but tests always run on CPU (`CUDA_VISIBLE_DEVICES=""` is set in `conftest.py`)
- LLM served locally via OpenAI-compatible API (default: `http://localhost:8000/v1`, model `sakamakismile/Qwen3.6-27B-NVFP4`)
- Copy `.env.example` to `.env` and set `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_DEVICE`, `RERANKER_DEVICE`

## Commands

```bash
# Run all tests (always uses CPU — no GPU needed)
pytest tests/ -v

# Run a single test file
pytest tests/test_chunking/test_splitter.py -v

# Run a single test by name
pytest tests/test_chunking/test_splitter.py::test_chunk_document -v

# Ingest all configured source documents
python -m rag_lab.cli ingest

# Ingest a single document
python -m rag_lab.cli ingest --doc path/to/doc.md

# Query
python -m rag_lab.cli query "Your question" [--hyde] [--rewrite] [--fast] [--top-k N] [--profile]

# Interactive chat with document filtering
python -m rag_lab.cli chat

# Document manager (list, add, delete, tag, search, interactive)
python -m rag_lab.doc_manager list
python -m rag_lab.doc_manager add path/to/doc.md
python -m rag_lab.doc_manager interactive

# Analyze collected feedback
python -m rag_lab.feedback.analyze_feedback

# Reconcile: cross-store consistency check (DocStore vs ChromaDB)
python -m rag_lab.maintenance.reconcile          # report only
python -m rag_lab.maintenance.reconcile --fix    # also removes orphaned IDs from ChromaDB

# Migrate existing docstore to schema v2 (sparse BLOBs + FTS5 virtual table)
python -m rag_lab.maintenance.migrate_to_v2
```

## Architecture

The RAG pipeline has 9 sequential phases, each in its own subpackage:

```
ingest → chunking → embedding → storage → retrieval → reranking → generation → verification → feedback
```

### Data flow

1. **`rag_lab/ingest/`** — `clean_document()` strips base64 images; `create_manifest()` writes `data/ingested.jsonl` to deduplicate runs.
2. **`rag_lab/chunking/`** — `chunk_document()` produces `Chunk` objects. The splitter never crosses H2+ headings; tables stay intact in one chunk. Each chunk has `doc_id`, `heading_path`, `line_start`/`line_end`, `tipo` (`texto`/`tabla`/`formula`).
3. **`rag_lab/embedding/`** — `encode_chunks()` uses BGE-M3 to produce both dense and sparse vectors simultaneously. Embedding model is lazily loaded and cached globally; call `reset_embedding_cache()` between tests.
4. **`rag_lab/storage/`** — Three stores:
   - `VectorStore`: ChromaDB (`storage/chroma_db/`), cosine similarity, HNSW index.
   - `SparseStore`: JSON-backed sparse index (`storage/sparse_index.json`).
   - `DocStore`: SQLite (`storage/docstore.sqlite`), source of truth for chunk text and metadata.
5. **`rag_lab/retrieval/`** — `hybrid_search()` fuses dense + sparse results via Reciprocal Rank Fusion (RRF). Optional `query_processor.py` expands the query with variants and/or HyDE. `query_rewriter.py` rewrites with domain terminology. `reranker.py` uses BGE-reranker-v2-m3 (cross-encoder); also cached globally.
6. **`rag_lab/generation/`** — `prompt_builder.py` formats numbered chunk context; `llm_client.py` calls the local OpenAI-compatible endpoint.
7. **`rag_lab/verification/`** — Runs after LLM generation. `verifier.py` checks citations via regex; `consistency.py` sends a second LLM call to detect hallucinations; `scoring.py` calculates a weighted trust score (citation 35%, retrieval 30%, consistency 25%, coverage 10%). Retrieval scores are min-max normalized before display.
8. **`rag_lab/feedback/`** — SQLite-backed (`rag_lab/feedback/feedback.db`). `FeedbackEntry` stores query, hyde flag, chunk metadata, score, and `useful` boolean.
9. **`rag_lab/doc_manager/`** — Separate document catalog (SQLite) with tag/collection support, duplicate detection by hash, and an interactive TUI (`interactive.py`).

### Configuration

All tuneable parameters live in `rag_lab/config.py`. Key ones:

| Parameter | Default | Notes |
|---|---|---|
| `CHUNK_MAX_TOKENS` | 800 | Must not exceed `EMBEDDING_MAX_LENGTH` (1024) |
| `RETRIEVAL_TOP_K` | 30 | Candidates before reranking |
| `RERANK_TOP_K` | 8 | Chunks passed to LLM |
| `RRF_K` | 60 | RRF fusion constant |
| `ENABLE_CONSISTENCY_CHECK` | True | Disabling saves one LLM call per query |
| `SOURCES` | list of Paths | Documents ingested by default |
| `MULTI_DOC_ENABLED` | True | Filter by `ACTIVE_DOCS` (empty = all) |

Devices are read from env vars `EMBEDDING_DEVICE` / `RERANKER_DEVICE` (default `cuda`). Tests override these to `cpu` via `conftest.py`.

### CLI entry points

- `python -m rag_lab.cli` — main pipeline (ingest / query / chat)
- `python -m rag_lab.doc_manager` — document management
- `rag_lab/cli_chat.py:run_chat()` — interactive chat loop, wraps the query pipeline with document selection

### Custom exceptions (`rag_lab/exceptions.py`)

`RAGLabError` → `DocumentIngestionError`, `ChunkingError`, `EmbeddingError`, `RetrievalError`, `LLMConnectionError`

### Logging

Use `logging.getLogger("rag_lab")` everywhere. Configured by `setup_logging()` from `rag_lab/logging_config.py`. Output goes to both console and `rag_lab.log`.
