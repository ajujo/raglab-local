# Installing RAG-Lab

RAG-Lab is a local CLI tool for querying Markdown documents using a RAG (Retrieval-Augmented Generation) pipeline. This document covers a complete installation from scratch.

---

## 1. Prerequisites

### Python 3.11

RAG-Lab requires **Python 3.11**. Using [conda](https://docs.conda.io/) is recommended to isolate the environment and avoid dependency conflicts.

```bash
# Check available versions
python --version
conda --version
```

If you don't have conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

### NVIDIA GPU (recommended)

The development environment uses an RTX 5090. An NVIDIA GPU with CUDA support significantly speeds up embedding and reranking. However, **CPU mode is fully supported** — it is the mode used automatically by the test suite.

If you don't have a GPU, see [CPU-only mode](#6-cpu-only-mode) below.

### Your own LLM server

RAG-Lab **does not bundle an LLM**. Response generation is delegated to an external server that exposes an OpenAI-compatible API. Common options include:

- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (with HTTP server)
- [Ollama](https://ollama.com/)

The server must be running and reachable before you can run queries. Embedding and reranking happen locally with HuggingFace models; the LLM is the only external component.

### Git

```bash
git --version
```

---

## 2. Setting up the environment

### Clone the repository

```bash
git clone https://github.com/ajujo/raglab-local.git
cd raglab-local
```

### Create and activate the conda environment

```bash
conda create -n rag-lab python=3.11
conda activate rag-lab
```

### Install RAG-Lab in editable mode

```bash
pip install -e .
```

This installs the package and registers the `rag-lab` command in the environment. From here on, all commands are invoked with the `rag-lab` prefix.

---

## 3. Configuration (.env)

RAG-Lab reads its configuration from a `.env` file at the project root.

```bash
cp .env.example .env
```

Open `.env` in your editor and adjust the variables:

```dotenv
# Base URL of the LLM server (OpenAI-compatible API)
LLM_BASE_URL=http://localhost:8000/v1

# Model name as exposed by the server
LLM_MODEL=your-model-name

# Device for the embedding model (cuda or cpu)
EMBEDDING_DEVICE=cuda

# Device for the reranking model (cuda or cpu)
RERANKER_DEVICE=cuda
```

### Variable reference

| Variable | Description | Default |
|---|---|---|
| `LLM_BASE_URL` | LLM server endpoint. Include `/v1` at the end if the server requires it. | `http://localhost:8000/v1` |
| `LLM_MODEL` | Identifier of the served model. Must match what the server exposes. | — |
| `EMBEDDING_DEVICE` | Device for BAAI/bge-m3. Use `cuda` with a GPU, `cpu` without one. | `cuda` |
| `RERANKER_DEVICE` | Device for BAAI/bge-reranker-v2-m3. Independent from `EMBEDDING_DEVICE`. | `cuda` |

> **Note:** Embedding and reranker models are downloaded automatically from HuggingFace the first time they are used. No manual download is required.

---

## 4. Required models

### Embedding: BAAI/bge-m3

RAG-Lab uses [BGE-M3](https://huggingface.co/BAAI/bge-m3) to produce both dense and sparse vectors simultaneously. This model is downloaded automatically on the first `rag-lab ingest`.

- Approximate size: ~570 MB
- Lazily loaded and cached in memory for the duration of the session

### Reranker: BAAI/bge-reranker-v2-m3

[BGE-Reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) reorders retrieved candidates before passing them to the LLM. It is also downloaded automatically on first use.

- Approximate size: ~570 MB
- Can be skipped with `--fast` on queries (no reranking)

### LLM (external server)

The language model is served **outside of RAG-Lab**. There is no automatic LLM download; you manage the server yourself. RAG-Lab communicates with it over HTTP using the OpenAI-compatible API.

Example with vLLM:

```bash
# In a separate terminal or process
python -m vllm.entrypoints.openai.api_server \
  --model sakamakismile/Qwen3.6-27B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000
```

---

## 5. Verifying the installation

Once `.env` is configured and the LLM server is running:

```bash
rag-lab doctor
```

The `doctor` command checks all subsystems: configuration, stores (ChromaDB, SQLite, sparse index), LLM connectivity, FTS5 health, and more. The output clearly shows what passes and what fails.

```bash
# List all available commands
rag-lab --help

# Get help for a specific subcommand
rag-lab query --help
rag-lab ingest --help
```

If `doctor` reports empty stores (ChromaDB, docstore), that is expected behavior on a fresh installation. You need to ingest at least one document before querying:

```bash
rag-lab ingest --doc path/to/your/document.md
```

---

## 6. CPU-only mode

If you don't have an NVIDIA GPU or prefer to run on CPU:

```dotenv
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

Performance is noticeably slower on CPU, especially for embedding and reranking. For occasional use or on machines without a GPU it works perfectly well.

> **The test suite always runs on CPU.** `conftest.py` sets `CUDA_VISIBLE_DEVICES=""` automatically, so a GPU is never required to run the tests.

You can also override the device per session from the command line:

```bash
rag-lab query "my question" --cpu-embedding --cpu-reranker
```

---

## 7. Common errors

### CUDA not available

```
RuntimeError: CUDA not available
```

**Cause:** `EMBEDDING_DEVICE=cuda` or `RERANKER_DEVICE=cuda` is set but no GPU is available (or the driver is not installed).

**Fix:** Switch to `cpu` in `.env`:

```dotenv
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

---

### Connection refused to the LLM

```
LLMConnectionError: Connection refused to http://localhost:8000/v1
```

**Cause:** The LLM server is not running or is listening on a different port.

**Fix:**

1. Verify the LLM server is running.
2. Check that `LLM_BASE_URL` points to the correct host and port.
3. If the server requires `/v1` in the base URL, make sure it is included.

---

### ModuleNotFoundError

```
ModuleNotFoundError: No module named 'rag_lab'
```

**Cause:** The package is not installed in the active environment, or a different environment is active.

**Fix:**

```bash
# Make sure you are in the correct environment
conda activate rag-lab

# Reinstall from the repository root
cd /path/to/raglab-local
pip install -e .
```

---

### Empty stores after installation

```
Warning: ChromaDB collection is empty
Warning: DocStore has 0 chunks
```

**Cause:** This is expected on a fresh installation. The stores are populated during ingest.

**Fix:** Ingest at least one document:

```bash
rag-lab docs validate path/to/doc.md
rag-lab ingest --doc path/to/doc.md
```

See the [USAGE.en.md](USAGE.en.md) guide for the full ingest workflow.

---

## Directory layout after installation

```
raglab-local/
├── .env                    # Your configuration (not in git)
├── .env.example            # Configuration template
├── rag_lab/                # Pipeline source code
├── data/
│   └── ingested.jsonl      # Manifest of ingested documents
├── storage/
│   ├── chroma_db/          # Vector store (ChromaDB)
│   ├── sparse_index.json   # Sparse index (BM25-like)
│   └── docstore.sqlite     # SQLite: chunk text and metadata
└── tests/                  # Test suite
```

The `storage/` and `data/` directories are created automatically on the first ingest.
