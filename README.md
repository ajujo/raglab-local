# RAG-Lab

A Retrieval-Augmented Generation (RAG) system built on the SDMX Standards: Technical Notes v2.1 document. The system enables natural language queries about technical specifications of data interchange formats (SDMX-ML, SDMX-EDI, information models, data structure definitions).

## 📋 Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [GPU Memory Management](#gpu-memory-management)
- [Configuration](#configuration)
- [Development](#development)

## 🏗️ Architecture

The RAG pipeline consists of 9 phases:

1. **Ingestion** - Clean documents and remove base64 images
2. **Chunking** - Semantic document splitting with hierarchical headings
3. **Embedding** - BGE-M3 dense + sparse embeddings
4. **Storage** - ChromaDB (HNSW, cosine similarity) + SQLite docstore
5. **Retrieval** - Hybrid search with reciprocal rank fusion
6. **Reranking** - Cross-encoder reranking (BGE-reranker-v2-m3)
7. **Generation** - LLM response with citation verification
8. **Verification** - Post-generation verification layer (citation check, self-consistency, scoring)
9. **Feedback** - User feedback collection and analysis

## ✨ Features

- **Hybrid Search**: Combines dense (BGE-M3) and sparse embeddings
- **Query Expansion**: Generates query variants for better retrieval
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers via LLM for better retrieval
- **Query Rewriting**: Reformulates user questions with domain terminology for improved semantic matching
- **Reranking**: Cross-encoder reranking for improved relevance
- **Citation Verification**: Ensures LLM responses include proper citations
- **Verification Layer**: Post-generation verification with citation checking, self-consistency evaluation, and trust scoring
- **Trust Scoring**: Calculates a confidence score (0-1) with HIGH/MEDIUM/LOW levels
- **Chunk Score Visualization**: Visual bar chart showing individual chunk relevance scores
- **Feedback Loop**: Collects user feedback on response usefulness with SQLite storage and analysis
- **Line Range Citations**: Citations include exact line ranges for precise source localization

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Conda (for environment management)
- GPU (RTX 5090 recommended, but CPU-only mode available)

### Setup

```bash
# Create and activate conda environment
conda create -n rag-lab python=3.11
conda activate rag-lab

# Install dependencies
pip install torch FlagEmbedding chromadb numpy huggingface_hub pytest

# Clone the repository
git clone https://github.com/ajujo/raglab-local.git
cd raglab-local

# Copy environment template
cp .env.example .env
```

### Environment Variables

Edit `.env` with your LLM endpoint:

```env
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=sakamakismile/Qwen3.6-27B-NVFP4
EMBEDDING_DEVICE=cuda
RERANKER_DEVICE=cuda
```

## 🔍 Verification Layer

The verification layer runs **after** the LLM generates a response, before returning it to the user. It consists of three components:

### 1. Citation Verification
- Extracts citations from the LLM response using regex
- Validates each citation against the retrieved chunks
- Classifies citations as `VALID`, `PARTIAL`, or `INVALID`
- Displays warnings for invalid citations

### 2. Self-Consistency Check (Faithfulness)
- Sends the response and retrieved chunks to the LLM for evaluation
- Checks for unsupported claims, contradictions, and hallucinations
- Configurable via `ENABLE_CONSISTENCY_CHECK` in `config.py`

### 3. Trust Scoring
Calculates a confidence score (0-1) from four sub-scores:
- **Citation Score** (35%): Ratio of valid citations
- **Retrieval Score** (30%): Average cosine similarity of retrieved chunks
- **Consistency Score** (25%): Result from the consistency check
- **Coverage Score** (10%): Ratio of chunks cited by the LLM

**Confidence Levels:**
- `HIGH` → score ≥ 0.75
- `MEDIUM` → score ≥ 0.50
- `LOW` → score < 0.50

### Chunk Score Visualization
The verification block now displays individual chunk scores with visual bars:
```
─────────────────────────────────────
Verificación de respuesta
  Fragmentos recuperados:
    [1] doc1 | Líneas 10-20  → 0.91 ████████████░░░
    [2] doc1 | Líneas 30-40  → 0.74 ██████████░░░░░
  Citas verificadas : 3/3 ✓
  Consistencia      : OK ✓
  Score de confianza: 0.87 — HIGH ✓
─────────────────────────────────────
```

### Configuration
```python
# rag_lab/config.py
ENABLE_CONSISTENCY_CHECK = True  # Set to False to disable
WEIGHT_CITATION = 0.35
WEIGHT_RETRIEVAL = 0.30
WEIGHT_CONSISTENCY = 0.25
WEIGHT_COVERAGE = 0.10
```

## 🔁 Feedback Loop

The feedback system collects user ratings on response usefulness:

```bash
# After each query, the system prompts:
# ¿Esta respuesta fue útil? [s/n] (Enter para omitir):
```

### Feedback Analysis
```bash
# Analyze collected feedback
python -m rag_lab.feedback.analyze_feedback
```

Output:
```
─────────────────────────────────────
Análisis de feedback
  Total de respuestas evaluadas: 42
  Útiles  : 35 (83%)
  No útiles: 7 (17%)

  Score medio en respuestas útiles   : 0.81
  Score medio en respuestas no útiles: 0.54

  Chunks más frecuentes en respuestas útiles:
    [1] Notas_Tecnicas_SDMX_2.1 | Líneas 45-80  → 12 veces
    [2] Notas_Tecnicas_SDMX_2.1 | Líneas 210-245 → 8 veces
─────────────────────────────────────
```

## 🚀 Usage

### Complete Pipeline

```bash
# Run full pipeline
conda activate rag-lab
python -m rag_lab.cli ingest      # Ingest documents
python -m rag_lab.cli query "Your question here"  # Query the system
```

### Query Options

```bash
# With HyDE (hypothetical document generation via LLM)
python -m rag_lab.cli query "What is SDMX?" --hyde

# With query rewriting (domain terminology expansion)
python -m rag_lab.cli query "What is SDMX?" --rewrite

# With both HyDE and rewriting
python -m rag_lab.cli query "What is SDMX?" --hyde --rewrite

# Fast mode (skip reranking)
python -m rag_lab.cli query "What is SDMX?" --fast

# Custom top-k
python -m rag_lab.cli query "What is SDMX?" --top-k 10

# Disable feedback prompt (for automated/batch usage)
python -m rag_lab.cli query "What is SDMX?" --no-feedback

# Force CPU for embedding and reranking
python -m rag_lab.cli query "What is SDMX?" --cpu-embedding --cpu-reranker
```

## 📁 Project Structure

```
RAG-Lab/
├── rag_lab/                    # Main package
│   ├── __init__.py
│   ├── config.py             # Centralized configuration
│   ├── cli.py              # CLI interface
│   ├── cli_chat.py         # Interactive chat mode
│   ├── ingest/             # Phase 1: Ingestion
│   ├── chunking/           # Phase 2: Chunking
│   ├── embedding/          # Phase 3: Embedding
│   ├── storage/            # Phase 4: Storage
│   ├── retrieval/          # Phases 5-6: Retrieval
│   │   ├── query_processor.py  # Query expansion + HyDE
│   │   ├── query_rewriter.py   # Query rewriting
│   │   ├── hybrid_search.py    # Hybrid search
│   │   └── reranker.py         # Cross-encoder reranking
│   ├── generation/         # Phase 7: Generation
│   ├── verification/       # Phase 8: Verification Layer
│   │   ├── verifier.py     # Citation verification
│   │   ├── consistency.py  # Self-consistency check
│   │   ├── scoring.py      # Trust scoring
│   │   └── pipeline.py     # Orchestration
│   └── feedback/          # Phase 9: Feedback Loop
│       ├── __init__.py
│       ├── feedback_store.py   # SQLite feedback storage
│       └── analyze_feedback.py # Feedback analysis script
├── data/                   # Processed data
├── storage/              # Database files
├── tests/                # All tests
│   ├── test_verification/ # Verification layer tests
│   ├── test_retrieval/   # Retrieval tests (including query_rewriter)
│   └── test_feedback/    # Feedback tests
└── ...
```

## 🧪 Testing

Run all tests:

```bash
conda activate rag-lab
pytest tests/ -v
```

Run specific test modules:

```bash
# All chunking tests
pytest tests/test_chunking/ -v

# All storage tests
pytest tests/test_storage/ -v

# All generation tests
pytest tests/test_generation/ -v

# All verification tests
pytest tests/test_verification/ -v

# All retrieval tests (including query rewriter)
pytest tests/test_retrieval/ -v

# All feedback tests
pytest tests/test_feedback/ -v
```

## 🎮 GPU Memory Management

The RTX 5090 has limited free VRAM. The system supports CPU-only mode for tests:

```bash
# Force CPU for tests (default for pytest)
export EMBEDDING_DEVICE=cpu
export RERANKER_DEVICE=cpu
```

Production uses GPU by default. Tests always use CPU to avoid OOM.

## ⚙️ Configuration

All parameters are centralized in `rag_lab/config.py`:

```python
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = 8
RETRIEVAL_TOP_K = 30
RERANK_TOP_K = 8
RRF_K = 60
HYDE_ENABLED = False
QUERY_REWRITING_ENABLED = False
ENABLE_CONSISTENCY_CHECK = True
WEIGHT_CITATION = 0.35
WEIGHT_RETRIEVAL = 0.30
WEIGHT_CONSISTENCY = 0.25
WEIGHT_COVERAGE = 0.10
```

## 🛠️ Development

### Logging

The system uses centralized logging:

```python
import logging
logger = logging.getLogger("rag_lab")
```

### Custom Exceptions

The system uses specific exceptions:

- `RAGLabError` - Base exception
- `DocumentIngestionError` - Document corruption/invalid
- `ChunkingError` - Chunking failures
- `EmbeddingError` - GPU OOM, embedding failures
- `RetrievalError` - Search failures
- `LLMConnectionError` - LLM server unavailable

### Test Structure

Each test module covers a specific aspect of the pipeline:

- **Unit Tests**: Each function is tested independently
- **Integration Tests**: Full pipeline execution
- **Regression Tests**: Benchmarks for Q1, Q2, Q3

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For issues and questions, please open an issue on GitHub.
