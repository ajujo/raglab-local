"""Central configuration for RAG-Lab system.

All parameters for the RAG pipeline are defined here for easy modification.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
SOURCES_DIR = BASE_DIR

# Document settings
SOURCES = [SOURCES_DIR / "Notas_Tecnicas_SDMX_2.1.md"]
# Alternative: support multiple documents
# SOURCES = [
#     SOURCES_DIR / "doc1.md",
#     SOURCES_DIR / "doc2.md",
# ]

# Chunking parameters
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP = 200
CHUNK_MIN_TOKENS = 50

# Embedding parameters
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 4
EMBEDDING_MAX_LENGTH = 1024

# Storage settings
VECTOR_STORE_PATH = "chroma_db"
SPARSE_INDEX_PATH = "sparse_index.json"
DOCDSTORE_SQLITE_PATH = "docstore.sqlite"

# Retrieval parameters
RETRIEVAL_TOP_K = 30
RERANK_TOP_K = 8
RRF_K = 60

# LLM settings
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6-35b-a3b@iq4_xs")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048

# Query processing
HYDE_ENABLED = False
FAST_MODE = False
VARIANTS_COUNT = 2

# Device settings
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")

# Test device settings — force CPU for tests to avoid OOM on GPU
TEST_EMBEDDING_DEVICE = "cpu"
TEST_RERANKER_DEVICE = "cpu"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "rag_lab.log"

# Test configuration
TEST_ASSETS_DIR = BASE_DIR / "tests" / "assets"