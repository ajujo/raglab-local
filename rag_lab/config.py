"""Panel de Control Centralizado — RAG-Lab

Todos los parámetros del sistema viven aquí para facilitar su modificación.

Estructura:
  1. Rutas del proyecto
  2. Modelo de Embedding (BGE-M3)
  3. Modelo de Reranking (BGE-reranker)
  4. Modelo LLM (Qwen 3.6 35B)
  5. Chunking
  6. Almacenamiento
  7. Recuperación (Retrieval)
  8. Consulta (Query)
  9. Dispositivos (CPU/GPU)
  10. Logging
"""

import os
from pathlib import Path

# =============================================================================
# 1. RUTAS DEL PROYECTO
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
SOURCES_DIR = BASE_DIR

# Fuentes de documentos — lista de rutas
SOURCES = [
    SOURCES_DIR / "Notas_Tecnicas_SDMX_2.1.md",
    # Agregar más documentos aquí:
    # SOURCES_DIR / "doc2.md",
    # SOURCES_DIR / "doc3.md",
]

# =============================================================================
# 2. MODELO DE EMBEDDING (BGE-M3)
# =============================================================================

# Modelo de embeddings denso + disperso
EMBEDDING_MODEL = "BAAI/bge-m3"

# Tamaño del batch para procesar embeddings (más alto = más rápido, más memoria)
# Propuesto: 8 (antes 4) — ~2x throughput en GPU
EMBEDDING_BATCH_SIZE = 8

# Longitud máxima de entrada para el modelo de embedding
EMBEDDING_MAX_LENGTH = 1024

# =============================================================================
# 3. MODELO DE RERANKING (BGE-reranker)
# =============================================================================

# Modelo de reranking (cross-encoder)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# =============================================================================
# 4. MODELO LLM (Qwen 3.6 35B)
# =============================================================================

# URL del servidor LLM (compatible con OpenAI)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")

# Nombre del modelo LLM
LLM_MODEL = os.getenv("LLM_MODEL", "sakamakismile/Qwen3.6-27B-NVFP4")

# Temperatura de generación (0.0 = determinista, 1.0 = aleatorio)
LLM_TEMPERATURE = 0.1

# Tokens máximos para la respuesta del LLM
LLM_MAX_TOKENS = 2048

# =============================================================================
# 5. CHUNKING
# =============================================================================

# Tokens máximos por chunk (no exceder EMBEDDING_MAX_LENGTH)
CHUNK_MAX_TOKENS = 800

# Superposición entre chunks en tokens (25% de CHUNK_MAX_TOKENS)
CHUNK_OVERLAP = 200

# Umbral mínimo de tokens para considerar un chunk válido
CHUNK_MIN_TOKENS = 50

# =============================================================================
# 6. ALMACENAMIENTO
# =============================================================================

# Rutas centralizadas para los almacenes
VECTOR_STORE_PATH = STORAGE_DIR / "chroma_db"
SPARSE_INDEX_PATH = STORAGE_DIR / "sparse_index.json"
DOCDSTORE_SQLITE_PATH = STORAGE_DIR / "docstore.sqlite"

# =============================================================================
# 7. RECUPERACIÓN (RETRIEVAL)
# =============================================================================

# Resultados de búsqueda híbrida antes del reranking
RETRIEVAL_TOP_K = 30

# Resultados finales después del reranking (contexto para el LLM)
RERANK_TOP_K = 8

# Constante para fusión de rangos recíprocos (RRF)
RRF_K = 60

# =============================================================================
# 8. CONSULTA (QUERY)
# =============================================================================

# Número de variantes de consulta para expansión
VARIANTS_COUNT = 2

# Activar HyDE (Hypothetical Document Embeddings)
HYDE_ENABLED = False

# Modo rápido (sin reranking)
FAST_MODE = False

# =============================================================================
# 9. DISPOSITIVOS (CPU/GPU)
# =============================================================================

# Dispositivo para el modelo de embedding
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

# Dispositivo para el modelo de reranking
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")

# Configuración de tests — fuerza CPU para evitar OOM en GPU
TEST_EMBEDDING_DEVICE = "cpu"
TEST_RERANKER_DEVICE = "cpu"

# =============================================================================
# 10. LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FILE = "rag_lab.log"

# =============================================================================
# CONFIGURACIÓN DE TESTS
# =============================================================================

TEST_ASSETS_DIR = BASE_DIR / "tests" / "assets"
