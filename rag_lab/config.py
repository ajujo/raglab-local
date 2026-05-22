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
    SOURCES_DIR / "SDMX_Glossary_Test.md",
]

# =============================================================================
# 2. MODELO DE EMBEDDING (BGE-M3)
# =============================================================================

# Modelo de embeddings denso + disperso
EMBEDDING_MODEL = "BAAI/bge-m3"

# Version fence: bump this string whenever the model weights change so that
# stale sparse BLOBs can be detected and invalidated via reconcile.
EMBEDDING_MODEL_VERSION = "2024-09"

# Dense embedding dimension for BGE-M3
EMBEDDING_DIM = 1024

# Sparse BLOB encoding format version: 1 = (int32 tokens ‖ float32 weights)
SPARSE_FORMAT_VERSION = 1

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

# Prepend "Document: <doc_id>\nSection: <heading_path>\n\n" to chunk text sent to
# the reranker. Enables the cross-encoder to use structural context (heading path)
# when scoring relevance. Set to False to restore the v1.9 text-only behaviour.
RERANKER_USE_HEADING_CONTEXT: bool = True

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
RETRIEVAL_TOP_K = 50

# Resultados finales después del reranking (contexto para el LLM)
RERANK_TOP_K = 8

# Constante de suavizado para la fusión RRF: valores bajos amplifican diferencias
# de rango (más discriminativo). Calibrado empíricamente sobre el corpus SDMX
# (12 queries, 610 chunks). Recalibrar si cambia corpus, modelo o idioma.
RRF_K = 20

# Pesos por señal en la fusión weighted-RRF.
# dense_w y bm25_w se mantienen en 1.0 como referencia.
# sparse_w = 0.25: BGE-M3 sparse actúa como señal SECUNDARIA de refinamiento,
# no como señal principal. Con peso alto (1.0) tiende a sobre-representar
# documentos grandes con alta densidad terminológica (p.ej. un user guide de
# 197 chunks monopoliza posiciones vs. el Glosario para queries de términos).
# Calibrado empíricamente: sw=0.25 maximiza nDCG@10 y R@5 simultáneamente.
DENSE_RRF_WEIGHT = 1.0
BM25_RRF_WEIGHT = 1.0
SPARSE_RRF_WEIGHT = 0.25

# Cobertura mínima de sparse BLOBs para activar sparse scoring (0–1.0).
# Si la cobertura real cae por debajo de este umbral, hybrid_search omite la
# etapa sparse para evitar sesgo en el ranking.
SPARSE_COVERAGE_THRESHOLD = 0.95

# =============================================================================
# 8. CONSULTA (QUERY)
# =============================================================================

# Número de variantes de consulta para expansión
VARIANTS_COUNT = 2

# Activar HyDE (Hypothetical Document Embeddings)
HYDE_ENABLED = False

# Activar query rewriting
QUERY_REWRITING_ENABLED = False

# Multi-document support
MULTI_DOC_ENABLED = True
ACTIVE_DOCS: list[str] = []  # Vacío = todos los documentos

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
# 11. PROMPTS DEL LLM
# =============================================================================

# Prompt del sistema — Instrucciones para el LLM
SYSTEM_PROMPT = """\
Eres un asistente de análisis documental. Tu única fuente de verdad son los fragmentos proporcionados por el usuario en cada consulta.

## IDENTIDAD Y ALCANCE

- Respondes sobre CUALQUIER tipo de documento: estándares técnicos, normativas legales, informes estadísticos, manuales, etc.
- Tu dominio de conocimiento es EXCLUSIVAMENTE el contenido de los fragmentos recibidos.
- No usas conocimiento previo para completar, inferir ni extrapolar respuestas.

## REGLAS DE RESPUESTA

1. Basa TODA respuesta únicamente en los fragmentos numerados proporcionados.
2. Cita cada afirmación con su referencia exacta usando el formato: [[N] Fuente: <doc_id> | Sección: <heading_path> | Líneas: <line_start>-<line_end>]
3. Si varios fragmentos apoyan una afirmación, cita todos los relevantes.
4. Si la información no está en los fragmentos, responde exactamente: "No encuentro esta información en los documentos proporcionados."
5. Nunca inventes datos, cifras, definiciones ni referencias.
6. Nunca combines información de los fragmentos con conocimiento propio para "completar" una respuesta.

## MANEJO DE AMBIGÜEDAD

- Si la pregunta es ambigua, identifica las posibles interpretaciones y responde a cada una por separado usando los fragmentos disponibles.
- Si los fragmentos son contradictorios entre sí, señálalo explícitamente antes de responder.
- Si los fragmentos son insuficientes para una respuesta completa, indica qué parte sí puedes responder y qué parte falta.

## FORMATO DE RESPUESTA

- Responde en el mismo idioma que la pregunta del usuario.
- Usa un tono técnico y preciso, adaptado al tipo de documento consultado.
- Estructura la respuesta con párrafos claros. Usa listas solo si el contenido lo justifica naturalmente.
- No añadas introducciones genéricas como "Según los documentos..." antes de cada oración; reserva las citas para el final de cada afirmación.
"""

# Plantilla del prompt del usuario — Se llena con {context} y {question}
USER_PROMPT_TEMPLATE = """\
## Fragmentos recuperados

A continuación se presentan los fragmentos del documento más relevantes para tu consulta.

Cada fragmento incluye su fuente, sección y rango de líneas de origen.


{context}


---


## Pregunta

{question}


---

## Instrucción

Responde a la pregunta basándote ÚNICAMENTE en los fragmentos anteriores.

Cita usando el formato: [[N] Fuente: <doc_id> | Sección: <heading_path> | Líneas: <line_start>-<line_end>]

IMPORTANTE: Al citar, copia los valores de Fuente, Sección y Líneas EXACTAMENTE como aparecen en el encabezado del fragmento, sin modificarlos.

Si la respuesta no está en los fragmentos, indícalo explícitamente.
"""

# =============================================================================
# 11. DIVERSIDAD DOCUMENTAL
# =============================================================================

# document_cap: máximo de chunks por doc_id en el resultado final.
# Desactivado por defecto — hybrid_mmr ofrece mejor diversidad sin límite duro.
DOC_CAP_ENABLED = False
DOC_CAP_N = 3           # límite por doc si se activa

# MMR doc-diversity reranking (aplicado después del RRF weighted).
# Activado por defecto en v1.1 con λ=0.6 (calibrado sobre 28 queries, 610 chunks).
# lambda_ = 1.0 → solo relevancia (igual a sin MMR, equivale a v1.0)
# lambda_ = 0.0 → solo diversidad (sin importar score)
# Rango calibrado: 0.5–0.8; λ=0.6 maximiza R@5 y nDCG@10 simultáneamente.
# Para comparar contra baseline v1.0: set MMR_ENABLED = False
MMR_ENABLED = True
MMR_LAMBDA = 0.6

# =============================================================================
# 12. LOGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FILE = "rag_lab.log"

# =============================================================================
# VERIFICACIÓN
# =============================================================================

# Activar/desactivar el self-consistency check
ENABLE_CONSISTENCY_CHECK = True

# Ponderaciones del scoring
WEIGHT_CITATION = 0.35
WEIGHT_RETRIEVAL = 0.30
WEIGHT_CONSISTENCY = 0.25
WEIGHT_COVERAGE = 0.10

# Umbrales de confianza
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.50

# =============================================================================
# 13. TOKENIZER
# =============================================================================

# Model used for real token counting — should match EMBEDDING_MODEL so that
# token counts reflect actual BGE-M3 (XLM-RoBERTa) subword tokenisation.
TOKENIZER_MODEL_NAME: str = EMBEDDING_MODEL  # "BAAI/bge-m3"

# "real"   — use AutoTokenizer (lazy-loaded, cached, no full model weights)
# "approx" — always use len(text) // 4 heuristic (useful for offline envs
#             or when a faster but less accurate estimate is acceptable)
TOKEN_COUNTING_MODE: str = "real"

# =============================================================================
# CONFIGURACIÓN DE TESTS
# =============================================================================

TEST_ASSETS_DIR = BASE_DIR / "tests" / "assets"
