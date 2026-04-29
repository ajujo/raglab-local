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

Si la respuesta no está en los fragmentos, indícalo explícitamente.
"""

LOG_LEVEL = "INFO"
LOG_FILE = "rag_lab.log"

# =============================================================================
# CONFIGURACIÓN DE TESTS
# =============================================================================

TEST_ASSETS_DIR = BASE_DIR / "tests" / "assets"
