# RAG-Lab — Panel de Control de Parámetros

> **Propósito:** Documentación completa de todos los parámetros del sistema RAG y su impacto.
> **Última actualización:** 2026-04-28

---

## 1. Parámetros de Chunking

### `CHUNK_MAX_TOKENS`
- **Valor actual:** 800
- **Qué hace:** Tamaño máximo de cada chunk en tokens.
- **Impacto:**
  - Más alto → chunks más grandes, menos chunks totales, más contexto por chunk.
  - Más bajo → más chunks, más granularidad en búsqueda, más sobrecarga.
  - 800 es el equilibrio óptimo para BGE-M3 (límite 1024 tokens).

### `CHUNK_OVERLAP`
- **Valor actual:** 200
- **Qué hace:** Tokens que se repiten entre chunks consecutivos.
- **Impacto:**
  - Más alto → mejor contexto continuo, menos cortes bruscos.
  - Más bajo → menos duplicación, pero riesgo de perder contexto entre chunks.
  - 200 = 25% de superposición (estándar de la industria).

### `CHUNK_MIN_TOKENS`
- **Valor actual:** 50
- **Qué hace:** Umbral mínimo de tokens para considerar un chunk válido.
- **Impacto:**
  - Más alto → se filtran más chunks pequeños, se fusionan con vecinos.
  - Más bajo → se mantienen más chunks pequeños, más ruido potencial.
  - 50 filtra fragmentos sin perder contenido relevante.

---

## 2. Parámetros de Embedding

### `EMBEDDING_MODEL`
- **Valor actual:** "BAAI/bge-m3"
- **Qué hace:** Modelo de embeddings a usar.
- **Impacto:**
  - BGE-M3 soporta embeddings densos + dispersos simultáneamente.
  - Cambiar modelo requiere actualizar `EMBEDDING_MAX_LENGTH` según el modelo.

### `EMBEDDING_BATCH_SIZE`
- **Valor actual:** 4
- **Qué hace:** Número de textos procesados simultáneamente.
- **Impacto:**
  - 4 → más lento, menos memoria.
  - 8 → ~2x más rápido, ~2x más memoria (seguro en RTX 5090 con 32GB).
  - **Propuesto:** 8 (mejora throughput sin riesgo de OOM en GPU).

### `EMBEDDING_MAX_LENGTH`
- **Valor actual:** 1024
- **Qué hace:** Longitud máxima de entrada para el modelo de embedding.
- **Impacto:**
  - Debe coincidir con el límite del modelo (BGE-M3 = 1024).
  - No cambiar a menos que cambies el modelo.

### `EMBEDDING_DEVICE`
- **Valor actual:** "cuda" (producción), "cpu" (tests)
- **Qué hace:** Dispositivo donde se ejecuta el modelo de embedding.
- **Impacto:**
  - `cuda` → rápido, usa GPU.
  - `cpu` → lento, pero evita OOM en GPU (modo tests).
  - CLI: `--cpu-embedding` fuerza CPU para liberar VRAM.

---

## 3. Parámetros de Almacenamiento

### `VECTOR_STORE_PATH`
- **Valor actual:** `STORAGE_DIR / "chroma_db"`
- **Qué hace:** Ruta del almacén vectorial (ChromaDB).
- **Impacto:** Cambiar ruta mueve toda la base de datos.

### `SPARSE_INDEX_PATH`
- **Valor actual:** `STORAGE_DIR / "sparse_index.json"`
- **Qué hace:** Ruta del índice disperso (JSON).
- **Impacto:** Cambiar ruta mueve el índice disperso.

### `DOCDSTORE_SQLITE_PATH`
- **Valor actual:** `STORAGE_DIR / "docstore.sqlite"`
- **Qué hace:** Ruta del almacén de documentos (SQLite).
- **Impacto:** Cambiar ruta mueve la base de datos de documentos.

---

## 4. Parámetros de Recuperación

### `RETRIEVAL_TOP_K`
- **Valor actual:** 30
- **Qué hace:** Número de resultados de la búsqueda híbrida antes del reranking.
- **Impacto:**
  - Más alto → más candidatos para reranquear, más lento.
  - Más bajo → más rápido, pero riesgo de perder resultados relevantes.
  - 30 es suficiente para dar opciones al reranker.

### `RERANK_TOP_K`
- **Valor actual:** 8
- **Qué hace:** Resultados finales después del reranking.
- **Impacto:**
  - Más alto → más contexto al LLM, prompt más grande.
  - Más bajo → prompt más pequeño y rápido.
  - 8 es el equilibrio entre contexto y eficiencia.

### `RRF_K`
- **Valor actual:** 60
- **Qué hace:** Constante para fusión de rangos recíprocos (RRF).
- **Impacto:**
  - Más alto → suaviza diferencias entre resultados densos y dispersos.
  - Más bajo → más peso a resultados de rango alto.
  - 60 es el valor estándar para fusión híbrida.

---

## 5. Parámetros de Reranker

### `RERANKER_MODEL`
- **Valor actual:** "BAAI/bge-reranker-v2-m3"
- **Qué hace:** Modelo de reranking (cross-encoder).
- **Impacto:**
  - Mejora la relevancia de los resultados recuperados.
  - Más caro en cómputo, pero mejora la precisión.

### `RERANKER_DEVICE`
- **Valor actual:** "cuda" (producción), "cpu" (tests)
- **Qué hace:** Dispositivo donde se ejecuta el reranker.
- **Impacto:**
  - `cuda` → rápido.
  - `cpu` → lento, pero evita OOM en GPU (modo tests).
  - CLI: `--cpu-reranker` fuerza CPU para liberar VRAM.

---

## 6. Parámetros del LLM

### `LLM_BASE_URL`
- **Valor actual:** `http://localhost:8000/v1`
- **Qué hace:** URL del servidor LLM (compatible con OpenAI).
- **Impacto:** Cambiar URL apunta a otro servidor (ej. SGLang, LM Studio).

### `LLM_MODEL`
- **Valor actual:** "qwen3.6-35b-a3b@iq4_xs"
- **Qué hace:** Modelo LLM a usar.
- **Impacto:**
  - Qwen 3.6 35B es el modelo principal.
  - Cambiar modelo requiere ajustar `LLM_MAX_TOKENS` según el modelo.

### `LLM_TEMPERATURE`
- **Valor actual:** 0.1
- **Qué hace:** Nivel de aleatoriedad en la generación.
- **Impacto:**
  - 0.0 → determinista (mismo input = mismo output).
  - 0.1 → mínima variación (ideal para respuestas técnicas).
  - 0.5+ → más creatividad, menos precisión.

### `LLM_MAX_TOKENS`
- **Valor actual:** 2048
- **Qué hace:** Límite de tokens para la respuesta del LLM.
- **Impacto:**
  - Más alto → respuestas más largas, más lento y caro.
  - Más bajo → respuestas más cortas, más rápido.
  - 2048 es suficiente para respuestas completas.

---

## 7. Parámetros de Consulta

### `VARIANTS_COUNT`
- **Valor actual:** 2
- **Qué hace:** Número de variantes de consulta para expansión.
- **Impacto:**
  - Más alto → más cobertura de búsqueda, más lento.
  - Más bajo → solo consulta original, más rápido.
  - 2 = consulta original + 1 variante (equilibrio).

### `HYDE_ENABLED`
- **Valor actual:** False
- **Qué hace:** Activa/desactiva HyDE (Hypothetical Document Embeddings).
- **Impacto:**
  - True → genera respuesta hipotética para mejorar búsqueda.
  - False → búsqueda directa (más rápido).
  - Se activa con `--hyde` en CLI.

### `FAST_MODE`
- **Valor actual:** False
- **Qué hace:** Activa/desactiva modo rápido (sin reranking).
- **Impacto:**
  - True → omite reranking, más rápido pero menos preciso.
  - False → usa reranking, más lento pero más preciso.
  - Se activa con `--fast` en CLI.

---

## 8. Configuración de Dispositivos

### `EMBEDDING_DEVICE`
- **Producción:** "cuda"
- **Tests:** "cpu" (vía `conftest.py`)
- **CLI:** `--cpu-embedding` fuerza CPU para liberar VRAM.

### `RERANKER_DEVICE`
- **Producción:** "cuda"
- **Tests:** "cpu" (vía `conftest.py`)
- **CLI:** `--cpu-reranker` fuerza CPU para liberar VRAM.

---

## 9. Resumen de Cambios Propuestos

| Parámetro | Actual | Propuesto | Justificación |
|-----------|---------|-----------|----------------|
| `EMBEDDING_BATCH_SIZE` | 4 | 8 | ~2x más rápido en ingesta, seguro en RTX 5090 |

---

## 10. Cómo Modificar Parámetros

1. **Panel centralizado:** `rag_lab/config.py` — todos los parámetros viven aquí.
2. **Variables de entorno:** `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_DEVICE`, `RERANKER_DEVICE` se pueden sobrescribir con `.env`.
3. **CLI:** `--cpu-embedding`, `--cpu-reranker`, `--hyde`, `--fast` para ajustes en tiempo de ejecución.
4. **Tests:** `conftest.py` fuerza `EMBEDDING_DEVICE=cpu` y `RERANKER_DEVICE=cpu` para evitar OOM.

---

## 11. Notas de Rendimiento

- **GPU:** RTX 5090 (32GB VRAM). ~1GB libre normalmente.
- **CPU:** Modo CPU para tests evita OOM en GPU.
- **Throughput:** `EMBEDDING_BATCH_SIZE=8` mejora rendimiento sin riesgo en GPU.
- **Latencia:** `RERANK_TOP_K=8` mantiene prompts pequeños y rápidos.
