# RAG-Lab Update 1.2

## Descripción
Nueva fase de desarrollo para el sistema RAG-Lab. Este documento registra las mejoras, correcciones y nuevas funcionalidades que se implementaron en esta versión.

## Tareas Completadas
- [x] Bloque 1: Estructura Base y Configuración
- [x] Bloque 2: Verifier Layer (Componente 1)
- [x] Bloque 3: Self-consistency Check (Componente 2)
- [x] Bloque 4: Scoring y Pipeline (Componente 3 y Orquestación)
- [x] Integración en CLI y modo chat
- [x] Actualización de documentación (README.md y update1-2.md)
- [x] Mejora 1: Conectar HyDE al LLM real
- [x] Mejora 2: Score de relevancia por chunk visible
- [x] Mejora 3: Query rewriting
- [x] Mejora 4: Feedback loop mínimo

## Mejoras Implementadas

### Mejora 1 — HyDE con LLM real
- **Archivos modificados**: `rag_lab/retrieval/query_processor.py`
- **Descripción**: Se reemplazó la plantilla fija de HyDE por una llamada real al LLM usando un prompt técnico especializado.
- **Log**: `HyDE: hipótesis generada (188 tokens) para query: "¿Qué es SDMX?..."`
- **Commit**: `b37b8a2`

### Mejora 2 — Score de relevancia por chunk visible
- **Archivos modificados**: `rag_lab/verification/pipeline.py`
- **Descripción**: Se añadió una sección de fragmentos recuperados en el bloque de verificación, mostrando el score individual de cada chunk con barras visuales (`██████████░░░░░`).
- **Advertencia**: Si algún score < 0.60, se muestra `⚠ Algunos fragmentos tienen relevancia baja. Considera reformular la pregunta o activar HyDE con --hyde para mejorar la recuperación.`
- **Commit**: `5111e8c`

### Mejora 3 — Query rewriting
- **Archivos nuevos**: `rag_lab/retrieval/query_rewriter.py`
- **Archivos modificados**: `rag_lab/retrieval/query_processor.py`, `rag_lab/config.py`, `rag_lab/cli.py`
- **Descripción**: Se añadió un módulo de rewriting que reformula las preguntas del usuario para maximizar la recuperación semántica (expansión de siglas, terminología técnica).
- **Flag CLI**: `--rewrite`
- **Log**: `QueryRewriter: "¿Qué es DSD?" → "¿Qué es la Definición de Estructura de Datos (DSD)?"`
- **Commit**: `ea16178`

### Mejora 4 — Feedback loop mínimo
- **Archivos nuevos**: `rag_lab/feedback/feedback_store.py`, `rag_lab/feedback/analyze_feedback.py`, `rag_lab/feedback/__init__.py`
- **Archivos modificados**: `rag_lab/cli.py`
- **Descripción**: Se añadió un sistema de feedback con almacenamiento SQLite que permite al usuario calificar la utilidad de las respuestas. El script `analyze_feedback.py` genera un resumen estadístico.
- **Flag CLI**: `--no-feedback` para desactivar el prompt de feedback.
- **Comando de análisis**: `python -m rag_lab.feedback.analyze_feedback`
- **Commit**: `a4e3d9b`

## Pipeline de Consulta Completo
```
Pregunta original
      ↓
Query rewriting (si --rewrite)
      ↓
HyDE (si --hyde) — opera sobre la pregunta reescrita
      ↓
Query expansion (variantes)
      ↓
Embedding + búsqueda híbrida
      ↓
Reranking
      ↓
Generación LLM
      ↓
Verificación (citas + consistency + scoring)
      ↓
Feedback (si --no-feedback no está activo)
```

## Plan de Implementación: Próximas Mejoras (Update 1.3)

Este documento contiene el plan detallado para tres líneas de mejora:
1. Mejorar el modo chat (HyDE, rewriting, feedback)
2. Métricas de rendimiento (timer, reporting)
3. Multi-document (múltiples fuentes, filtrado)

---

### Opción 1 — Mejorar el modo chat

**Objetivo:** Integrar HyDE, query rewriting y feedback loop en el modo chat interactivo.

#### Paso 1: Estructura base de comandos de chat
- **Archivos:** `rag_lab/cli_chat.py`
- **Acciones:**
  - Añadir método `handle_command(cmd, args)` que soporte:
    - `/hyde [on|off]` — Activar/desactivar HyDE
    - `/rewrite [on|off]` — Activar/desactivar query rewriting
    - `/feedback [on|off]` — Activar/desactivar prompt de feedback
    - `/docs [doc1,doc2]` — Filtrar por documento (para multi-doc)
    - `/mode [fast|standard|hyde]` — Cambiar modo de consulta
  - Almacenar estado en atributos de la clase `ChatSession`: `self.hyde_enabled`, `self.rewrite_enabled`, `self.feedback_enabled`, `self.active_docs`
- **Tests:** `tests/test_cli/test_chat_commands.py` — 4 tests para cada comando

#### Paso 2: Integración de HyDE en chat
- **Archivos:** `rag_lab/cli_chat.py`, `rag_lab/retrieval/query_processor.py`
- **Acciones:**
  - Cuando `self.hyde_enabled == True`, pasar `use_hyde=True` a `process_query()`
  - Asegurar que el LLM genera hipótesis técnicas reales (ya implementado en Mejora 1)
  - Registrar en el log: `Chat: HyDE activado/desactivado`
- **Tests:** Verificar que `process_query` recibe el flag correcto

#### Paso 3: Integración de query rewriting en chat
- **Archivos:** `rag_lab/cli_chat.py`, `rag_lab/retrieval/query_processor.py`
- **Acciones:**
  - Cuando `self.rewrite_enabled == True`, pasar `use_rewriting=True` a `process_query()`
  - Capturar la pregunta reescrita para guardarla en feedback (`rewritten_query`)
  - Registrar en el log: `Chat: Query rewriting activado/desactivado`
- **Tests:** Verificar que el rewriting se aplica antes de HyDE

#### Paso 4: Integración de feedback en chat
- **Archivos:** `rag_lab/cli_chat.py`, `rag_lab/feedback/feedback_store.py`
- **Acciones:**
  - Después de cada respuesta, si `self.feedback_enabled == True`, mostrar prompt `¿Útil? [s/n]`
  - Guardar entrada en SQLite con metadatos completos (question, rewritten_query, hyde_used, chunks, score, level)
  - Soportar `/feedback off` para desactivar el prompt
- **Tests:** `tests/test_cli/test_chat_feedback.py` — 3 tests para flujo de feedback

#### Paso 5: Estado persistente entre turnos
- **Archivos:** `rag_lab/cli_chat.py`
- **Acciones:**
  - Mantener `self.hyde_enabled`, `self.rewrite_enabled`, `self.feedback_enabled` entre consultas
  - Mostrar estado actual al iniciar sesión: `Modo: HyDE=ON, Rewriting=OFF, Feedback=ON`
- **Tests:** Verificar que el estado se mantiene tras múltiples turnos

**Criterio de éxito:** El modo chat soporta `/hyde on`, `/rewrite on`, `/feedback on/off`, y el estado se mantiene entre turnos.

---

### Opción 2 — Métricas de rendimiento

**Objetivo:** Medir y visualizar la latencia de cada fase del pipeline.

#### Paso 1: Timer utility
- **Archivos:** Nuevo `rag_lab/performance/timer.py`
- **Acciones:**
  - Crear clase `PhaseTimer` con métodos:
    - `start(phase_name: str)` — Iniciar cronómetro para una fase
    - `stop()` — Parar cronómetro
    - `get_duration() -> float` — Duración en segundos
    - `get_all_durations() -> dict[str, float]` — Todos los tiempos registrados
  - Soportar anidación de timers (ej. embedding dentro de retrieval)
- **Tests:** `tests/test_performance/test_timer.py` — 5 tests para funcionalidad del timer

#### Paso 2: Instrumentar CLI
- **Archivos:** `rag_lab/cli.py`
- **Acciones:**
  - Envolver cada fase del pipeline con `PhaseTimer`:
    - `embedding` — `encode_chunks()`
    - `hybrid_search` — `hybrid_search()`
    - `reranking` — `rerank()`
    - `llm_generation` — `generate_response()`
    - `verification` — `verify_and_score()`
    - `hyde` — `_generate_hypothetical_answer()` (si está activo)
    - `query_rewriting` — `rewrite_query()` (si está activo)
  - Al final, imprimir tabla de métricas con barras visuales
- **Tests:** Verificar que los timers se ejecutan sin romper el pipeline

#### Paso 3: Report de rendimiento
- **Archivos:** Nuevo `rag_lab/performance/report.py`
- **Acciones:**
  - Crear función `generate_report(durations: dict) -> str`
  - Formatear salida con barras visuales y totales
  - Soportar guardado en archivo JSON para análisis histórico
  - Calcular percentiles (p50, p95, p99) si hay historial
- **Tests:** `tests/test_performance/test_report.py` — 4 tests para formateo

#### Paso 4: Integración en CLI
- **Archivos:** `rag_lab/cli.py`
- **Acciones:**
  - Añadir flag `--profile` para activar el reporte de métricas
  - Imprimir tabla al final de cada consulta cuando `--profile` está activo
  - Guardar métricas en `performance/metrics.json` para análisis acumulativo
- **Tests:** Verificar que `--profile` no afecta el funcionamiento normal

**Criterio de éxito:** Con `--profile`, se imprime una tabla con tiempos por fase, y los datos se guardan para análisis histórico.

---

### Opción 3 — Multi-document

**Objetivo:** Soportar múltiples documentos fuente con filtrado por documento.

#### Paso 1: Estructura de configuración multi-doc
- **Archivos:** `rag_lab/config.py`
- **Acciones:**
  - Expandir `SOURCES` para soportar múltiples rutas
  - Añadir `ACTIVE_DOCS: list[str] = []` para filtrado por documento
  - Añadir `MULTI_DOC_ENABLED: bool = False` como flag de configuración
- **Tests:** Verificar que `SOURCES` acepta múltiples entradas

#### Paso 2: Ingesta multi-document
- **Archivos:** `rag_lab/ingest/cleaner.py`, `rag_lab/ingest/manifest.py`
- **Acciones:**
  - Modificar `ingest()` para procesar todos los documentos en `SOURCES`
  - Cada chunk debe llevar `doc_id` único por documento
  - Generar `ingested.jsonl` con una entrada por documento
  - Soportar `--doc <path>` para ingesta de un solo documento
- **Tests:** `tests/test_ingest/test_multi_doc.py` — 4 tests para ingesta múltiple

#### Paso 3: Chunking con doc_id
- **Archivos:** `rag_lab/chunking/splitter.py`
- **Acciones:**
  - Asegurar que `chunk_document()` acepta `doc_id` como parámetro
  - Cada chunk generado incluye `doc_id` en sus metadatos
  - Mantener `line_start` y `line_end` precisos por documento
- **Tests:** Verificar que los chunks de diferentes documentos no se mezclan

#### Paso 4: Búsqueda multi-document
- **Archivos:** `rag_lab/retrieval/hybrid_search.py`, `rag_lab/storage/vector_store.py`
- **Acciones:**
  - La búsqueda híbrida ya soporta múltiples `doc_id` en metadatos
  - Añadir filtrado por `doc_id` en ChromaDB: `where={"doc_id": "doc1"}`
  - Soportar búsqueda en subconjunto de documentos activos
- **Tests:** `tests/test_retrieval/test_multi_doc_search.py` — 3 tests para filtrado

#### Paso 5: Citas multi-document
- **Archivos:** `rag_lab/verification/verifier.py`
- **Acciones:**
  - El verificador ya soporta múltiples `doc_id` en las citas
  - Asegurar que `verify_citations_layer()` busca en todos los documentos
  - Mantener formato de citas: `[[N] Fuente: <doc_id> | Sección: ... | Líneas: ...]`
- **Tests:** Verificar que las citas de diferentes documentos se validan correctamente

#### Paso 6: Filtrado en modo chat
- **Archivos:** `rag_lab/cli_chat.py`
- **Acciones:**
  - Añadir comando `/docs <doc1,doc2>` para filtrar por documento
  - Almacenar `self.active_docs` y pasar a `hybrid_search()` como filtro
  - Mostrar documentos activos al iniciar sesión
- **Tests:** `tests/test_cli/test_chat_docs.py` — 3 tests para filtrado

#### Paso 7: Documentación multi-doc
- **Archivos:** `MULTI_DOC.md`, `README.md`, `AGENTS.md`
- **Acciones:**
  - Actualizar `MULTI_DOC.md` con instrucciones de uso
  - Documentar cómo agregar nuevos documentos a `SOURCES`
  - Explicar el comando `/docs` en modo chat
- **Tests:** N/A (documentación)

**Criterio de éxito:** El sistema puede ingestar, buscar y citar de múltiples documentos, con filtrado por documento en modo chat.

---

## Orden recomendado de implementación

1. **Opción 1 (Chat)** — Mayor impacto en UX, reutiliza código ya existente
2. **Opción 2 (Métricas)** — Baja complejidad, alto valor para debugging
3. **Opción 3 (Multi-doc)** — Mayor complejidad, requiere cambios en múltiples capas

## Resumen de archivos a crear/modificar

| Opción | Archivos nuevos | Archivos a modificar | Tests nuevos |
|---------|-----------------|----------------------|----------------|
| 1. Chat | 0 | `cli_chat.py` | 7 tests |
| 2. Métricas | 2 (`timer.py`, `report.py`) | `cli.py` | 9 tests |
| 3. Multi-doc | 0 | `config.py`, `cleaner.py`, `manifest.py`, `splitter.py`, `hybrid_search.py`, `verifier.py`, `cli_chat.py`, `MULTI_DOC.md` | 10 tests |

## Update 1.3 — Cierre

### Mejoras Implementadas en 1.3

| Mejora | Archivos | Commit |
|---------|-----------|---------|
| Mejorar modo chat | `cli_chat.py` | `8d1dfd8` |
| Métricas de rendimiento | `performance/timer.py`, `performance/report.py`, `cli.py` | `38784ae` |
| Multi-document | `config.py`, `vector_store.py`, `hybrid_search.py`, `cli_chat.py` | `5e9e8b9` |

### Comandos de Chat
| Comando | Uso |
|---------|-----|
| `/hyde [on|off]` | Activar/desactivar HyDE |
| `/rewrite [on|off]` | Activar/desactivar query rewriting |
| `/feedback [on|off]` | Activar/desactivar feedback |
| `/docs [doc1,doc2]` | Filtrar por documento |
| `/mode [fast|standard|hyde]` | Cambiar modo |
| `/temp <valor>` | Cambiar temperatura |
| `/topk <n>` | Cambiar top-k |

### Flags CLI
| Flag | Propósito |
|------|-----------|
| `--hyde` | Activar HyDE |
| `--rewrite` | Activar query rewriting |
| `--profile` | Mostrar métricas de rendimiento |
| `--no-feedback` | Desactivar prompt de feedback |
| `--cpu-embedding` | Ejecutar embedding en CPU |
| `--cpu-reranker` | Ejecutar reranker en CPU |
| `--fast` | Modo rápido (sin reranking) |
| `--top-k <n>` | Número de chunks |

### Flujo de Trabajo
1. **Ingesta**: `python -m rag_lab.cli ingest` (o `--doc <archivo>` para un solo documento)
2. **Consulta**: `python -m rag_lab.cli query "Pregunta" --hyde --rewrite --profile`
3. **Chat**: `python -m rag_lab.cli chat` → comandos `/hyde on`, `/docs doc1,doc2`
4. **Feedback**: `python -m rag_lab.feedback.analyze_feedback`

### Notas
- Fecha de cierre: 2026-04-30
- Total commits 1.3: 3 (8d1dfd8, 38784ae, 5e9e8b9)
- Tests totales: 14 (chat) + 16 (performance) + 4 (multi-doc) + 20 (verification) + 6 (query_rewriter) + 7 (feedback) = **67 tests**
