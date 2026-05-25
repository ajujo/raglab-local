# Guía de uso de RAG-Lab

Esta guía cubre el uso diario de RAG-Lab: ingesta de documentos, consultas, gestión del sistema y recuperación ante fallos.

---

## 1. Flujo básico diario

```bash
# Activar el entorno
conda activate rag-lab

# Hacer una consulta
rag-lab query "¿Qué es SDMX?"

# Chat interactivo
rag-lab chat
```

Eso es todo para el uso habitual. Los apartados siguientes describen cada área en detalle.

---

## 2. Ingestar documentos

El pipeline de ingesta tiene varias fases. Se recomienda seguir este orden para evitar problemas:

### Flujo completo de ingesta (recomendado)

```bash
# 1. Validar el frontmatter y la estructura del documento
rag-lab docs validate path/to/doc.md

# 2. Inspeccionar metadatos que se extraerán
rag-lab docs inspect path/to/doc.md

# 3. Previsualizar cómo se partirá el documento en chunks
rag-lab docs preview-chunks path/to/doc.md

# 4. Ingestar el documento
rag-lab ingest --doc path/to/doc.md

# 5. Verificar que se ha registrado correctamente
rag-lab docs show <doc_id>

# 6. Comprobar consistencia entre stores
rag-lab reconcile --check

# 7. Health check general
rag-lab doctor
```

### Opciones de ingesta

**Ingestar todos los documentos configurados en `SOURCES`:**

```bash
rag-lab ingest
```

Procesa todos los ficheros listados en `rag_lab/config.py` bajo `SOURCES`.

**Ingestar un solo documento:**

```bash
rag-lab ingest --doc path/to/doc.md
```

**Ingestar todos los `.md` de un directorio:**

```bash
rag-lab ingest --doc path/to/directory/
```

**Forzar re-ingesta aunque el documento ya esté ingestado:**

```bash
rag-lab ingest --doc path/to/doc.md --force
```

Por defecto, la ingesta es incremental: omite documentos cuyo hash no ha cambiado. Con `--force` se reprocesa siempre.

**Activar el modo estricto (warnings bloquean la ingesta):**

```bash
rag-lab ingest --doc path/to/doc.md --strict
```

Sin `--strict`, los warnings (por ejemplo, frontmatter incompleto) se registran pero no detienen el proceso. Con `--strict`, cualquier warning o error aborta la ingesta.

**Ingesta paralela con múltiples workers:**

```bash
rag-lab ingest --workers 4
```

Útil para ingestar directorios grandes. El número óptimo depende de la CPU y la memoria disponibles.

**Reanudar un batch incompleto:**

```bash
rag-lab ingest --resume
```

Si una ingesta masiva se interrumpió a mitad, `--resume` continúa desde donde se quedó.

**Reintentar documentos que fallaron en el último batch:**

```bash
rag-lab ingest --retry-failed
```

### Inspeccionar el historial de ingesta

```bash
# Listar batches recientes
rag-lab ingest batches

# Listar runs recientes
rag-lab ingest runs

# Filtrar runs por estado
rag-lab ingest runs --status FAILED

# Ver runs de un documento específico
rag-lab ingest runs --doc <doc_id>

# Ver todos los detalles de un run
rag-lab ingest show <run_id>
```

---

## 3. Consultar

### Uso básico

```bash
rag-lab query "¿Cuál es la diferencia entre un DataSet y un DataStructureDefinition?"
```

La consulta pasa por el pipeline completo: embedding, búsqueda híbrida (densa + dispersa), reranking y generación con el LLM. Se imprime la respuesta con las citas a los chunks fuente.

### Flags disponibles

| Flag | Descripción |
|---|---|
| `--hyde` | Genera una respuesta hipotética para expandir la consulta antes de recuperar. Experimental; los benchmarks no muestran mejora consistente. |
| `--rewrite` | Reescribe la consulta con terminología del dominio antes de recuperar. Experimental. |
| `--fast` | Omite el reranker. Más rápido, menor precisión. |
| `--top-k N` | Número de chunks que se pasan al LLM tras el reranking (por defecto: 8). |
| `--no-cache` | Ignora la caché de resultados. Útil para medir la latencia real. |
| `--profile` | Muestra los tiempos de cada fase del pipeline. |
| `--cpu-embedding` | Fuerza el embedding en CPU para esta sesión. |
| `--cpu-reranker` | Fuerza el reranking en CPU para esta sesión. |

### Cuándo usar cada flag

**`--hyde`:** HyDE (Hypothetical Document Embeddings) puede ayudar con consultas muy abstractas o cuando los documentos usan terminología distinta a la de la pregunta. Sin embargo, los benchmarks de este proyecto no muestran mejora consistente, por lo que está desactivado por defecto. Pruébalo si las respuestas sin él son muy pobres.

**`--rewrite`:** Útil para dominos técnicos donde la consulta del usuario puede no usar la nomenclatura exacta de los documentos. Añade latencia por la llamada adicional al LLM.

**`--fast`:** Indicado cuando la velocidad importa más que la precisión, por ejemplo en demostraciones o en una primera exploración del corpus. Sin reranker, los chunks pasados al LLM son simplemente los top-N de la fusión RRF.

**`--no-cache`:** Imprescindible cuando quieres medir latencias reales o cuando sospechas que la caché tiene resultados obsoletos.

**`--profile`:** Útil para diagnosticar cuellos de botella (embedding lento, reranker lento, LLM lento).

### Ejemplos

```bash
# Consulta simple
rag-lab query "¿Qué es un Code List en SDMX?"

# Con reranking desactivado (rápido)
rag-lab query "Explica los niveles de agregación" --fast

# Midiendo tiempos
rag-lab query "¿Cómo se define un concepto?" --profile

# Forzando 12 chunks al LLM
rag-lab query "Diferencias entre SDMX 2.1 y 3.0" --top-k 12

# Sin caché, para medir latencia real
rag-lab query "¿Qué es un dataflow?" --no-cache
```

---

## 4. Chat interactivo

```bash
rag-lab chat
```

Lanza un chat interactivo donde puedes hacer múltiples preguntas en secuencia. El chat mantiene contexto de documentos seleccionados durante la sesión.

Dentro del chat puedes usar comandos especiales con el prefijo `/`:

- `/docs` — muestra y permite seleccionar qué documentos filtrar
- `/mode` — cambia el modo de consulta (por ejemplo, activar/desactivar HyDE)
- `/quit` o `/exit` — salir del chat

---

## 5. Gestión de documentos

### Listar documentos

```bash
# Todos los documentos
rag-lab docs list

# Filtrar por etiqueta
rag-lab docs list --tag sdmx

# Filtrar por estado
rag-lab docs list --status active
```

### Ver detalles de un documento

```bash
rag-lab docs show mi_doc_id
```

Muestra metadatos, etiquetas, número de chunks, fechas de ingesta, etc.

### Validar un documento antes de ingestar

```bash
rag-lab docs validate path/to/doc.md

# En modo estricto (warnings también son errores)
rag-lab docs validate path/to/doc.md --strict
```

Comprueba el frontmatter YAML, campos obligatorios, campos prohibidos y estructura general del Markdown.

### Inspeccionar metadatos extraídos

```bash
rag-lab docs inspect path/to/doc.md
```

Muestra los metadatos que se extraerán del frontmatter sin llegar a ingestar el documento.

### Previsualizar chunks

```bash
rag-lab docs preview-chunks path/to/doc.md

# Limitar el número de chunks mostrados
rag-lab docs preview-chunks path/to/doc.md --limit 10
```

Muy útil para verificar que el documento se partirá correctamente antes de ingestarlo.

### Etiquetar y des-etiquetar

```bash
rag-lab docs tag mi_doc_id sdmx
rag-lab docs untag mi_doc_id sdmx
```

### Eliminar un documento

```bash
# Con confirmación interactiva
rag-lab docs delete mi_doc_id

# Sin confirmación
rag-lab docs delete mi_doc_id --force
```

Elimina el documento de todos los stores (ChromaDB, SQLite, índice disperso).

### Asociar a una fuente

```bash
rag-lab docs set-source mi_doc_id mi_source_id
```

---

## 6. Etiquetas

Las etiquetas permiten organizar documentos y filtrar consultas por subconjunto del corpus.

### Listar todas las etiquetas

```bash
rag-lab tags list
```

### Renombrar una etiqueta

```bash
rag-lab tags rename sdmx-v2 sdmx
```

Renombra la etiqueta en todos los documentos que la tengan.

### Eliminar una etiqueta

```bash
# Con confirmación
rag-lab tags delete obsoleta

# Sin confirmación
rag-lab tags delete obsoleta --force
```

### Etiquetas generadas automáticamente desde el frontmatter

Cuando un documento tiene frontmatter con los campos recomendados, se generan etiquetas automáticamente:

| Campo en frontmatter | Etiqueta generada |
|---|---|
| `domain: sdmx` | `domain:sdmx` |
| `source_type: manual` | `source_type:manual` |
| `language: en` | `lang:en` |
| `version: "2.1"` | `version:2.1` |

---

## 7. Caché de queries

RAG-Lab cachea los resultados de las consultas para evitar repetir el pipeline completo en preguntas idénticas.

### Qué se cachea

- Vectores de embedding de la consulta
- Resultados de retrieval (antes del reranking)
- Respuesta final del LLM

### Qué no se cachea

- Ingesta de documentos
- Resultados de `doctor` y `reconcile`
- Feedback

### Comandos

```bash
# Ver estadísticas de la caché (tamaño, entradas, hit rate)
rag-lab cache stats

# Ver el contenido de una entrada concreta
rag-lab cache inspect <key>

# Limpiar toda la caché
rag-lab cache clear

# Compactar la caché (libera espacio en disco sin borrar entradas válidas)
rag-lab cache vacuum
```

### Cuándo invalidar la caché

Después de ingestar nuevos documentos o de borrar documentos existentes, la caché puede devolver resultados desactualizados. Usa `rag-lab cache clear` o pasa `--no-cache` a las consultas siguientes hasta que confíes en que la caché refleja el estado actual del corpus.

---

## 8. Feedback

El feedback permite registrar la utilidad de los resultados para análisis posterior. **No afecta al ranking ni al comportamiento del sistema en tiempo real.**

### Añadir feedback

```bash
rag-lab feedback add --query "¿Qué es un dataflow?" --chunk-id "chunk_abc123" --feedback relevant
```

### Tipos de feedback disponibles

| Tipo | Significado |
|---|---|
| `relevant` | El chunk recuperado es relevante para la consulta |
| `irrelevant` | El chunk recuperado no es relevante |
| `useful` | La respuesta generada es útil |
| `not_useful` | La respuesta generada no es útil |
| `wrong_doc` | Se recuperó un documento incorrecto |
| `outdated` | La información del chunk está desactualizada |
| `duplicate` | El chunk es duplicado de otro ya recuperado |
| `bad_citation` | La cita al chunk en la respuesta es incorrecta |

### Consultar y exportar feedback

```bash
# Listar las últimas 20 entradas
rag-lab feedback list

# Listar más entradas
rag-lab feedback list --limit 50

# Filtrar por tipo
rag-lab feedback list --feedback irrelevant

# Ver estadísticas agregadas
rag-lab feedback stats

# Exportar a JSON
rag-lab feedback export

# Exportar a un fichero específico
rag-lab feedback export --output feedback_export.json
```

### Limpiar el feedback

```bash
rag-lab feedback clear --yes
```

---

## 9. Doctor

`rag-lab doctor` es el health check del sistema. Comprueba que todos los componentes están en buen estado.

### Cuándo ejecutarlo

- Después de instalar o actualizar RAG-Lab
- Después de un `rag-lab ingest` masivo
- Cuando una consulta devuelve resultados inesperados
- Periódicamente como mantenimiento preventivo

### Qué comprueba

| Check | Descripción |
|---|---|
| `config` | Variables de entorno y parámetros de configuración |
| `docstore` | Integridad del SQLite (chunks, metadatos) |
| `chromadb` | Estado de la colección vectorial |
| `fts5` | Índice de búsqueda de texto completo |
| `sparse_coverage` | Cobertura del índice disperso |
| `reconcile` | Inconsistencias entre stores |
| `ingest_health` | Estado de los runs de ingesta recientes |
| `test_query` | Ejecuta una consulta de prueba extremo a extremo |

### Ejecutar checks selectivos

```bash
# Solo verificar configuración y stores
rag-lab doctor --checks config,docstore,chromadb

# Ejecutar test de consulta con una pregunta específica
rag-lab doctor --query "¿Qué es SDMX?"
```

### Interpretar la salida

- `PASS` — componente en buen estado
- `WARN` — posible problema no crítico
- `FAIL` — problema que requiere atención

---

## 10. Reconcile

`rag-lab reconcile` detecta y repara inconsistencias entre los distintos stores (ChromaDB, SQLite, índice disperso, FTS5).

### Cuándo ejecutarlo

- Después de cualquier `rag-lab ingest` masivo
- Si `doctor` reporta inconsistencias entre stores
- Si borras documentos manualmente (sin usar el CLI)
- Como paso de mantenimiento periódico

### Modos

```bash
# Solo detectar problemas, no reparar
rag-lab reconcile --check

# Eliminar huérfanos en ChromaDB (chunks sin entrada en docstore)
rag-lab reconcile --repair

# Reparar duplicados en FTS5
rag-lab reconcile --repair-fts

# Rellenar metadatos NULL en el modelo (back-fill)
rag-lab reconcile --repair-metadata

# Guardar el informe en JSON
rag-lab reconcile --check --report-json reconcile_report.json
```

Se puede combinar varios flags:

```bash
rag-lab reconcile --repair --repair-fts --repair-metadata
```

---

## 11. Diagnose

`rag-lab diagnose` va más allá de `doctor`: permite analizar en detalle qué está recuperando el pipeline para una consulta concreta.

### Cuándo usar diagnose vs doctor

- **`doctor`** — comprueba el estado general del sistema (stores, config, conectividad).
- **`diagnose`** — depura por qué una consulta específica devuelve buenos o malos resultados.

### Uso

```bash
# Diagnosticar una consulta
rag-lab diagnose --query "¿Qué es un dataflow SDMX?"

# Con explicación detallada del retrieval
rag-lab diagnose --query "¿Qué es un dataflow?" --explain

# Limitar a documentos de un doc_id específico
rag-lab diagnose --query "Explica la estructura" --doc-id mi_doc_id

# Filtrar por etiqueta
rag-lab diagnose --query "Conceptos básicos" --tag sdmx

# Excluir documentos con una etiqueta
rag-lab diagnose --query "Conceptos básicos" --exclude-tag borrador
```

`--explain` muestra los scores de cada chunk recuperado, el motivo de su posición en el ranking y qué fase del pipeline lo seleccionó.

---

## 12. Benchmark

`rag-lab benchmark` evalúa el rendimiento del pipeline con un conjunto de consultas y respuestas de referencia.

### Cuándo ejecutarlo

- Al cambiar parámetros de retrieval (`RETRIEVAL_TOP_K`, `RRF_K`, `RERANK_TOP_K`)
- Al cambiar el modelo de embedding o reranker
- Al habilitar o deshabilitar HyDE o rewrite
- Para comparar variantes del pipeline antes de un cambio importante

### Uso

```bash
# Suite oficial con todas las variantes
rag-lab benchmark --suite official --variants full

# Solo retrieval denso
rag-lab benchmark --suite official --variants dense

# Sin caché (resultados reproducibles)
rag-lab benchmark --suite official --variants full --no-cache

# Guardar los resultados en un fichero
rag-lab benchmark --suite official --variants full --output resultados.json
```

### Suites disponibles

| Suite | Descripción |
|---|---|
| `official` | Conjunto de consultas de referencia del proyecto |
| `candidates` | Consultas candidatas en evaluación |
| `all` | Todas las suites combinadas |

### Variantes del pipeline

| Variante | Descripción |
|---|---|
| `full` | Pipeline completo: embedding denso + disperso + reranker |
| `dense` | Solo embedding denso |
| `bm25` | Solo índice disperso (BM25-like) |
| `hybrid` | Denso + disperso sin reranker |

### Interpretar los resultados

Los benchmarks reportan métricas como nDCG@10 y R@5. Valores más altos son mejores. Compara la variante que estás evaluando con el baseline de referencia (actualmente v1.11) para saber si el cambio supone una mejora real.

---

## 13. Frontmatter YAML

El frontmatter YAML es el mecanismo principal para asociar metadatos a un documento antes de ingestarlo. Un frontmatter bien construido mejora la calidad de las búsquedas y el filtrado por etiqueta.

### Estructura completa

```yaml
---
doc_id: mi_doc_id          # OBLIGATORIO — identificador único del documento
title: Título del documento # Recomendado
domain: sdmx               # Recomendado — genera etiqueta domain:sdmx
source_type: manual        # Recomendado — genera etiqueta source_type:manual
language: en               # Recomendado — genera etiqueta lang:en
version: "2.1"             # Opcional — genera etiqueta version:2.1
tags:
  - sdmx
  - technical_notes
---
```

### Campos obligatorios

| Campo | Tipo | Descripción |
|---|---|---|
| `doc_id` | string | Identificador único. Sin espacios. Debe ser estable. |

### Campos recomendados

| Campo | Tipo | Descripción |
|---|---|---|
| `title` | string | Título legible del documento |
| `domain` | string | Dominio temático. Genera etiqueta `domain:<valor>` |
| `source_type` | string | Tipo de fuente (`manual`, `spec`, `guide`, etc.). Genera etiqueta `source_type:<valor>` |
| `language` | string | Código de idioma ISO 639-1 (`en`, `es`, etc.). Genera etiqueta `lang:<valor>` |
| `version` | string | Versión del documento. Genera etiqueta `version:<valor>` |
| `tags` | lista | Etiquetas adicionales libres |

### Campos prohibidos

| Campo | Motivo |
|---|---|
| `dataset` | Causa ERROR al validar |
| `dataset_id` | Causa ERROR al validar |

RAG-Lab es para documentos Markdown narrativos, no para conjuntos de datos tabulares.

### Validar antes de ingestar

```bash
rag-lab docs validate path/to/doc.md
```

Si el frontmatter es incorrecto, la validación describe exactamente qué campo falta o está mal.

---

## 14. Flujo de recuperación tras un fallo de ingesta

Si una ingesta falla a mitad o produce resultados incorrectos, sigue estos pasos:

### 1. Identificar el run fallido

```bash
rag-lab ingest runs --status FAILED
rag-lab ingest show <run_id>
```

### 2. Hacer rollback del run

```bash
rag-lab ingest rollback <run_id>
```

El rollback elimina los chunks y vectores parcialmente ingesta dos de ese run de todos los stores.

### 3. Verificar que el rollback fue limpio

```bash
rag-lab reconcile --check
```

Si `reconcile` reporta huérfanos, repáralos:

```bash
rag-lab reconcile --repair
```

### 4. Corregir el documento y re-ingestar

```bash
# Validar primero para asegurarse de que el problema está resuelto
rag-lab docs validate path/to/doc.md

# Re-ingestar
rag-lab ingest --doc path/to/doc.md --force
```

### 5. Reintentar un run fallido directamente

Si el fallo fue transitorio (por ejemplo, el LLM no estaba disponible) y el documento en sí está correcto:

```bash
rag-lab ingest retry <run_id>
```

### 6. Verificar el estado final

```bash
rag-lab docs show <doc_id>
rag-lab reconcile --check
rag-lab doctor
```

---

## Documentos no soportados

RAG-Lab **solo procesa ficheros Markdown (`.md`)**. Los siguientes formatos no están soportados:

- PDF (no hay extracción de texto)
- DOCX / ODT
- HTML
- CSV, JSON, Excel u otros formatos de datos tabulares

Para usar un documento en otro formato, conviértelo a Markdown primero, añade el frontmatter correspondiente y luego ingestalo.
