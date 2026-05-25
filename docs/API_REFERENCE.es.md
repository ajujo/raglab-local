# RAG-Lab — Referencia de CLI (v1.19.1)

Referencia completa de todos los comandos disponibles en `rag-lab`. Todos los comandos asumen
que el entorno conda `rag-lab` está activo y el servidor LLM local está disponible para los
comandos que generan respuestas.

---

## Tabla de contenidos

- [query](#query)
- [chat](#chat)
- [ingest](#ingest)
- [docs](#docs)
- [tags](#tags)
- [cache](#cache)
- [feedback](#feedback)
- [doctor](#doctor)
- [reconcile](#reconcile)
- [diagnose](#diagnose)
- [benchmark](#benchmark)

---

## query

Ejecuta una consulta sobre el corpus ingestado y devuelve una respuesta generada con contexto.

```
rag-lab query "pregunta" [opciones]
```

### Opciones

| Opción | Descripción |
|---|---|
| `--hyde` | Activa HyDE (Hypothetical Document Embeddings). **Desactivado por defecto.** Benchmarked: R@5 -3.8pp, latencia ×12.5. Disponible para experimentación. |
| `--rewrite` | Activa query rewriting con terminología de dominio. **Desactivado por defecto.** Sin benchmark oficial. |
| `--fast` | Modo rápido: desactiva el reranker y la verificación. Menor calidad, menor latencia. |
| `--top-k N` | Número de chunks a recuperar antes del reranking. Por defecto: valor de `RETRIEVAL_TOP_K` en config (30). |
| `--no-cache` | Ignora el caché de queries y fuerza re-ejecución completa del pipeline. |
| `--profile` | Muestra traza detallada del pipeline: retrieval scores, verification trace, trust score desglosado. |
| `--cpu-embedding` | Fuerza el modelo de embedding a usar CPU independientemente de la variable de entorno. |
| `--cpu-reranker` | Fuerza el reranker a usar CPU independientemente de la variable de entorno. |

### Ejemplos

```bash
# Consulta básica
rag-lab query "¿Qué es SDMX?"

# Con traza de pipeline completa
rag-lab query "¿Cómo funciona el RRF?" --profile

# Sin caché, sin reranker, rápido
rag-lab query "Definición de dimensión en SDMX" --no-cache --fast

# Recuperar más candidatos antes de reranking
rag-lab query "estructura de un Data Flow" --top-k 50
```

---

## chat

Inicia una sesión de chat interactiva con selección de documentos activos.

```
rag-lab chat
```

No tiene opciones de línea de comandos. El modo interactivo permite:
- Seleccionar qué documentos del corpus están activos en la sesión
- Hacer preguntas de forma conversacional
- Ver la respuesta con contexto de chunks recuperados

### Ejemplo

```bash
rag-lab chat
```

---

## ingest

Ingesta documentos en el corpus. Soporta ingesta individual, batch y reanudación.

### Ingesta básica

```
rag-lab ingest [opciones]
```

Sin `--doc`, ingesta todos los documentos configurados en `SOURCES` dentro de `config.py`.

| Opción | Descripción |
|---|---|
| `--doc PATH` | Ingesta un único documento en el path especificado. |
| `--force` | Reingesta el documento aunque ya exista en el corpus (reemplaza). |
| `--resume` | Continúa una ingesta interrumpida desde el último checkpoint. |
| `--retry-failed` | Reintenta automáticamente los documentos con estado `failed` del run más reciente. |
| `--workers N` | Número de workers paralelos para ingesta batch. Por defecto: 1 (secuencial). |
| `--strict` | Aplica validación estricta: los WARNs de Markdown bloquean la ingesta (como ERRORs). |

### Ejemplos

```bash
# Ingestar todos los documentos configurados
rag-lab ingest

# Ingestar un documento específico
rag-lab ingest --doc data/docs/sdmx_user_guide.md

# Reingestar forzando reemplazo
rag-lab ingest --doc data/docs/sdmx_user_guide.md --force

# Ingesta paralela con 4 workers y validación estricta
rag-lab ingest --workers 4 --strict

# Reintentar documentos fallidos del último run
rag-lab ingest --retry-failed
```

---

### ingest batches

Lista los batches de ingesta con filtros opcionales.

```
rag-lab ingest batches [--status STATUS] [--doc DOC]
```

| Opción | Descripción |
|---|---|
| `--status S` | Filtra por estado: `success`, `failed`, `partial`, `running`. |
| `--doc D` | Filtra por nombre o path de documento. |

```bash
# Ver todos los batches
rag-lab ingest batches

# Ver solo los fallidos
rag-lab ingest batches --status failed
```

---

### ingest runs

Lista los runs de ingesta con filtros opcionales.

```
rag-lab ingest runs [--status STATUS] [--doc DOC]
```

| Opción | Descripción |
|---|---|
| `--status S` | Filtra por estado: `success`, `failed`, `partial`, `running`. |
| `--doc D` | Filtra por nombre o path de documento. |

```bash
# Ver todos los runs
rag-lab ingest runs

# Ver runs fallidos
rag-lab ingest runs --status failed
```

---

### ingest show

Muestra el detalle de un run específico.

```
rag-lab ingest show <run_id>
```

```bash
rag-lab ingest show abc123def456
```

---

### ingest rollback

Deshace una ingesta completa, eliminando todos los chunks, vectores y registros de ese run.

```
rag-lab ingest rollback <run_id>
```

```bash
rag-lab ingest rollback abc123def456
```

**Nota:** El rollback es atómico. Elimina el documento de DocStore, ChromaDB, FTS5 y Sparse
BLOBs simultáneamente. No puede deshacerse.

---

### ingest retry

Reintenta los documentos fallidos de un run específico.

```
rag-lab ingest retry <run_id>
```

```bash
rag-lab ingest retry abc123def456
```

---

## docs

Gestión de documentos del corpus.

### docs list

Lista todos los documentos ingestados.

```
rag-lab docs list [--tag TAG]
```

| Opción | Descripción |
|---|---|
| `--tag TAG` | Filtra documentos que tengan el tag especificado. |

```bash
# Listar todos los documentos
rag-lab docs list

# Listar solo documentos con tag 'sdmx'
rag-lab docs list --tag sdmx
```

---

### docs show

Muestra la información completa de un documento: metadatos de clasificación, tags explícitos
y derivados, path, hash, timestamps y número de chunks.

```
rag-lab docs show <id>
```

`<id>` es el `doc_id` del documento (valor del campo `doc_id` en el frontmatter).

```bash
rag-lab docs show sdmx_user_guide_2_1
```

Salida incluye secciones:
- **Classification:** title, domain, source_type, language, version, tags explícitos, tags derivados
- **Technical:** path, hash, ingest timestamp, chunks count

---

### docs tag

Añade un tag a un documento.

```
rag-lab docs tag <id> <tag>
```

```bash
rag-lab docs tag sdmx_user_guide_2_1 reviewed
```

---

### docs untag

Elimina un tag de un documento.

```
rag-lab docs untag <id> <tag>
```

```bash
rag-lab docs untag sdmx_user_guide_2_1 draft
```

---

### docs delete

Elimina un documento del corpus (DocStore, ChromaDB, FTS5, Sparse BLOBs).

```
rag-lab docs delete <id> [--force]
```

| Opción | Descripción |
|---|---|
| `--force` | Elimina sin pedir confirmación. |

```bash
# Eliminar con confirmación
rag-lab docs delete sdmx_glossary_legacy

# Eliminar sin confirmación
rag-lab docs delete sdmx_glossary_legacy --force
```

---

### docs set-source

Asocia un documento a una fuente de datos.

```
rag-lab docs set-source <id> <src_id>
```

```bash
rag-lab docs set-source sdmx_user_guide_2_1 sdmx_official
```

---

### docs validate

Valida un fichero Markdown antes de ingestarlo. Comprueba frontmatter, estructura, headings,
tablas y otros criterios de calidad.

```
rag-lab docs validate <path> [--strict]
```

| Opción | Descripción |
|---|---|
| `--strict` | Trata los WARNs como ERRORs. Sale con código 1 si hay cualquier issue. |

Códigos de salida:
- `0`: OK (sin ERRORs; puede haber WARNs en modo normal)
- `1`: ERRORs presentes, o WARNs presentes en modo `--strict`

```bash
# Validación normal
rag-lab docs validate data/docs/sdmx_user_guide.md

# Validación estricta (WARNs bloquean)
rag-lab docs validate data/docs/sdmx_user_guide.md --strict
```

---

### docs inspect

Muestra la estructura completa de un documento sin ingestarlo: frontmatter parseado (incluidos
tags derivados), árbol de headings, estimación de tokens y chunks, y resultado de validación.

```
rag-lab docs inspect <path>
```

```bash
rag-lab docs inspect data/docs/sdmx_user_guide.md
```

---

### docs preview-chunks

Genera los chunks que se crearían al ingestar el documento, sin escribir nada en los stores.
Útil para auditar el chunking antes de comprometerse.

```
rag-lab docs preview-chunks <path> [--limit N]
```

| Opción | Descripción |
|---|---|
| `--limit N` | Muestra solo los primeros N chunks. Por defecto: todos. |

```bash
# Ver todos los chunks que se crearían
rag-lab docs preview-chunks data/docs/sdmx_user_guide.md

# Ver solo los 10 primeros
rag-lab docs preview-chunks data/docs/sdmx_user_guide.md --limit 10
```

---

## tags

Gestión del catálogo de tags.

### tags list

Lista todos los tags existentes en el corpus con el número de documentos asociados.

```
rag-lab tags list
```

```bash
rag-lab tags list
```

---

### tags rename

Renombra un tag en todos los documentos que lo tienen.

```
rag-lab tags rename <old> <new>
```

```bash
rag-lab tags rename draft reviewed
```

---

### tags delete

Elimina un tag de todos los documentos que lo tienen.

```
rag-lab tags delete <name> [--force]
```

| Opción | Descripción |
|---|---|
| `--force` | Elimina sin pedir confirmación. |

```bash
rag-lab tags delete obsolete --force
```

---

## cache

Gestión del caché de queries.

### cache stats

Muestra estadísticas del caché: número de entradas, tamaño, TTL, hit rate de la sesión.

```
rag-lab cache stats
```

```bash
rag-lab cache stats
```

---

### cache clear

Elimina todas las entradas del caché.

```
rag-lab cache clear
```

```bash
rag-lab cache clear
```

---

### cache vacuum

Elimina las entradas expiradas del caché (más de 7 días por defecto) y compacta la base de
datos SQLite.

```
rag-lab cache vacuum
```

```bash
rag-lab cache vacuum
```

---

### cache inspect

Muestra el contenido de una entrada del caché por su clave (fingerprint).

```
rag-lab cache inspect <key>
```

```bash
rag-lab cache inspect abc123
```

---

## feedback

Gestión del store de feedback de usuarios.

### feedback add

Añade un evento de feedback para una query y chunk específicos.

```
rag-lab feedback add --query "QUERY" --chunk-id "CHUNK_ID" --feedback TYPE
```

| Opción | Descripción |
|---|---|
| `--query TEXT` | Texto de la query original. Obligatorio. |
| `--chunk-id ID` | ID del chunk al que aplica el feedback. Obligatorio. |
| `--feedback TYPE` | Tipo de feedback. Valores: `relevant`, `irrelevant`, `useful`, `not_useful`, `wrong_doc`, `outdated`, `duplicate`, `bad_citation`. |

```bash
rag-lab feedback add \
  --query "¿Qué es un Data Flow en SDMX?" \
  --chunk-id "sdmx_user_guide_2_1:chunk_042" \
  --feedback useful
```

---

### feedback list

Lista los eventos de feedback registrados.

```
rag-lab feedback list [--limit N] [--feedback TYPE]
```

| Opción | Descripción |
|---|---|
| `--limit N` | Número máximo de eventos a mostrar. Por defecto: 20. |
| `--feedback TYPE` | Filtra por tipo de feedback. |

```bash
# Últimos 20 eventos
rag-lab feedback list

# Solo feedback negativo
rag-lab feedback list --feedback not_useful

# Últimos 100 eventos de citas incorrectas
rag-lab feedback list --limit 100 --feedback bad_citation
```

---

### feedback stats

Muestra estadísticas agregadas del feedback: distribución por tipo, documentos con más
feedback negativo, queries con más fallos.

```
rag-lab feedback stats
```

```bash
rag-lab feedback stats
```

---

### feedback export

Exporta todos los eventos de feedback a un fichero JSON o CSV.

```
rag-lab feedback export [--output PATH]
```

| Opción | Descripción |
|---|---|
| `--output PATH` | Path de salida. Por defecto: stdout en formato JSON. |

```bash
# Exportar a fichero
rag-lab feedback export --output feedback_export.json
```

---

### feedback clear

Elimina todos los eventos de feedback. Operación irreversible.

```
rag-lab feedback clear --yes
```

`--yes` es obligatorio para evitar borrado accidental.

```bash
rag-lab feedback clear --yes
```

---

## doctor

Ejecuta un health check completo del sistema y reporta el estado de cada componente.

```
rag-lab doctor [--checks CHECKS] [--query TEXT]
```

| Opción | Descripción |
|---|---|
| `--checks CHECKS` | Lista de checks a ejecutar separada por comas. Por defecto: todos. |
| `--query TEXT` | Query de prueba a usar en el check `test_query`. Por defecto: query predefinida. |

### Checks disponibles

| Check | Qué verifica |
|---|---|
| `config` | Variables de entorno, paths de configuración, conectividad con el LLM. |
| `docstore` | Integridad de la base de datos SQLite: tablas, índices, conteo de registros. |
| `chromadb` | Colección ChromaDB: existencia, número de vectores, consistencia con DocStore. |
| `fts5` | Tabla FTS5: existencia, número de filas, consistency con DocStore. |
| `sparse_coverage` | Cobertura de sparse BLOBs: qué porcentaje de chunks tienen vector sparse. |
| `reconcile` | Ejecuta una reconciliación rápida entre stores y reporta inconsistencias. |
| `ingest_health` | Estado de los runs de ingesta recientes: failed, partial, orphaned. |
| `test_query` | Ejecuta una query de prueba completa y verifica que produce resultado. |

### Ejemplos

```bash
# Health check completo
rag-lab doctor

# Solo checks de almacenamiento
rag-lab doctor --checks docstore,chromadb,fts5,sparse_coverage

# Con query de prueba personalizada
rag-lab doctor --query "¿Qué es SDMX?"
```

---

## reconcile

Verifica y repara inconsistencias entre los stores del sistema (DocStore, ChromaDB, FTS5,
Sparse BLOBs).

```
rag-lab reconcile [opciones]
```

| Opción | Descripción |
|---|---|
| `--check` | Solo reporta inconsistencias sin modificar nada. Código de salida 1 si hay problemas. |
| `--repair` | Elimina orphans de ChromaDB (chunks en ChromaDB sin entrada en DocStore). |
| `--repair-fts` | Corrige duplicados en la tabla FTS5. |
| `--repair-metadata` | Rellena campos NULL de metadatos de modelo (backfill). |
| `--report-json PATH` | Escribe el reporte de reconciliación en formato JSON al path especificado. |

### Ejemplos

```bash
# Solo verificar (sin modificar)
rag-lab reconcile --check

# Reparar orphans de ChromaDB
rag-lab reconcile --repair

# Reparar duplicados FTS5
rag-lab reconcile --repair-fts

# Reparar metadatos NULL
rag-lab reconcile --repair-metadata

# Verificar y guardar reporte
rag-lab reconcile --check --report-json reconcile_report.json
```

---

## diagnose

Ejecuta una query de diagnóstico con traza detallada del pipeline para investigar problemas
de retrieval o verificar el estado de un documento específico.

```
rag-lab diagnose [opciones]
```

| Opción | Descripción |
|---|---|
| `--query TEXT` | Query a ejecutar en el diagnóstico. |
| `--explain` | Muestra explicación detallada de cada fase del pipeline. |
| `--doc-id ID` | Filtra el retrieval a un documento específico. |
| `--tag TAG` | Filtra el retrieval a documentos con este tag. |
| `--exclude-tag TAG` | Excluye del retrieval documentos con este tag. |

### Ejemplos

```bash
# Diagnosticar una query con explicación completa
rag-lab diagnose --query "¿Qué es un Data Flow?" --explain

# Diagnosticar sobre un documento específico
rag-lab diagnose --query "estructura de un Data Flow" --doc-id sdmx_user_guide_2_1 --explain

# Diagnosticar filtrando por tag
rag-lab diagnose --query "conceptos básicos" --tag sdmx --explain

# Diagnosticar excluyendo documentos obsoletos
rag-lab diagnose --query "sintaxis SDMX" --exclude-tag obsolete --explain
```

---

## benchmark

Ejecuta el benchmark de retrieval sobre el corpus SDMX con el conjunto de queries de evaluación.

```
rag-lab benchmark [opciones]
rag-lab benchmark run [opciones]
```

`run` es un alias de `benchmark` — ambas formas son equivalentes.

| Opción | Descripción |
|---|---|
| `--suite SUITE` | Suite a ejecutar: `official`, `candidates`, `all`. Por defecto: `official`. |
| `--variants V [V...]` | Variantes a benchmarkear. Por defecto: `full`. |
| `--no-cache` | Desactiva el caché de queries durante el benchmark. |
| `--output PATH` | Escribe los resultados en el path especificado (JSON). |
| `--top-k N` | Número de chunks a recuperar en el benchmark. |
| `--rrf-k N` | Constante K del algoritmo RRF. |

### Suites disponibles

| Suite | Descripción |
|---|---|
| `official` | Conjunto oficial de queries de evaluación sobre el corpus SDMX. Referencia para comparaciones. |
| `candidates` | Conjunto de queries candidatas en evaluación. |
| `all` | Todas las suites. |

### Métricas reportadas

| Métrica | Valor actual (baseline v1.11) |
|---|---|
| R@5 | 0.821 |
| R@10 | 0.896 |
| R@30 | 0.978 |
| MRR | 0.939 |
| nDCG@10 | 0.837 |

### Ejemplos

```bash
# Benchmark completo con suite oficial
rag-lab benchmark --suite official --variants full --no-cache

# Benchmark con alias 'run'
rag-lab benchmark run --suite official --variants full --no-cache

# Benchmark con parámetros personalizados y guardado de resultados
rag-lab benchmark --suite official --top-k 50 --rrf-k 30 --output results.json

# Benchmark de todas las suites
rag-lab benchmark --suite all --no-cache
```

---

## Flujos de trabajo habituales

### Incorporar un documento nuevo

```bash
# 1. Validar calidad del Markdown
rag-lab docs validate path/to/doc.md

# 2. Inspeccionar frontmatter y estructura
rag-lab docs inspect path/to/doc.md

# 3. Ver chunks que se crearían
rag-lab docs preview-chunks path/to/doc.md

# 4. Ingestar
rag-lab ingest --doc path/to/doc.md

# 5. Verificar metadatos
rag-lab docs show <doc_id>

# 6. Confirmar integridad
rag-lab reconcile --check
rag-lab doctor --checks docstore,chromadb,fts5
```

### Investigar una query que no funciona bien

```bash
# Traza completa del pipeline con filtro de documento
rag-lab diagnose --query "tu pregunta" --explain

# Con documento específico
rag-lab diagnose --query "tu pregunta" --doc-id <doc_id> --explain

# Query normal con profile
rag-lab query "tu pregunta" --profile --no-cache
```

### Health check periódico

```bash
rag-lab doctor
rag-lab reconcile --check
```

### Verificar una mejora de retrieval

```bash
# Benchmark sin caché para medir cambio real
rag-lab benchmark --suite official --variants full --no-cache

# Comparar con baseline guardado
rag-lab benchmark --suite official --no-cache --output nuevo_resultado.json
```

---

*Versión: v1.19.1*
