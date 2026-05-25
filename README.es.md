# RAG-Lab — Guia completa (castellano)

## 1. Que es RAG-Lab

RAG-Lab es un sistema RAG (Retrieval-Augmented Generation) local y orientado a la linea de comandos. Su funcion es permitir consultas en lenguaje natural sobre un corpus de documentacion tecnica en formato Markdown, usando un modelo de lenguaje servido localmente. Todo el procesamiento ocurre en la maquina local: no hay llamadas a APIs externas, no se envian datos a ningun servicio en la nube.

El corpus actual esta formado por documentacion tecnica del estandar SDMX (Statistical Data and Metadata eXchange): notas tecnicas, glosario, guias de usuario y materiales de formacion. El sistema ingesta esos documentos, los divide en fragmentos semanticos, los embebe con BAAI/bge-m3, los indexa en ChromaDB y SQLite, y responde consultas combinando busqueda densa, BM25 y un reranker cross-encoder antes de pasar el contexto al LLM.

---

## 2. Que problema resuelve

La documentacion tecnica extensa — especificaciones de mas de mil paginas, glosarios, apendices — es dificil de consultar manualmente. RAG-Lab permite hacer preguntas en lenguaje natural y obtener respuestas contextualizadas con citas exactas (numero de chunk, lineas del documento fuente), sin necesidad de leer el documento completo. El sistema indica cuanto confia en su propia respuesta mediante una puntuacion de confianza.

---

## 3. Que NO es RAG-Lab

- No es un servicio web ni una API publica.
- No es un sistema multiusuario.
- No soporta ingesta directa de PDF, DOCX ni HTML. El corpus debe estar en Markdown limpio.
- No es una herramienta para datos tabulares (sin CSV, Parquet, DuckDB ni datasets).
- No es un producto SaaS ni esta pensado para despliegue en produccion compartida.
- No tiene interfaz grafica. Es exclusivamente CLI.

---

## 4. Estado actual

| Campo | Valor |
|-------|-------|
| Version | v1.19.1 |
| Estado | Estable, en uso local controlado |
| Tests | 1031 pasando |
| Corpus | 610 chunks en produccion |
| Python | 3.11 (conda env `rag-lab`) |
| GPU recomendada | RTX 5090 (modo CPU disponible) |

---

## 5. Instalacion rapida

```bash
# Crear y activar el entorno conda
conda create -n rag-lab python=3.11
conda activate rag-lab

# Instalar el paquete en modo editable (incluye la entrada rag-lab en PATH)
pip install -e .

# Configurar el entorno
cp .env.example .env
# Editar .env con los valores correctos:
#   LLM_BASE_URL=http://localhost:8000/v1
#   LLM_MODEL=nombre-del-modelo-local
#   EMBEDDING_DEVICE=cuda   # o cpu
#   RERANKER_DEVICE=cuda    # o cpu

# Verificar que todo esta en orden
rag-lab doctor
```

El LLM debe estar disponible como API compatible con OpenAI en la URL configurada. Si no hay GPU, se puede usar `EMBEDDING_DEVICE=cpu` y `RERANKER_DEVICE=cpu` a costa de mayor latencia.

---

## 6. Primer uso

```bash
# Validar un documento antes de ingestarlo
rag-lab docs validate path/to/doc.md

# Ingestar todos los documentos configurados en SOURCES
rag-lab ingest

# Hacer una consulta
rag-lab query "Que es un Data Structure Definition en SDMX?"
```

---

## 7. Comandos principales

### Consultas

| Comando | Descripcion |
|---------|-------------|
| `rag-lab query "pregunta"` | Consulta estandar |
| `rag-lab query "pregunta" --fast` | Omite el reranker (mas rapido, menor precision) |
| `rag-lab query "pregunta" --top-k N` | Cambia el numero de chunks finales |
| `rag-lab query "pregunta" --hyde` | Activa HyDE (desactivado por defecto; ver seccion 14) |
| `rag-lab query "pregunta" --rewrite` | Activa reescritura de query (desactivado por defecto) |
| `rag-lab query "pregunta" --no-cache` | Ignora la cache de resultados |
| `rag-lab query "pregunta" --profile` | Muestra tiempos por fase |
| `rag-lab chat` | Modo interactivo con seleccion de documentos |

### Ingesta

| Comando | Descripcion |
|---------|-------------|
| `rag-lab ingest` | Ingesta todos los documentos configurados |
| `rag-lab ingest --doc path/to/doc.md` | Ingesta un documento concreto |
| `rag-lab ingest --strict` | Trata los WARNs de frontmatter como errores |
| `rag-lab ingest --force` | Reingesta aunque el documento no haya cambiado |
| `rag-lab ingest --resume` | Retoma un lote incompleto |
| `rag-lab ingest --retry-failed` | Reintenta los documentos fallidos |
| `rag-lab ingest --workers N` | Paralelismo para lotes grandes |

### Gestion de documentos

| Comando | Descripcion |
|---------|-------------|
| `rag-lab docs list` | Lista todos los documentos ingestados |
| `rag-lab docs show <doc_id>` | Muestra metadatos completos de un documento |
| `rag-lab docs tag <doc_id> <tag>` | Asigna una etiqueta |
| `rag-lab docs untag <doc_id> <tag>` | Elimina una etiqueta |
| `rag-lab docs delete <doc_id>` | Elimina un documento de todos los stores |
| `rag-lab docs validate path/to/doc.md` | Valida el frontmatter YAML |
| `rag-lab docs inspect path/to/doc.md` | Muestra estructura, tokens y chunks estimados |
| `rag-lab docs preview-chunks path/to/doc.md` | Muestra chunks sin escribir nada |

### Etiquetas

| Comando | Descripcion |
|---------|-------------|
| `rag-lab tags list` | Lista todas las etiquetas |
| `rag-lab tags rename <old> <new>` | Renombra una etiqueta |
| `rag-lab tags delete <tag>` | Elimina una etiqueta |

### Cache

| Comando | Descripcion |
|---------|-------------|
| `rag-lab cache stats` | Estadisticas de uso de cache |
| `rag-lab cache clear` | Limpia toda la cache |
| `rag-lab cache vacuum` | Elimina entradas expiradas |
| `rag-lab cache inspect <key>` | Inspecciona una entrada |

### Feedback

| Comando | Descripcion |
|---------|-------------|
| `rag-lab feedback list` | Lista los registros de feedback |
| `rag-lab feedback stats` | Estadisticas agregadas |
| `rag-lab feedback export --output path.jsonl` | Exporta a JSONL |
| `rag-lab feedback clear --yes` | Elimina todos los registros |

### Operaciones y diagnostico

| Comando | Descripcion |
|---------|-------------|
| `rag-lab doctor` | Comprobacion de salud completa |
| `rag-lab doctor --checks config,docstore` | Solo algunos checks |
| `rag-lab reconcile --check` | Comprueba consistencia entre stores |
| `rag-lab reconcile --repair` | Elimina orphans en ChromaDB |
| `rag-lab reconcile --repair-fts` | Corrige duplicados en FTS5 |
| `rag-lab reconcile --repair-metadata` | Rellena metadatos NULL |
| `rag-lab diagnose --query "..." --explain` | Diagnostico con desglose de senales |
| `rag-lab benchmark --suite official --variants full --no-cache` | Ejecuta el benchmark oficial |

---

## 8. Arquitectura

La pipeline tiene 9 fases secuenciales, cada una en su propio subpaquete bajo `rag_lab/`:

```
ingest -> chunking -> embedding -> storage -> retrieval -> reranking -> generation -> verification -> feedback
```

### Componentes principales

- **`rag_lab/config.py`** — parametros centralizados (CHUNK_MAX_TOKENS, RETRIEVAL_TOP_K, RERANK_TOP_K, RRF_K, etc.)
- **`rag_lab/exceptions.py`** — jerarquia de excepciones propias (RAGLabError y subclases)
- **`rag_lab/logging_config.py`** — logging centralizado a consola y `rag_lab.log`
- **`rag_lab/cli.py`** — punto de entrada principal
- **`rag_lab/cli_chat.py`** — modo chat interactivo

### Stores (fase 4)

| Store | Tecnologia | Contenido |
|-------|-----------|-----------|
| VectorStore | ChromaDB (HNSW, cosine) | Vectores densos BGE-M3 |
| DocStore | SQLite | Texto de chunks, metadatos, sparse BLOBs, FTS5 |

---

## 9. Retrieval — descripcion detallada

El retrieval es hibrido y tiene cuatro contribuciones:

1. **Dense search** — ChromaDB cosine similarity sobre vectores BGE-M3.
2. **BM25** — busqueda por palabras exactas sobre el indice FTS5 en SQLite.
3. **Sparse rescore** — los vectores sparse BGE-M3 almacenados como BLOBs en SQLite reescoran el pool de candidatos.
4. **RRF fusion** — Reciprocal Rank Fusion combina los rankings de las tres senales anteriores.

Tras la fusion, el pool de candidatos pasa opcionalmente por MMR (habilitado por defecto, `MMR_ENABLED=True`) para diversidad antes del reranker.

El reranker (BAAI/bge-reranker-v2-m3, cross-encoder) evalua cada par (query, chunk) con contexto de encabezado para producir el orden final. Los `RERANK_TOP_K` mejores chunks se pasan al LLM.

**Parametros clave:**

| Parametro | Valor por defecto |
|-----------|------------------|
| RETRIEVAL_TOP_K | 30 candidatos antes del reranker |
| RERANK_TOP_K | 8 chunks al LLM |
| RRF_K | 60 |
| CHUNK_MAX_TOKENS | 800 |

---

## 10. Verificacion de respuestas

Tras la generacion, el sistema verifica automaticamente la respuesta en tres pasos:

1. **Comprobacion de citas** — extrae citas del texto del LLM por regex y las valida contra los chunks recuperados. Clasifica cada cita como VALID, PARTIAL o INVALID.

2. **Comprobacion de consistencia** — una segunda llamada al LLM evalua si la respuesta esta respaldada por los chunks o introduce afirmaciones sin soporte. Configurable con `ENABLE_CONSISTENCY_CHECK` (activo por defecto).

3. **Puntuacion de confianza** — combina cuatro sub-puntuaciones en un valor 0-1:

| Sub-puntuacion | Peso |
|---------------|------|
| Citas (ratio de citas validas) | 35% |
| Retrieval (similitud coseno media) | 30% |
| Consistencia | 25% |
| Cobertura (chunks citados / chunks recuperados) | 10% |

Niveles: HIGH (>= 0.75), MEDIUM (>= 0.50), LOW (< 0.50).

---

## 11. Contrato de frontmatter

Desde v1.19, los documentos Markdown deben incluir frontmatter YAML con metadatos de clasificacion. El unico campo obligatorio es `doc_id`. Los demas generan WARNs si estan ausentes (ERRORs con `--strict`).

### Ejemplo completo recomendado

```yaml
---
doc_id: sdmx_user_guide_2_1
title: SDMX User Guide 2.1
domain: sdmx
source_type: manual
language: en
version: "2.1"
tags:
  - sdmx
  - technical_notes
---
```

### Campos

| Campo | Obligatorio | Comportamiento si ausente |
|-------|-------------|--------------------------|
| `doc_id` | Si | ERROR — ingesta bloqueada |
| `title` | Recomendado | WARN — se usa el primer H1 |
| `domain` | Recomendado | WARN |
| `source_type` | Recomendado | WARN |
| `language` | Recomendado | WARN |
| `version` | Opcional | Sin WARN |
| `tags` | Opcional | Sin WARN |

### Tags derivados

Los campos de clasificacion generan tags automaticamente durante la ingesta:

| Campo | Tag derivado |
|-------|-------------|
| `domain: sdmx` | `domain:sdmx` |
| `source_type: manual` | `source_type:manual` |
| `language: en` | `lang:en` |
| `version: "2.1"` | `version:2.1` |

Estos tags permiten filtrar el corpus en retrieval sin modificar el algoritmo de ranking.

Los campos `dataset` y `dataset_id` estan **prohibidos** — su presencia genera un ERROR `frontmatter_scope_violation`. RAG-Lab no soporta datos tabulares.

Documentos sin frontmatter son tecnicamente validos (WARN `frontmatter_missing`) pero se recomienda anadir el contrato completo a todos los documentos nuevos.

Ver documentacion completa: [docs/FRONTMATTER.es.md](docs/FRONTMATTER.es.md)

---

## 12. Benchmark

El benchmark mide la calidad del retrieval sobre 65 queries curadas con relevancia ground-truth. **No mide la calidad de las respuestas completas del LLM.**

### Resultados oficiales (baseline v1.11)

Suite: official | 65 queries | variante: full | sin cache | corpus: 610 chunks

| Metrica | Valor |
|---------|-------|
| Recall@5 | 0.821 |
| Recall@10 | 0.896 |
| Recall@30 | 0.978 |
| MRR | 0.939 |
| nDCG@10 | 0.837 |
| P50 latencia | 334 ms |
| P95 latencia | 384 ms |

La variante `full` incluye el reranker cross-encoder (el paso mas impactante para el orden final) pero no incluye MMR pre-reranker. Es el proxy correcto para regression guards porque es completamente reproducible y determinista.

### Ejecutar el benchmark

```bash
rag-lab benchmark --suite official --variants full --no-cache
```

### Comparar contra el baseline oficial

```bash
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current /tmp/current_run.json
```

Codigos de salida: 0 = OK, 1 = WARN, 2 = FAIL. Umbrales de regresion: R@5 caida > 2 pp = FAIL, nDCG@10 caida > 2 pp = FAIL, MRR caida > 3 pp = FAIL.

Ver documentacion completa: [docs/BENCHMARKS.en.md](docs/BENCHMARKS.en.md)

---

## 13. Feedback

Tras cada consulta el sistema puede recoger una valoracion del usuario (util / no util). Estos datos se almacenan en SQLite (`rag_lab/feedback/feedback.db`).

**El feedback es estrictamente observacional. No modifica el ranking ni ninguna puntuacion de retrieval.** Esta decision es explicita y deliberada: el conjunto de datos de feedback actual es demasiado pequeno para usarlo como senal de ranking sin riesgo de sobreajuste.

```bash
rag-lab feedback stats      # estadisticas
rag-lab feedback export --output feedback.jsonl  # exportar
```

---

## 14. Funciones desactivadas por defecto

### HyDE (Hypothetical Document Embeddings)

HyDE genera una respuesta hipotetica del LLM y usa su embedding como vector de consulta adicional. Esta disponible con `--hyde` pero **desactivado por defecto** por los siguientes motivos:

Resultado del A/B benchmark sobre 65 queries (v1.12, 2026-05-22):

| Metrica | full (baseline) | full_hyde | Delta |
|---------|----------------|-----------|-------|
| R@5 | 0.821 | 0.782 | -3.8 pp |
| R@10 | 0.896 | 0.858 | -3.8 pp |
| MRR | 0.939 | 0.939 | 0.0 pp |
| nDCG@10 | 0.837 | 0.819 | -1.9 pp |
| P50 latencia | 237 ms | 2966 ms | x12.5 |

Interpretacion: el embedding BGE-M3 ya es suficientemente fuerte en este corpus SDMX. El texto hipotetico desplaza la busqueda hacia vocabulario ligeramente diferente, causando mas misses que aciertos. La latencia extra (una llamada LLM adicional por query) es inaceptable para uso interactivo.

Cuando reconsiderar HyDE: si el corpus crece con documentos de vocabulario muy especializado donde el embedding de la pregunta original es sistematicamente debil, o si el modelo LLM local mejora significativamente en velocidad.

### Reescritura de query

La reescritura reformula la pregunta del usuario con terminologia de dominio antes del retrieval. Disponible con `--rewrite`. No se ha benchmarked de forma sistematica; se mantiene desactivado por defecto hasta tener evidencia solida de mejora.

### Feedback como senal de ranking

El feedback se recoge pero no se usa para modificar rankings. Esta congelado explicitamente.

---

## 15. Documentacion adicional

| Documento | Contenido |
|-----------|-----------|
| [README.en.md](README.en.md) | Guia completa en ingles |
| [docs/BENCHMARKS.en.md](docs/BENCHMARKS.en.md) | Benchmarks detallados, historial de baselines, variantes |
| [docs/FRONTMATTER.es.md](docs/FRONTMATTER.es.md) | Contrato YAML completo con todos los campos |
| [docs/OPERATIONS.es.md](docs/OPERATIONS.es.md) | Guia operacional: doctor, reconcile, diagnose, mantenimiento |
| [docs/ANSWER_VERIFICATION.md](docs/ANSWER_VERIFICATION.md) | Documentacion del sistema de verificacion de respuestas |

---

## 16. Licencia

MIT
