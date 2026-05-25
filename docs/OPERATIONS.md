# RAG-Lab Operations Guide

Operational reference for diagnosing, maintaining, and protecting the RAG-Lab system.

---

## Quick-reference commands

| Goal | Command |
|------|---------|
| Full health check | `python -m rag_lab.doctor` |
| Validate a document | `rag-lab docs validate path/to/doc.md` |
| Validate (strict — warns block) | `rag-lab docs validate --strict path/to/doc.md` |
| Inspect document structure | `rag-lab docs inspect path/to/doc.md` |
| Preview chunks without ingesting | `rag-lab docs preview-chunks path/to/doc.md` |
| Ingest a document (transactional) | `rag-lab ingest --doc path/to/doc.md` |
| Ingest a directory of .md files | `rag-lab ingest --doc path/to/dir/` |
| Ingest with strict validation | `rag-lab ingest --strict --doc path/to/doc.md` |
| Ingest with parallel workers | `rag-lab ingest --workers 4` |
| Ingest all sources | `rag-lab ingest` |
| Resume the last incomplete batch | `rag-lab ingest --resume` |
| Retry all failed documents | `rag-lab ingest --retry-failed` |
| List batch history | `rag-lab ingest batches` |
| List ingest run history | `rag-lab ingest runs` |
| Show a specific ingest run | `rag-lab ingest show <run_id>` |
| Roll back a failed run manually | `rag-lab ingest rollback <run_id>` |
| Retry a specific failed run | `rag-lab ingest retry <run_id>` |
| Health check with test query | `python -m rag_lab.doctor --query "What is SDMX?"` |
| Run specific checks only | `python -m rag_lab.doctor --checks config,docstore,chromadb` |
| System health checks | `rag-lab doctor` |
| Health check subset | `rag-lab doctor --checks config,docstore` |
| Store consistency report | `rag-lab reconcile` |
| Store consistency (CI mode) | `rag-lab reconcile --check` |
| Remove ChromaDB orphans | `rag-lab reconcile --repair` |
| Repair FTS5 duplicates | `rag-lab reconcile --repair-fts` |
| Save reconcile report | `rag-lab reconcile --report-json out.json` |
| Full system diagnostic | `rag-lab diagnose` |
| Diagnostic with test query | `rag-lab diagnose --query "What is SDMX?"` |
| Diagnostic with signal breakdown | `rag-lab diagnose --query "..." --explain` |
| Run benchmark | `rag-lab benchmark --suite official --variants full --no-cache` |
| Benchmark regression check | `python -m rag_lab.benchmark.compare --baseline data/benchmark_v1_1_mmr_20260521.json --current data/benchmark_latest.json` |
| Ingest all documents | `python -m rag_lab.cli ingest` |
| Backfill sparse BLOBs | `python -m rag_lab.maintenance.backfill_sparse` |
| Migrate to schema v2 | `python -m rag_lab.maintenance.migrate_to_v2` |
| Migrate to schema v3 (metadata) | `python -m rag_lab.maintenance.migrate_to_v3` |
| List documents | `python -m rag_lab.cli docs list` |
| List documents by tag | `python -m rag_lab.cli docs list --tag glossary` |
| Show document details | `python -m rag_lab.cli docs show SDMX_Glossary` |
| Tag a document | `python -m rag_lab.cli docs tag SDMX_Glossary glossary` |
| Untag a document | `python -m rag_lab.cli docs untag SDMX_Glossary glossary` |
| Delete document (all stores) | `python -m rag_lab.cli docs delete SDMX_Glossary` |
| List all tags | `python -m rag_lab.cli tags list` |
| Rename a tag | `python -m rag_lab.cli tags rename old-name new-name` |
| Cache stats | `rag-lab cache stats` |
| Clear cache | `rag-lab cache clear` |
| Vacuum cache (remove expired) | `rag-lab cache vacuum` |
| Inspect cache entry | `rag-lab cache inspect <key>` |
| Benchmark without cache (default) | `python -m rag_lab.benchmark --suite official --variants full` |
| Benchmark with cache | `python -m rag_lab.benchmark --suite official --variants full --cache` |
| Add chunk feedback | `rag-lab feedback add --query "..." --chunk-id "..." --feedback relevant` |
| List feedback events | `rag-lab feedback list` |
| Feedback statistics | `rag-lab feedback stats` |
| Export feedback JSONL | `rag-lab feedback export --output path.jsonl` |
| Clear all feedback | `rag-lab feedback clear --yes` |

---

## Installation

Install the `rag-lab` CLI wrapper with editable mode:

```bash
pip install -e .
which rag-lab      # → /path/to/envs/rag-lab/bin/rag-lab
rag-lab --help
```

After this, `rag-lab ingest`, `rag-lab docs`, `rag-lab tags`, etc. all work from PATH.

---

## Doctor command

`rag-lab doctor` (or `python -m rag_lab.doctor`) runs 8 sequential health checks and exits with a clear status.

### Checks

| Check | What it verifies |
|-------|-----------------|
| `config` | Required config constants exist and have valid values |
| `docstore` | SQLite DocStore opens and contains chunks |
| `chromadb` | ChromaDB collection is reachable and non-empty |
| `fts5` | FTS5 index is in sync — uses real ID comparison, not COUNT(*). Reports missing/orphan chunks, not cosmetic counter inflation. |
| `sparse_coverage` | Fraction of chunks with sparse BLOBs meets `SPARSE_COVERAGE_THRESHOLD` |
| `reconcile` | Cross-store consistency (DocStore vs ChromaDB); calls reconcile internally |
| `ingest_health` | No stale IN_PROGRESS runs (>30 min) or FAILED runs without rollback |
| `test_query` | End-to-end retrieval returns at least one result for the test query. Falls back to CPU if GPU is OOM (WARN, not FAIL). |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All checks OK |
| `1` | At least one WARN, no FAIL |
| `2` | At least one FAIL |

### Options

```
--checks NAME[,NAME,...]   Run only the specified checks (comma-separated)
--query TEXT               Query used for test_query check (default: "What is SDMX?")
```

### Example output

```
───────────────────────────────────────────────────
RAG-Lab Doctor
───────────────────────────────────────────────────
  ✓ config               OK  — dim=1024, model_ver=2024-09, top_k=50
  ✓ docstore             OK  — 610 chunks
  ✓ chromadb             OK  — 610 vectors
  ✓ fts5                 OK  — 610/610 chunks indexed
  ✓ sparse_coverage      OK  — 610/610 (100%)
  ✓ reconcile            OK  — DocStore=610, ChromaDB=610
  ✓ test_query           OK  — 3 results — top: SDMX_Glossary rrf=0.0412

───────────────────────────────────────────────────
  ✓ Overall: OK
───────────────────────────────────────────────────
```

---

## Reconcile command

`rag-lab reconcile` (or `python -m rag_lab.maintenance.reconcile`) checks consistency between DocStore and ChromaDB.

### Modes

| Flag | Behaviour |
|------|-----------|
| _(none)_ | Print report, exit 0 if consistent, exit 1 if issues |
| `--check` | CI mode — same as default (explicit) |
| `--repair` | Remove orphaned entries from ChromaDB (destructive) |
| `--fix` | Alias for `--repair` (backward compatibility) |
| `--repair-fts` | Remove FTS5 duplicate rows (v1.16.1+) |
| `--repair-metadata` | Back-fill NULL embedding_model_name/version for pre-v2 chunks (v1.16.3+) |
| `--report-json PATH` | Save full JSON report to PATH |

### Extended checks (v1.2+)

Beyond orphan detection, reconcile now reports:

- **Duplicate chunk IDs** — should never occur (PK constraint), but reported if found
- **Model version mismatches** — chunks ingested with an older `embedding_model_version`
- **Embedding dim mismatches** — chunks with a different `embedding_dim` than current config
- **Sparse format version mismatches** — chunks with a stale `sparse_format_version`

Any of these conditions causes exit code 1 and is flagged in the report.

### JSON report format

```json
{
  "docstore_count": 610,
  "chroma_count": 610,
  "fts_count": 610,
  "sparse_blob_count": 610,
  "chroma_orphans": [],
  "missing_from_chroma": [],
  "duplicate_chunk_ids": [],
  "model_version_mismatches": [],
  "embedding_dim_mismatches": [],
  "sparse_format_version_mismatches": [],
  "repaired": false
}
```

### Recovery actions

| Issue | Command |
|-------|---------|
| ChromaDB orphans | `rag-lab reconcile --repair` |
| FTS5 duplicates | `rag-lab reconcile --repair-fts` |
| NULL model metadata | `rag-lab reconcile --repair-metadata` |
| Missing from ChromaDB | `rag-lab ingest --force` |
| FTS5 incomplete | `python -m rag_lab.maintenance.migrate_to_v2` |
| Sparse BLOBs missing | `python -m rag_lab.maintenance.backfill_sparse` |
| Model version mismatch | Re-ingest affected documents |

---

## Diagnose command

`rag-lab diagnose` (or `python -m rag_lab.maintenance.diagnose`) gives a detailed view of store counts and coverage.

### Options

```
--query TEXT     Run a test retrieval query
--explain        Show per-signal rank breakdown for each result (requires --query)
```

### Explain mode

With `--explain`, each result shows:

```
  [1] SDMX_Glossary | lines 142-160
       rrf=0.0412  dense=0.0231  bm25=12.50  sparse=0.0871
       in_dense=True  in_bm25=True  in_sparse=True
       ┌─ signal ranks: dense[rank=2]  bm25[rank=1]  sparse[rank=3]
       │  rrf_rank=1  chunk_id=abc123de…
       └─ mmr_score=0.7231 ← MMR reordered
```

Reranked chunks also carry `heading_path_used: bool` indicating whether structural
context was prepended to the text sent to the cross-encoder (v1.10+).

This is the primary tool for diagnosing why a specific chunk appears or doesn't appear in results.

---

## Query expansion variants (v1.11+)

`process_query()` can generate additional query variants to broaden the candidate pool.
Both variants are **disabled by default** — A/B testing showed zero retrieval quality benefit
at 2-2.4× per-query latency cost.

### Config flags

```python
QUERY_VARIANT_STOPWORD_ENABLED: bool = False   # key-terms only (stop-words removed)
QUERY_VARIANT_LAST_TERMS_ENABLED: bool = False  # last 5 key terms of the query
```

Set in `rag_lab/config.py` or `.env`.

### When to enable

Re-enable only if empirical A/B testing on your specific query distribution shows
a measurable improvement. Each variant roughly doubles or triples embedding + search cost.

### Variant types

| Variant | Type tag | Example input | Example output |
|---------|----------|--------------|----------------|
| Stopword | `variant_stopword` | "What is the role of SDMX?" | "role sdmx" |
| Tail terms | `variant_last_terms` | "What are the specs for DSD key families?" | "specs dsd key families" |

### Legacy note

Before v1.11, this was controlled by `VARIANTS_COUNT = 2`. That parameter has been removed.

---

## HNSW del vector store — v1.13+

ChromaDB usa HNSW (Hierarchical Navigable Small World) para búsqueda densa.

### Parámetros configurables (`rag_lab/config.py`, sección 6.5)

```python
VECTOR_HNSW_SPACE = "cosine"       # distancia: "cosine", "l2", "ip"
VECTOR_HNSW_M = 16                 # conexiones por nodo (calidad vs memoria)
VECTOR_HNSW_CONSTRUCTION_EF = 100  # calidad del grafo en indexación
VECTOR_HNSW_SEARCH_EF = 100        # pool de candidatos en búsqueda
```

### Restricciones importantes de ChromaDB 1.x

**Todos los parámetros son build-time.** No existe mecanismo de query-time mutable.
Cambiar `hnsw:space` en una colección existente lanza un error explícito.
Llamar a `col.modify(metadata={"hnsw:search_ef": N})` persiste el valor en metadata
pero **no modifica el índice hnswlib en memoria** — el índice sigue usando el valor
con el que fue construido.

### Fuente de verdad: `configuration_json` vs `metadata`

ChromaDB 1.5+ expone dos fuentes de información HNSW:

| Fuente | Qué representa |
|--------|----------------|
| `col.configuration_json['hnsw']` | Parámetros reales del índice construido (**autoritativa**) |
| `col.metadata` | Anotaciones opcionales; pueden ser stale tras `modify()` |

`VectorStore.initialize()` usa `configuration_json` como fuente autoritativa para
la detección de mismatch. Esto evita falsos warnings por anotaciones vestigiales en
metadata de experimentos pasados.

### Colección oficial de producción (baseline aceptado)

```
collection: sdmx_rag
configuration_json.hnsw:
  space:           cosine
  max_neighbors:   16     (= hnsw:M)
  ef_construction: 100
  ef_search:       100
metadata (vestigial): {hnsw:search_ef: 500}  <- stale, sin efecto
```

El `hnsw:search_ef=500` en `metadata` es un residuo de un experimento anterior
(llamada a `modify()`). El índice real usa `ef_search=100`. No se debe modificar
la colección para corregir esto — es inofensivo y no produce warnings.

### Cuándo se aplican los parámetros

| Momento | Efecto |
|---------|--------|
| Primera ingest (colección nueva) | Sí — se aplican al crear la colección |
| Ingest adicional (colección existente) | No — se usa la colección existente |
| Cambiar config sin rebuild | No — se emite WARNING de mismatch |
| Rebuild (eliminar chroma_db + reingestar) | Sí — nueva colección con nuevos params |

### Cómo detectar la configuración activa real

```python
import chromadb
c = chromadb.PersistentClient("storage/chroma_db")
col = c.get_collection("sdmx_rag")
print(col.configuration_json['hnsw'])  # parámetros reales del índice (autoritativo)
print(col.metadata)                    # anotaciones (pueden ser stale)
```

O con el doctor:
```bash
rag-lab doctor
```

### Cómo hacer rebuild

```bash
rm -rf storage/chroma_db/
python -m rag_lab.cli ingest
```

### Mismatch warning

Si los parámetros reales del índice (`configuration_json`) difieren de los valores
en `config.py`, `VectorStore.initialize()` emite un WARNING con las diferencias y el
comando de rebuild. La colección **no se destruye ni modifica** en ningún caso.

Las anotaciones de metadata que difieran del config pero no reflejen los parámetros
reales del índice **no producen warning** (no son mismatches reales).

### Benchmark de perfiles (2026-05-23, 610 chunks)

| Perfil   |  M | ef_c | ef_s | p50(ms) | recall vs prod |
|----------|----|----- |------|---------|----------------|
| current  | 16 |  100 |  100 |    1.87 | 0.9547         |
| fast     |  8 |   64 |   50 |    1.87 | **0.8313** ❌  |
| balanced | 16 |  128 |  100 |    1.91 | 0.9553         |
| recall   | 32 |  200 |  200 |    2.09 | 0.9533         |

**Recomendación:** mantener `current` (M=16). La latencia HNSW (~2ms) es irrelevante
frente al reranker (~250ms). `fast` degrada recall. `balanced`/`recall` aportan < 0.001
de mejora a 610 chunks. Beneficio real de `recall` solo a partir de ~10k chunks.

### Herramienta de perfiles

```bash
python -m rag_lab.maintenance.hnsw_profiles
```

Crea colecciones temporales (sin tocar producción), copia embeddings, mide latencia y recall.

---

## HyDE (Hypothetical Document Embeddings) — v1.12+, opt-in

HyDE generates a short hypothetical answer via LLM, encodes it with BGE-M3, and uses
that embedding as the dense retrieval query. The theory: hypothetical vocabulary is closer
to the target documents than the bare question.

**Status:** Disabled by default. A/B benchmark (65 queries, 2026-05-22) shows net negative
on this corpus: R@5 −3.8pp, nDCG@10 −1.9pp, latency ×12.5. See `docs/BENCHMARKS.md`.

### Config flags

```python
HYDE_ENABLED: bool = False        # main on/off switch
HYDE_MAX_TOKENS: int = 300        # token budget for the hypothetical answer
HYDE_TEMPERATURE: float = 0.1     # low temperature → factual density
HYDE_FORCE_NO_THINKING: bool = True  # skip 4× multiplier (thinking suppressed)
HYDE_TIMEOUT_SECONDS: int = 15    # timeout for LLM call; 0 = no timeout
HYDE_USE_FOR_DENSE: bool = True   # hypothetical → dense retrieval
HYDE_USE_FOR_BM25: bool = False   # original text → BM25 (not generated)
HYDE_USE_FOR_SPARSE: bool = False  # original sparse weights preserved
```

### Fallback behaviour

If the LLM call fails (server down, timeout, empty response), HyDE silently falls back
to the original query. Search quality is the same as without HyDE. No query fails.

### Benchmark experiment

```bash
python -m rag_lab.benchmark --suite official --variants full_hyde \
    --top-k 50 --rrf-k 20 --output /tmp/hyde_experiment.json
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current /tmp/hyde_experiment.json --variant full_hyde
```

### Dependency

Requires a live LLM server. `full_hyde` is NOT included in default benchmark runs.

---

## Query rewriting (LLM-based) — opt-in

Rewrites the user's question before processing to expand acronyms and add domain
terminology. Unlike HyDE, rewriting replaces the original query (not additive).

**Status:** Disabled by default. Not yet benchmarked on official suite.

### Config flags

```python
QUERY_REWRITING_ENABLED: bool = False
QUERY_REWRITING_MAX_TOKENS: int = 200
QUERY_REWRITING_TEMPERATURE: float = 0.0   # deterministic
QUERY_REWRITING_TIMEOUT_SECONDS: int = 10
```

### CLI

```bash
rag-lab query "What is DSD?" --rewrite
```

---

## Caché de queries — v1.14+

### Qué cachea y qué no cachea

| Elemento | ¿Cacheado? |
|----------|------------|
| Resultados de `hybrid_search` + reranker (lista de chunks con scores) | **Sí** |
| Respuesta final del LLM | **No** (intencionado: LLM es estocástico) |
| Verificación / trust score | **No** |
| Embeddings de queries | **No** (recalculados siempre; eficientes) |

### Configuración

```python
QUERY_CACHE_ENABLED: bool = True          # activa/desactiva globalmente
QUERY_CACHE_PATH = DATA_DIR / "query_cache.sqlite"
QUERY_CACHE_TTL_SECONDS: int = 604800     # 7 días; 0 = sin TTL
```

### Cache key

La clave incluye: query normalizada, filtros, top_k, rrf_k, pesos dense/bm25/sparse,
MMR, HyDE, reranker_heading_context, versión del modelo de embedding, sparse format
version, y **corpus fingerprint** (`n_chunks:max_ingest_run_id:revision`).

El corpus fingerprint cambia automáticamente cuando el corpus o sus metadatos cambian.

### Invalidación automática

El fingerprint tiene tres componentes:

| Componente | Cambia cuando |
|------------|---------------|
| `n_chunks` | Se ingesta o elimina un documento |
| `max_ingest_run_id` | Cualquier operación de ingest/delete |
| `revision` | Se asigna, desasigna, renombra o elimina un tag; se elimina un documento |

El contador `revision` vive en `MetadataStore.cache_revision` y se incrementa en
cada operación de mutación de tags o documentos. Esto garantiza que las queries
filtradas por tag no devuelvan resultados obsoletos tras cambios en el tagging.

Las entradas antiguas no se devuelven aunque existan físicamente (el fingerprint
no coincide). `rag-lab cache vacuum` limpia entradas expiradas por TTL o con TTL 0.

### Comandos CLI

```bash
rag-lab cache stats            # estadísticas (entries, hits, tamaño DB)
rag-lab cache clear            # eliminar todas las entradas
rag-lab cache vacuum           # borrar expiradas + VACUUM SQLite
rag-lab cache inspect <key>    # inspeccionar una entrada por su clave
```

### Bypass temporal

```bash
rag-lab query "..." --no-cache  # ignorar la caché para esta consulta
```

### Comportamiento en benchmark

El benchmark ignora la caché por defecto para medir calidad y latencia real:
```bash
python -m rag_lab.benchmark --suite official --variants full          # cache desactivada (default)
python -m rag_lab.benchmark --suite official --variants full --cache  # cache activada (mide beneficio)
```

La caché **no cambia las métricas de calidad** (R@5, nDCG, MRR). Solo reduce latencia
en hits. La comparación oficial contra baseline siempre debe hacerse sin caché.

### Por qué no se cachean respuestas LLM

Las respuestas del LLM son estocásticas (temperatura > 0) y dependen del contexto
completo del sistema. Cachearlas introduciría respuestas potencialmente stale ante
cambios de corpus, config, o prompt. El retrieval es el cuello de botella a reducir;
el LLM se llama una vez por query de todas formas.

---

## Reranker heading context (v1.10+)

The cross-encoder receives enriched text with structural context:

```
Document: SDMX_Technical_Notes
Section: ## 4. Data Structure Definition > ### 4.2 Key Families

<chunk text>
```

### Config flag

`RERANKER_USE_HEADING_CONTEXT = True` (default: on)

Set to `False` in `.env` or `rag_lab/config.py` to restore v1.9 text-only behaviour.

### When to disable

- If you observe regressions on multilingual (ES/EN) queries in your domain.
- During A/B testing when you need a clean text-only reranker baseline.

### Diagnose field

Each chunk returned by `rerank()` carries `heading_path_used: bool`. Query with
`--explain` to see this per-result.

---

## Benchmark regression guard

`python -m rag_lab.benchmark.compare` compares a current benchmark run against a saved baseline.

### Usage

```bash
python -m rag_lab.benchmark.compare \
    --baseline data/benchmark_v1_1_mmr_20260521.json \
    --current  data/benchmark_latest.json \
    --variant  hybrid_mmr \
    --output   data/regression_report.json
```

### Default thresholds

| Metric | Threshold | Severity |
|--------|-----------|----------|
| R@5 | drop > 2 pp | FAIL |
| nDCG@10 | drop > 2 pp | FAIL |
| MRR | drop > 3 pp | FAIL |
| P95 latency | increase > 25% (relative) | WARN |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | No regressions |
| `1` | At least one WARN |
| `2` | At least one FAIL |

### Canonical baseline (activo: v1.11)

**`data/baselines/v1.11_official_full_eval.json`** — baseline activo desde v1.11.

65 queries (suite `official`), variante `full`, `top_k=50`, `rrf_k=20`,
`RERANKER_USE_HEADING_CONTEXT=True`, `QUERY_VARIANT_STOPWORD_ENABLED=False`,
`QUERY_VARIANT_LAST_TERMS_ENABLED=False`.

| Metric  | Value  |
|---------|--------|
| R@5     | 0.8205 |
| R@10    | 0.8962 |
| MRR     | 0.9385 |
| nDCG@10 | 0.8373 |

Métricas idénticas a v1.10 (Δ+0.0000). v1.11 reduce la latencia de candidate generation
~2× al eliminar variantes de query con beneficio nulo (A/B evidencia sobre 65 queries).

Comando estándar de regression guard:

```bash
python -m rag_lab.benchmark --suite official --variants full --output /tmp/current.json
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current  /tmp/current.json
```

**Regresión conocida (heredada de v1.10):** q070 (`cross_lingual_es_en`) MRR 1.000→0.500.
Pre-reranker MRR=1.000 — es efecto puro del cross-encoder con heading context en español.
Documentada en `v1.11_official_full_eval.json` meta → `known_regressions`.

**Baseline anterior:** `data/baselines/v1.10_official_full_eval.json` (histórico, activo v1.10).
**Baseline histórico:** `data/baselines/v1.8.1_official_full_eval.json` (v1.9 y anterior).

---

## Markdown quality gate (v1.6+)

Every ingest run validates the source document against the canonical Markdown contract before opening any transaction.

For full documentation of the frontmatter contract (fields, derived tags, filter integration), see **[docs/FRONTMATTER.md](FRONTMATTER.md)**.

### Validation checks

#### General document checks

| Code | Severity | Description |
|------|----------|-------------|
| `encoding_error` | ERROR | File is not valid UTF-8 |
| `empty_file` | ERROR | File is empty or whitespace-only |
| `min_content` | WARN | Content below `min_content_tokens` threshold (default 50) |
| `missing_title` | WARN | No H1 heading found |
| `heading_hierarchy_skip` | WARN | Heading levels skip (e.g. H1→H3) |
| `section_too_long` | WARN | Section exceeds `max_section_tokens` (default 1600) |
| `large_table` | WARN | Table has more than `max_table_rows` rows (default 200) |
| `estimated_chunks_high` | WARN | Document will exceed `max_estimated_chunks` (default 200) |
| `long_line` | INFO | Line length exceeds `max_line_length` (default 500 chars) |

#### Frontmatter checks (v1.19+)

| Code | Severity | Description |
|------|----------|-------------|
| `frontmatter_missing` | WARN | No `---` frontmatter block found at start of file |
| `frontmatter_unclosed` | WARN | `---` block opened but never closed |
| `frontmatter_invalid_yaml` | ERROR | YAML frontmatter fails to parse |
| `frontmatter_not_mapping` | ERROR | Frontmatter parses but is not a key-value mapping |
| `frontmatter_scope_violation` | ERROR | Prohibited field present (`dataset` or `dataset_id`) |
| `frontmatter_missing_doc_id` | ERROR | `doc_id` field is absent — required for ingestion |
| `frontmatter_missing_title` | WARN | `title` field is absent (H1 will be used as fallback) |
| `frontmatter_missing_domain` | WARN | `domain` classification field is absent |
| `frontmatter_missing_source_type` | WARN | `source_type` classification field is absent |
| `frontmatter_missing_language` | WARN | `language` classification field is absent |
| `frontmatter_tags_not_list` | ERROR | `tags` field is present but is not a YAML list |
| `frontmatter_tag_not_string` | ERROR | A tag element is not a string |
| `frontmatter_tag_empty` | WARN | A tag element is an empty string |
| `frontmatter_tag_whitespace` | WARN | A tag element contains only whitespace |
| `frontmatter_tag_duplicate` | WARN | A tag element appears more than once |
| `yaml_unavailable` | INFO | `pyyaml` is not installed; frontmatter skipped |

### Blocking behaviour

| Mode | Blocks on |
|------|-----------|
| Normal (default) | ERROR only |
| `--strict` | ERROR + WARN |

On block: no stores are written and no `IngestTransaction` is opened. Exit 0 is still returned (the run is just skipped).

### CLI tools

```bash
# Validate before ingesting
rag-lab docs validate path/to/doc.md
rag-lab docs validate --strict path/to/doc.md  # treats warnings as errors

# Inspect document structure (headings, tokens, chunks estimate, issues)
rag-lab docs inspect path/to/doc.md

# Preview chunking result without writing to stores
rag-lab docs preview-chunks path/to/doc.md
rag-lab docs preview-chunks path/to/doc.md --limit 10  # first 10 chunks only
```

### Ingest with strict validation

```bash
# Block on any warning (useful in CI / before bulk ingests)
rag-lab ingest --strict --doc path/to/doc.md
rag-lab ingest --strict  # validate all SOURCES
```

---

## Ingest pipeline (v1.16+)

The ingest command runs a 4-phase pipeline with a **single-writer guarantee**:
worker threads only compute (no DB access); all writes happen on the main thread.

### Phases

1. **Parallel preparation** (`--workers N`) — validate + clean + chunk each doc.
   Thread-safe: no SQLite or ChromaDB access. Default: `INGEST_MAX_WORKERS = 2`.
2. **SKIPPED detection** (main thread) — check `ingest_documents` for a previous
   COMMITTED row with the same content hash. Fallback: scan `data/ingested.jsonl`
   for pre-v1.16 ingests.
3. **Sequential embedding** — GPU/CPU model is not thread-safe; always on main thread.
4. **Sequential write** (`IngestTransaction` per document) — ChromaDB → SQLite chunks
   → FTS5 → metadata. Failure in any phase triggers compensation rollback for that doc
   only; previously committed docs are unaffected.

### Batch tracking (v1.16+)

Each CLI invocation creates one **batch** row (`ingest_batches`) and one
**document** row (`ingest_documents`) per file.

Document status lifecycle:
```
PENDING → VALIDATED → EMBEDDING → WRITING → COMMITTED
                ↓                       ↓
             FAILED               ROLLED_BACK
                                       ↑
                              SKIPPED (hash unchanged)
```

### Checking ingest history

```bash
rag-lab ingest batches                   # last 20 batches (high-level)
rag-lab ingest runs                      # last 20 runs (low-level, per-document)
rag-lab ingest runs --status FAILED      # only failed
rag-lab ingest runs --doc SDMX_Glossary  # runs for one doc
rag-lab ingest show abc123def456         # full details for a run
```

### Ingesting directories

```bash
rag-lab ingest --doc path/to/dir/       # all *.md files, sorted
rag-lab ingest --doc path/to/dir/ --workers 4
```

### Recovering from failures

```bash
# Resume the most recent incomplete batch (PENDING/FAILED docs only)
rag-lab ingest --resume

# Retry all FAILED/ROLLED_BACK documents (creates a new batch)
rag-lab ingest --retry-failed

# Force re-ingest even if content hash is unchanged
rag-lab ingest --doc path/to/doc.md --force

# Manual: roll back a specific run, then re-ingest
rag-lab ingest rollback abc123def456
rag-lab ingest --doc path/to/doc.md --force
```

`rollback` is **idempotent** — safe to run multiple times.

### What rollback does

1. Delete from ChromaDB: `VectorStore.delete_by_doc_id(doc_id)`
2. Delete from SQLite chunks + FTS5 + documents table
3. Mark run status as `ROLLED_BACK`

### Stale IN_PROGRESS detection

Runs that have been `IN_PROGRESS` for more than 30 minutes are considered
stale (process likely crashed). Both `reconcile` and `doctor ingest_health`
detect and report them.

---

## Routine maintenance checklist

Run after any bulk ingest or schema change:

1. `rag-lab doctor` — quick health gate
2. `rag-lab reconcile --check` — cross-store consistency
3. `python -m rag_lab.benchmark.compare --baseline data/benchmark_v1_1_mmr_20260521.json --current <new_run>` — regression guard
4. `pytest tests/ -v` — full test suite

---

## Troubleshooting

### "Sparse scoring disabled: coverage X% < threshold 95%"

Sparse BLOBs are below the configured threshold. Run:

```bash
python -m rag_lab.maintenance.backfill_sparse
```

### "FTS5 table is empty — run migrate_to_v2"

The FTS5 virtual table was not created or is empty. Run:

```bash
python -m rag_lab.maintenance.migrate_to_v2
```

### ChromaDB orphans after integration tests

Some integration tests insert temporary chunks that may not be cleaned up. Run:

```bash
rag-lab reconcile --repair
```

### Stale model version / embedding dim mismatches

Chunks were ingested with an older config. Re-ingest the affected documents:

```bash
python -m rag_lab.cli ingest --doc path/to/document.md --force
```

---

## Document metadata and tags (v1.3+)

### Schema

v1.3 adds five metadata tables to docstore.sqlite (same file as chunks):

| Table | Purpose |
|-------|---------|
| `documents` | One row per doc: path, content_hash, source_id, status, timestamps |
| `tags` | Normalized tag names (tag_id, name UNIQUE) |
| `document_tags` | Many-to-many join, ON DELETE CASCADE |
| `sources` | Optional source catalogue |

### Initial migration

After upgrading to v1.3, run once to populate the documents table from chunks
and migrate any tags from the legacy doc_manager.db:

```bash
python -m rag_lab.maintenance.migrate_to_v3
```

Safe to re-run — fully idempotent.

### Tagging documents

```bash
rag-lab docs tag SDMX_Glossary glossary
rag-lab docs tag SDMX_Glossary sdmx-core
rag-lab docs tag SDMX_2-1_User_Guide_6 user-guide
rag-lab tags rename sdmx-core sdmx
```

### Filtering retrieval by tags

Tags are resolved to doc_ids before any retrieval call. The ranking pipeline
(RRF, MMR, etc.) is unchanged — tags only restrict the candidate pool.

Via diagnose (for debugging):
```bash
rag-lab diagnose --query "code list" --tag glossary
rag-lab diagnose --query "REST API" --exclude-tag test --explain
```

Via Python (`hybrid_search`):
```python
from rag_lab.retrieval.filters import FilterSpec
results = hybrid_search(
    query, vs, ds, fts,
    query_dense=emb,
    query_sparse=sparse,
    filter_spec=FilterSpec(tags_include=["glossary"]),
)
```

### tag include logic (AND)

`tags_include=["glossary", "sdmx-core"]` returns only documents that have
**all** listed tags. Use multiple `--tag` flags in diagnose for AND logic.

### Deleting a document consistently

```bash
rag-lab docs delete SDMX_Glossary_Test
```

Removes from: SQLite chunks + FTS5 + documents table + ChromaDB vectors.

### Reconcile metadata checks

After v3 migration, reconcile also checks:
- Documents in documents table with no chunks (possible after manual deletion)
- Chunk doc_ids with no documents row (migration not run yet)
- document_tags pointing to deleted documents (should be zero due to CASCADE)

```bash
rag-lab reconcile --check
```

---

## Test isolation

Unit tests always use `tmp_path` fixtures and never touch production stores.

Integration tests in `tests/integration/` fall into two categories:

1. **Read-only regression tests** (`test_benchmarks.py`): Access production stores
   to validate retrieval quality on the live corpus. Protected by
   `guard_read_only_integration` fixture which raises `AssertionError` on any write.

2. **Full-pipeline isolation tests** (`test_full_pipeline.py`): Write to stores
   located in `tmp_path` by patching both `rag_lab.config` AND the module-level
   bindings in `docstore.py` / `vector_store.py`. After the test, bindings are
   restored to production paths.

To run only integration tests:
```bash
pytest tests/integration/ -v -m integration
```

To run only unit tests (skip integration):
```bash
pytest tests/ -v -m "not integration"
```

---

## Operational audit checklist

Before starting a new version branch, verify:

```bash
# 1. Tests
pytest tests/ -q

# 2. Store consistency
rag-lab reconcile --check

# 3. System health
rag-lab doctor

# 4. Benchmark regression
python -m rag_lab.benchmark.compare \
  --baseline data/benchmark_v1_1_mmr_20260521.json \
  --current data/benchmark_results_latest.json \
  --variant hybrid

# 5. Scope guard (no tabular/dataset references)
rg -n -i "dataset|csv|parquet|duckdb|forecast|automl" rag_lab tests

# 6. Validate production documents
rag-lab docs validate docs/SDMX_Glossary.md
rag-lab docs validate docs/SDMX_2-1_User_Guide_6.md
```

All commands must exit 0 (or WARN-only for doctor with justified reason).

---

## Feedback — chunk-level signals — v1.15+

### Qué es y qué no es

El módulo de feedback captura juicios de relevancia por chunk. En v1.15, es
**puramente un log de observación**: no altera el ranking, los scores, la caché
ni ningún índice. El feedback acumulado está previsto para alimentar señales de
re-ranking en v1.16.

### Tipos de feedback admitidos

| Valor | Significado |
|-------|-------------|
| `relevant` | Chunk relevante para la query |
| `irrelevant` | Chunk no relevante para la query |
| `useful` | Respuesta basada en este chunk fue útil |
| `not_useful` | Respuesta basada en este chunk no fue útil |
| `wrong_doc` | El chunk es de un documento equivocado |
| `outdated` | El contenido del chunk está desactualizado |
| `duplicate` | Este chunk repite contenido de otro chunk recuperado |
| `bad_citation` | La cita generada a partir de este chunk es incorrecta |

### Comandos CLI

```bash
# Registrar feedback sobre un chunk específico
rag-lab feedback add --query "What is SDMX?" --chunk-id "<chunk_id>" --feedback relevant
rag-lab feedback add --query "What is SDMX?" --chunk-id "<chunk_id>" --feedback irrelevant --reason wrong_doc
rag-lab feedback add --query "..." --chunk-id "..." --feedback bad_citation --note "cita línea 42 incorrecta"

# Listar eventos recientes
rag-lab feedback list                     # últimos 20
rag-lab feedback list --limit 50
rag-lab feedback list --feedback irrelevant
rag-lab feedback list --chunk-id "<id>"

# Estadísticas
rag-lab feedback stats

# Exportar para evaluación futura
rag-lab feedback export                   # imprime JSONL en stdout
rag-lab feedback export --output data/feedback_export.jsonl

# Limpiar
rag-lab feedback clear --yes
```

### Integración con query

Tras mostrar la respuesta, `rag-lab query` imprime el rank, chunk_id, doc_id y score
de cada chunk recuperado, más un comando de ejemplo:

```
── Retrieved chunks (for feedback) ──
   1. chunk=<id>…  doc=sdmx_glossary  score=0.912
   2. chunk=<id>…  doc=sdmx_user_guide  score=0.887
   ...

  To give feedback: rag-lab feedback add --query "..." --chunk-id "..." --feedback relevant
```

### Backend y schema

- **DB:** `storage/docstore.sqlite` (misma base que corpus metadata)
- **Tabla:** `feedback_events`
- **Campos clave:** `query_text`, `query_hash`, `chunk_id`, `doc_id`, `rank`,
  `feedback`, `rating`, `reason`, `source`, `pipeline_variant`, `cache_hit`,
  `cache_key`, `corpus_fingerprint`, `retrieval_config_hash`, `user_note`, `created_at`

La tabla se crea automáticamente en el primer `feedback add`.

### Trazabilidad

- `query_hash` — SHA-256 de la query normalizada. Permite agrupar feedback
  de la misma pregunta independientemente de mayúsculas o espacios extra.
- `corpus_fingerprint` — captura el estado del corpus en el momento del feedback
  (`n_chunks:max_ingest_run_id:revision`). Útil para detectar si el feedback es
  de un corpus diferente al actual.
- `retrieval_config_hash` — SHA-256 de los parámetros de retrieval (top_k, rrf_k,
  pesos, MMR, HNSW, etc.). Permite filtrar feedback tomado con configuraciones distintas.

### Exportación como benchmark queries

El JSONL exportado puede convertirse en queries de benchmark curadas:

```bash
# Exportar feedback negativo para revisar
rag-lab feedback export | jq 'select(.feedback == "irrelevant" or .feedback == "wrong_doc")'
```

Los casos negativos recurrentes son candidatos a nuevas queries en
`data/benchmark_queries.yaml` con `expected_behavior: low_recall_expected` o similar.

### Garantías de aislamiento (v1.15)

- Leer o escribir `feedback_events` no invoca código de retrieval.
- `make_query_hash()` y `make_retrieval_config_hash()` no leen embeddings ni índices.
- `FeedbackStore.add()` no llama a `MetadataStore.bump_revision()` —
  el `corpus_fingerprint` no cambia con operaciones de feedback.
- Tests explícitos en `tests/test_feedback/test_feedback_events.py` verifican
  que los resultados de retrieval son idénticos antes y después de añadir feedback.

### Plan v1.16

En v1.16 se evaluará usar el feedback acumulado como señal adicional en el re-ranking:
- Boost a chunks marcados como `relevant` para queries similares (query_hash match).
- Penalización a chunks marcados como `irrelevant` o `wrong_doc`.
- El uso de feedback como señal será opt-in y benchmarkeado antes de activarse.
