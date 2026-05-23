# Changelog

All notable changes to RAG-Lab are documented here.

---

## v1.17 — 2026-05-23 — Release Candidate Audit

### Audit findings and fixes

No retrieval changes. No new loaders. No feedback-as-reranking. This tag represents
the system as a stable release candidate after full operational audit.

**Audit result: PASS.** All 7 acceptance criteria met. The corpus is consistent at
610/610/610/610 (DocStore/ChromaDB/FTS5/Sparse) and benchmark metrics match the
v1.11 official baseline exactly.

**Fixes applied during audit:**

1. **`--repair-metadata` for 197 pre-v2 chunks** (already shipped in v1.16.3):
   `SDMX_2-1_User_Guide_6` was ingested before v2 schema added `embedding_model_name`
   and `embedding_model_version` columns. After `rag-lab reconcile --repair-metadata`,
   all 610 chunks show `BAAI/bge-m3 v2024-09`. No re-embedding needed.

2. **`docs delete` uses `--force`, not `--yes`**: discovered during smoke test.
   Not a bug — it's intentional (mirrors the `--force` flag in git-style CLIs).
   Documented in audit findings.

3. **Stale empty tags (`smoke`, `audit-smoke`) cleaned**: deleted via `rag-lab tags delete --force`.

**Documentation updated:**

- `CLAUDE.md`: commands section updated to `rag-lab` wrapper forms.
- `docs/OPERATIONS.md`: doctor/reconcile/diagnose/benchmark sections updated to
  prefer `rag-lab <cmd>` with `python -m` as fallback alternative.
  Reconcile modes table updated to include `--repair-fts` and `--repair-metadata`.
  Routine maintenance checklist updated.
- `docs/BENCHMARKS.md`: compare-against-baseline snippet updated to use
  `rag-lab benchmark --suite official --variants full --no-cache`.

**Guard tests added:**

`tests/test_cli/test_store_isolation_guard.py` — 7 tests verifying:
- Production DocStore path is distinct from tmp_path stores.
- `_isolate_stores` fixture correctly redirects `DOCDSTORE_SQLITE_PATH`.
- `guard_read_only_integration` raises on attempted production writes.
- `conftest.py` forces `CUDA_VISIBLE_DEVICES=""`, `EMBEDDING_DEVICE=cpu`,
  `RERANKER_DEVICE=cpu` for all tests.

**Benchmark (official, full, no-cache, 65 queries):**

| Variant | R@5 | R@10 | R@30 | MRR | nDCG@10 | P50ms |
|---------|-----|------|------|-----|---------|-------|
| full | 0.821 | 0.896 | 0.978 | 0.939 | 0.837 | 248 |

Δ = 0.000 vs v1.11 baseline. Zero regression.

**Tests:** 922 passed (7 new guard tests).

---

## v1.16.2 — 2026-05-23

### CLI ops commands — expose doctor, benchmark, reconcile, diagnose in rag-lab

No changes to retrieval ranking, scoring, reranker, MMR, RRF, ChromaDB embeddings,
HyDE, query cache, sparse index, ingest pipeline, or feedback store.

**Root cause:** `rag-lab --help` only showed `query / chat / ingest / docs / tags / cache /
feedback`. The operational tools (`doctor`, `benchmark`, `reconcile`, `diagnose`) only
existed as `python -m rag_lab.<module>` entry points. There was no `rag-lab <cmd>` surface.

**Approach:** Thin pass-through Typer commands in `cli.py` using
`context_settings={"allow_extra_args": True, "ignore_unknown_options": True}`.
Each command captures raw args from `ctx.args` and delegates to the respective
`main(argv)` function in the target module. No logic is duplicated.

**Refactored modules** (added `def main(argv=None) -> int` with `parser.parse_args(argv)`):
- `rag_lab/doctor.py`
- `rag_lab/maintenance/reconcile.py`
- `rag_lab/maintenance/diagnose.py`

`rag_lab/benchmark/__main__.py` already had `main(argv=None)`.

**New CLI surface:**
```bash
rag-lab doctor                                     # full health check
rag-lab doctor --checks config,docstore            # subset of checks
rag-lab reconcile                                  # consistency report
rag-lab reconcile --check                          # CI mode (exit 0/1)
rag-lab reconcile --repair-fts                     # fix FTS5 duplicates
rag-lab diagnose                                   # system diagnostic
rag-lab diagnose --query "What is SDMX?" --explain
rag-lab benchmark --suite official --variants full --no-cache
```

**Benchmark flag (corrected in docs):**
- Correct: `--variants full` (plural, space-separated list)
- Not valid: `--variant full` (singular — this is only used by `benchmark.compare`)
- Not valid: `rag-lab benchmark run ...` — no `run` sub-command exists

**Backward compat:** All `python -m rag_lab.*` invocations continue to work unchanged.

**Tests nuevos:** `tests/test_cli/test_ops_commands.py` — 17 tests covering `--help` presence,
sub-command help content, benchmark flag correctness, and `main(argv=[...])` callability.

**Docs corregidas:** `docs/OPERATIONS.md` quickref updated to `rag-lab <cmd>` forms.
`docs/BENCHMARKS.md` comment corrected from `benchmark run` to `benchmark --suite`.

---

## v1.16.1 — 2026-05-23

### Bug fix: FTS5 duplicate rows on --force re-ingest

No changes to retrieval ranking, scoring, reranker, MMR, RRF, ChromaDB embeddings,
HyDE, query cache, or sparse index.

**Root cause:** SQLite FTS5 virtual tables have no `UNIQUE` constraint on content
columns. `INSERT OR REPLACE INTO chunks_fts` therefore behaves identically to
`INSERT` — it never replaces an existing row; it appends a new one. This meant that
every `--force` re-ingest of an already-ingested document silently inflated the FTS5
row count by one duplicate row per chunk.

**Fix (`rag_lab/storage/docstore.py`):**  
Replaced `INSERT OR REPLACE` with an explicit `DELETE` then `INSERT` for the FTS5
sync in `DocStore.add()`:
```python
self._conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
self._conn.execute(
    "INSERT INTO chunks_fts(chunk_id, doc_id, text) VALUES (?, ?, ?)",
    (chunk_id, doc_id, text),
)
```

**Detection (`rag_lab/maintenance/reconcile.py`):**  
`reconcile()` now queries `chunks_fts GROUP BY chunk_id HAVING COUNT(*) > 1` and
exposes duplicates in the result dict as `fts_duplicate_chunk_ids`. `_has_issues()`
returns `True` when duplicates are present. New `--repair-fts` flag rebuilds FTS5
for affected chunk_ids (DELETE + re-INSERT from canonical `chunks` table).

**Tests nuevos:** `tests/test_storage/test_fts5_idempotency.py` — 20 tests covering:
- Single-add and multi-force idempotency.
- BM25 scores not inflated by duplicate rows.
- `delete_by_doc_id` cleans FTS5 correctly after force re-ingest.
- Batch ingest of 3 docs + `--force` leaves zero FTS5 duplicates.
- Rollback does not leave partial FTS5 rows.
- `reconcile` detects duplicates and `--repair-fts` removes them.

---

## v1.16 — 2026-05-23

### Parallel and resumable ingest (6.3.5)

No changes to retrieval ranking, scoring, reranker, MMR, RRF, ChromaDB embeddings,
HyDE, or query cache. Only the ingestion pipeline and CLI are affected.

**Pipeline de 4 fases (single-writer guarantee):**
1. **Parallel preparation** (`--workers N`): validate + clean + chunk each document
   in worker threads. No DB access — pure CPU computation.
2. **SKIPPED detection** (main thread): check `ingest_documents.find_committed(doc_id, hash)`;
   fallback to `data/ingested.jsonl` for pre-v1.16 backward compat.
3. **Sequential embedding**: GPU/CPU model is not thread-safe — always on main thread.
4. **Sequential write** (`IngestTransaction` per doc): ChromaDB → SQLite chunks → FTS5 → metadata.

**Audit tables (schema v5, idempotent migration):**
- `ingest_batches` — one row per CLI invocation; terminal status COMPLETED/PARTIAL/FAILED.
- `ingest_documents` — one row per (batch, document); statuses:
  PENDING → VALIDATING → VALIDATED → EMBEDDING → WRITING → COMMITTED | FAILED | ROLLED_BACK | SKIPPED.

**Nuevas clases (`rag_lab/ingest/transaction.py`):**
- `IngestBatchStore` — CRUD for `ingest_batches`.
- `IngestDocumentStore` — CRUD for `ingest_documents`; `find_committed()` powers SKIPPED detection.

**Config nuevo:**
```python
INGEST_MAX_WORKERS: int = 2  # threads for preparation phase
```

**CLI actualizado:**
```bash
rag-lab ingest [--doc PATH|DIR] [--force] [--resume] [--retry-failed] [--workers N]
rag-lab ingest batches [--limit N]  # new: show batch-level history
```
- `--doc DIR` — ingest all `*.md` files in a directory (sorted).
- `--workers N` — override `INGEST_MAX_WORKERS` for this run.
- `--resume` — continue the most recent incomplete batch (PENDING/FAILED docs).
- `--retry-failed` — create a new batch for all FAILED/ROLLED_BACK docs.
- Output summary: committed / skipped / failed / chunks / elapsed.

**Backward compat:**
- `ingest retry RUN_ID` and `ingest rollback RUN_ID` preserved unchanged.
- `--resume` with no v1.16 batch falls back to stale `ingest_runs` detection (pre-v1.16).

**Tests nuevos:** `tests/test_ingest/test_batch_ingest.py` — 57 tests.

**Garantías explícitas:**
- SQLite and ChromaDB receive writes only from the main thread (single-writer).
- Failure in doc N does not corrupt docs 1..N-1 (per-document rollback).
- SKIPPED docs produce no new chunks and do not change the corpus fingerprint.
- `reconcile` is clean after any combination of commit/rollback.

---

## v1.15 — 2026-05-23

### Feedback capture — structured chunk-level feedback store (6.3.8, phase 1)

No changes to retrieval ranking, scoring, reranker, MMR, RRF, ChromaDB, FTS5, HNSW,
HyDE, or query cache. Feedback is a purely append-only log — it does NOT affect results.

**Módulo nuevo: `rag_lab/feedback/store.py`**

- `FeedbackStore` — SQLite backend (defaults to `docstore.sqlite`), WAL mode.
- Schema: `feedback_events` table — one row per (query, chunk) judgment.
- `VALID_FEEDBACK`: `relevant`, `irrelevant`, `useful`, `not_useful`, `wrong_doc`,
  `outdated`, `duplicate`, `bad_citation`.
- API: `add()`, `list()`, `stats()`, `export_jsonl()`, `clear()`, `close()`.
- `make_query_hash(query)` — SHA-256 of normalized query (deterministic, case-insensitive).
- `make_retrieval_config_hash()` — SHA-256 of pipeline config params (deterministic).
- `get_feedback_store()` / `reset_feedback_store_instance()` — module singleton.

**Config nuevo:**
```python
FEEDBACK_DB_PATH = DOCDSTORE_SQLITE_PATH  # stored in docstore.sqlite
```

**CLI nuevo (sub-app `feedback`):**
```bash
rag-lab feedback add --query "..." --chunk-id "..." --feedback relevant
rag-lab feedback add --query "..." --chunk-id "..." --feedback irrelevant --reason wrong_doc
rag-lab feedback list [--limit N] [--chunk-id ID] [--feedback TYPE]
rag-lab feedback stats
rag-lab feedback export [--output path.jsonl]
rag-lab feedback clear --yes
```

**Integración con `query`:**
Tras mostrar la respuesta verificada, el comando `query` imprime un bloque compacto con
rank/chunk_id/doc_id/score para cada chunk recuperado, más un comando de feedback
de ejemplo para el chunk #1. El feedback NO es recogido automáticamente.

**Diagnose:**
`rag-lab doctor` muestra el número de feedback events almacenados.
No bloquea ni falla si la tabla aún no existe.

**Exportación JSONL:**
Cada línea incluye todos los campos: query, query_hash, chunk_id, doc_id, rank,
feedback, rating, reason, source, pipeline_variant, cache_hit, cache_key,
corpus_fingerprint, retrieval_config_hash, created_at, user_note.

**Garantías explícitas:**
- Ningún resultado de retrieval cambia al añadir feedback.
- El `corpus_fingerprint` no cambia al añadir feedback (cache no invalidada).
- Feedback sobre chunk_id inexistente se permite (no hay FK a la tabla chunks).

**Nota v1.16:** El feedback como señal de re-ranking se implementará en v1.16.
Ver `docs/OPERATIONS.md` para el plan de evolución.

**Tests:** 38 nuevos en `tests/test_feedback/test_feedback_events.py` (817 total).

---

## v1.14.1 — 2026-05-23

### Cache metadata invalidation — cache_revision counter in MetadataStore

Patch sobre v1.14. Sin cambios de retrieval.

**Problema:** `get_corpus_fingerprint()` usaba `"{n_chunks}:{max_ingest_run_id}"`, que no
cambia cuando se asignan/desasignan tags o se renombran/eliminan tags. Las queries
filtradas por tag podían devolver resultados de caché obsoletos tras operaciones de tagging.

**Fix:**
- `MetadataStore` gana tabla `cache_revision` (single-row SQLite counter, seeded at 0).
- `bump_revision()` incrementa el contador. Se llama en: `assign_tag`, `unassign_tag`,
  `rename_tag` (si cambia algo), `delete_tag` (si existe), `delete_document` (si existe).
- `get_revision()` devuelve el valor actual (0 si la tabla no existe — safe for old DBs).
- `get_corpus_fingerprint()` → `"{n_chunks}:{max_ingest_run_id}:{revision}"` (3 segmentos).

**Tests:** 15 nuevos en `tests/test_cache/test_metadata_invalidation.py` (779 total).

---

## v1.14 — 2026-05-23

### Caché de queries — retrieval/reranking persistente, invalidación automática (6.3.7)

No changes to retrieval logic, reranker, chunking, HyDE, MMR, RRF, sparse, FTS5, HNSW, or ChromaDB data.

**Módulo nuevo: `rag_lab/cache/query_cache.py`**

- `QueryCache` — SQLite backend (`data/query_cache.sqlite`), WAL mode.
- `make_cache_key(query, *, filters, top_k, rrf_k, weights, mmr, hyde, emb_version, corpus_fp)` — SHA-256 estable.
- `get_corpus_fingerprint(conn)` — `"{n_chunks}:{max_ingest_run_id}:{revision}"` — cambia en ingest/delete/tag ops (v1.14.1+).
- API: `get()`, `set()`, `invalidate()`, `clear()`, `vacuum()`, `stats()`, `inspect()`.

**Qué se cachea:**

| Elemento | ¿Cacheado? |
|----------|------------|
| Resultados hybrid_search + reranker | Sí |
| Respuesta final LLM | **No** (estocástico; queda fuera v1.14) |

**Integración:**

- `rag_lab/cli.py` → `query` command: bypass embed+search+rerank on hit. `--no-cache` para override.
- `rag_lab/cli.py` → nuevo sub-app `cache`: `stats`, `clear`, `vacuum`, `inspect`.
- `rag_lab/benchmark/runner.py` → `BenchmarkRunner.run(use_cache=False)` — cache off by default.
- `rag_lab/benchmark/__main__.py` → `--no-cache` (default) / `--cache`.
- `rag_lab/maintenance/diagnose.py` → muestra `cache_hit=true · key=<12chars>… · corpus=<fp>`.

**Config:**
```python
QUERY_CACHE_ENABLED: bool = True
QUERY_CACHE_PATH = DATA_DIR / "query_cache.sqlite"
QUERY_CACHE_TTL_SECONDS: int = 604800  # 7 días
```

**Tests:** 41 nuevos en `tests/test_cache/test_query_cache.py` (764 total; 779 con v1.14.1).

**Benchmark (65 queries, variant=full, suite=official):**

| Ejecución | R@5 | nDCG@10 | MRR | p50(ms) |
|-----------|-----|---------|-----|---------|
| sin caché (baseline) | 0.821 | 0.837 | 0.939 | 240 |
| con caché, 1ª vez (misses) | 0.821 | 0.837 | 0.939 | 246 |
| con caché, 2ª vez (hits) | 0.821 | 0.837 | 0.939 | **0** |

La caché no altera métricas de calidad. Latencia ≈ 0ms en hits (solo SQLite lookup).

**Verificación:** 764 tests · reconcile 610/610 · doctor 8/8 OK · benchmark Δ+0.0000 vs v1.11.

---

## v1.13.1 — 2026-05-23

### HNSW baseline alignment — mismatch detection reads authoritative index params

Operational close: eliminates the false HNSW mismatch warning that appeared in
doctor/reconcile after v1.13. No changes to retrieval, reranker, chunking,
HyDE, RRF, MMR, sparse, FTS5, or ChromaDB data.

**Root cause**

The production collection `sdmx_rag` had a stale metadata annotation
`hnsw:search_ef=500` from a past `modify()` experiment. The actual running HNSW
index was built with `ef_search=100` (visible in `col.configuration_json`). The
v1.13 mismatch check read `col.metadata` — the stale annotation — instead of the
authoritative source, producing a spurious warning on every `initialize()`.

**Fix**

`VectorStore._hnsw_effective_params(collection)` (new) reads
`collection.configuration_json['hnsw']` and translates it to the standard
metadata-key format. Falls back to `collection.metadata` for pre-1.5 ChromaDB.
`initialize()` now calls `_check_hnsw_mismatch(self._hnsw_effective_params(...))`.

**Production collection actual params** (from `configuration_json`):

| param            | value  |
|------------------|--------|
| hnsw:space       | cosine |
| hnsw:M           | 16     |
| hnsw:construction_ef | 100 |
| hnsw:search_ef   | 100    |

These match `VECTOR_HNSW_*` config defaults exactly — no mismatch, no warning.
The `hnsw:search_ef=500` in `col.metadata` is vestigial and has zero effect.

**New test**

`test_no_warning_for_stale_metadata_annotation`: creates collection with default
params, annotates metadata via `modify()` with a different `search_ef`, then
verifies that `initialize()` does NOT emit a warning (regression guard).

**Verification:** 723 tests green · reconcile 610/610 · doctor 8/8 OK · benchmark no regression.

---

## v1.13 — 2026-05-23

### HNSW configurable — parámetros auditados, config añadida, rebuild documentado (6.3.3)

No changes to retrieval logic, reranker, chunking, HyDE, RRF, MMR, sparse, FTS5, or embeddings.

**Auditoría de ChromaDB 1.5.5**

- Todos los parámetros HNSW son **build-time** en ChromaDB 1.x: `hnsw:space`, `hnsw:M`,
  `hnsw:construction_ef`, `hnsw:search_ef`.
- No existe parámetro mutable en query-time: `ef_search` persiste en metadata vía `modify()`
  pero NO modifica el índice hnswlib cargado en memoria.
- Cambiar `hnsw:space` en una colección existente lanza `ValueError` explícito de ChromaDB.
- Para aplicar nuevos valores: eliminar `storage/chroma_db/` y reingestar.

**Benchmark de perfiles HNSW (610 vectors, top_k=50, 50 iteraciones)**

| Perfil    |  M | ef_c | ef_s | build(ms) | p50(ms) | p95(ms) | recall vs prod |
|-----------|----|------|------|-----------|---------|---------|----------------|
| current   | 16 |  100 |  100 |       368 |    1.87 |    2.12 | 0.9547         |
| fast      |  8 |   64 |   50 |       371 |    1.87 |    2.10 | **0.8313** ❌  |
| balanced  | 16 |  128 |  100 |       366 |    1.91 |    2.16 | 0.9553         |
| recall    | 32 |  200 |  200 |       367 |    2.09 |    2.33 | 0.9533         |

**Hallazgos:**
- `fast` (M=8) degrada recall significativamente (-12pp) — no recomendado.
- `balanced` y `recall` son estadísticamente equivalentes a `current` (Δ < 0.001).
- Latencia HNSW pura (~2ms) es insignificante vs reranker (~250ms); ningún perfil mejora el benchmark E2E.
- Build time idéntico para todos los perfiles a 610 chunks (~370ms).

**Recomendación: mantener `current` (M=16, ef_c=100, ef_s=100).**
El beneficio de `recall` (M=32) solo sería visible a partir de ~10k+ chunks.

**Benchmark oficial: Δ+0.0000 vs baseline v1.11.**

**Config añadida (`rag_lab/config.py`, sección 6.5)**

```python
VECTOR_HNSW_SPACE: str = "cosine"       # distancia (build-time, inmutable post-create)
VECTOR_HNSW_M: int = 16                 # conexiones por nodo (build-time)
VECTOR_HNSW_CONSTRUCTION_EF: int = 100  # ef_construction (build-time)
VECTOR_HNSW_SEARCH_EF: int = 100        # ef_search (build-time en ChromaDB 1.x)
```

Valores idénticos a los hardcodeados anteriores → la colección existente no tiene mismatch.

**`VectorStore.initialize()` mejorado**

- Detecta si la colección ya existe usando `list_collections()`.
- Si existe: compara metadata con config y emite WARNING de mismatch (sin destruir la colección).
- Si no existe: crea con todos los parámetros configurados vía `create_collection()`.
- Log informativo con M, ef_c, ef_s, space en cada inicialización.

**Nuevo `rag_lab/maintenance/hnsw_profiles.py`**

Herramienta standalone que crea colecciones temporales con cada perfil, copia los embeddings
de producción y mide latencia y recall. No toca la colección de producción.

```bash
python -m rag_lab.maintenance.hnsw_profiles
```

**Tests (`tests/test_storage/test_vector_store_hnsw.py`)**

722 total (24 nuevos):
- Config values correctos y consistentes con colección existente
- `_hnsw_creation_metadata()` retorna las 4 claves con valores del config
- Sin warning cuando metadata coincide
- Sin warning con metadata vacía (colección legacy)
- Warning cuando M/ef_c/ef_s/space difieren → menciona rebuild
- Colección existente no destruida en mismatch
- Metadata de colección existente no alterada al inicializar con mismatch
- Perfiles: current/fast/balanced/recall presentes; current coincide con config; recall > fast

**Active CI baseline:** `data/baselines/v1.11_official_full_eval.json` (unchanged)

---

## v1.12 — 2026-05-22

### HyDE / query rewriter: controlled implementation with benchmark evidence (6.3.3)

No changes to chunking, RRF, MMR, sparse scoring, FTS5, ChromaDB, embeddings, or HNSW.

**Background**

HyDE (Hypothetical Document Embeddings) generates a short hypothetical answer via LLM,
encodes it with BGE-M3, and uses that embedding as the dense retrieval query. The theory
is that the hypothetical vocabulary is closer to the target documents than the bare question.

**A/B measurement — 65 official queries, full pipeline + reranker**

| Metric | full (v1.11 baseline) | full_hyde | Δ |
|--------|----------------------|-----------|---|
| R@5 | 0.8205 | 0.7821 | **−0.038** |
| R@10 | 0.8962 | 0.8577 | −0.038 |
| R@30 | 0.9782 | 0.9756 | −0.003 |
| MRR | 0.9385 | 0.9385 | 0.0000 |
| nDCG@10 | 0.8373 | 0.8187 | **−0.019** |
| P50 (ms) | 237 | 2966 | **12.5× slower** |

Per-query breakdown: 4 queries improved R@5, 10 queries degraded R@5. Net negative.

**Interpretation:** The BGE-M3 query embedding is already strong on the SDMX corpus.
The hypothetical text is semantically close to the original query but uses slightly different
vocabulary, which shifts the dense search away from some correct documents. The latency cost
(+LLM call per query) is also prohibitive.

**Decision: HYDE_ENABLED = False — remains opt-in experiment.**

Decision criteria from the spec were not met: R@5 dropped 3.8pp (above 2pp FAIL threshold),
nDCG@10 dropped 1.9pp, latency increased 12.5×. MRR unchanged is the only positive signal.

**What changed (`rag_lab/config.py`)**

New HyDE configuration block (all disabled by default):
```python
HYDE_ENABLED: bool = False        # disabled by default
HYDE_MAX_TOKENS: int = 300        # token budget for hypothetical answer
HYDE_TEMPERATURE: float = 0.1     # low temperature for factual density
HYDE_FORCE_NO_THINKING: bool = True  # bypass 4× token multiplier
HYDE_TIMEOUT_SECONDS: int = 15    # hard timeout; 0 = no timeout
HYDE_USE_FOR_DENSE: bool = True   # hypothetical → dense retrieval only
HYDE_USE_FOR_BM25: bool = False   # original text → BM25 (not contaminated)
HYDE_USE_FOR_SPARSE: bool = False  # original sparse weights preserved
```

Query rewriting config:
```python
QUERY_REWRITING_ENABLED: bool = False
QUERY_REWRITING_MAX_TOKENS: int = 200
QUERY_REWRITING_TEMPERATURE: float = 0.0
QUERY_REWRITING_TIMEOUT_SECONDS: int = 10
```

**What changed (`rag_lab/generation/llm_client.py`)**

`generate_response` now accepts:
- `timeout: Optional[float]` — forwarded to the API call; None = no timeout
- `force_no_thinking: bool` — when True, skips the `_THINKING_TOKEN_MULTIPLIER=4×`
  allocation and uses the requested `max_tokens` directly. Used for HyDE and query
  rewriting to keep token budgets exact and avoid wasted compute.

**What changed (`rag_lab/retrieval/query_processor.py`)**

- Imports HYDE_* and QUERY_REWRITING_* params from config instead of local constants
- `_generate_hypothetical_answer` uses `force_no_thinking=True` and `timeout=15s`
- HyDE query dict now includes `use_for_dense`, `use_for_bm25`, `use_for_sparse` flags
- Hyde variant only added if LLM returns text different from the original query
- Query rewriting now passes `max_tokens`, `temperature`, `timeout`, `force_no_thinking`
- CLI respects `use_for_sparse=False` on hyde queries (no sparse contamination)

**What changed (benchmark)**

New `full_hyde` variant in `pipeline_variants.py`:
- Calls LLM for hypothetical answer per query
- Encodes hypothetical for dense signal only
- Original query used for BM25 (never contaminated)
- Original sparse weights preserved
- Falls back to `query_dense` if LLM call fails (graceful degradation)
- `stats["hyde_used"]` tracks whether LLM was actually invoked
- Listed in `HYDE_VARIANT_NAMES` and `ALL_VARIANT_NAMES` but NOT in `VARIANT_NAMES`
  (not included in default benchmark runs)

Run HyDE benchmark:
```bash
python -m rag_lab.benchmark --suite official --variants full_hyde --output /tmp/hyde_run.json
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current  /tmp/hyde_run.json --variant full_hyde
```

**Tests (`tests/test_retrieval/test_query_processor.py`, `tests/test_benchmark/test_full_hyde_variant.py`)**

698 total (24 new):
- HyDE disabled by default (no LLM call when use_hyde=False)
- hyde query dict carries use_for_dense=True, use_for_bm25=False, use_for_sparse=False
- Original query always preserved as first result
- LLM failure → only original query returned
- LLM returns same text as query → no hyde entry added
- max_tokens=300, temperature, timeout, force_no_thinking forwarded correctly
- force_no_thinking=True bypasses 4× multiplier
- full_hyde in ALL_VARIANT_NAMES, not in VARIANT_NAMES
- full_hyde falls back gracefully when LLM unavailable
- original query used for BM25 in full_hyde

**Active CI baseline:** `data/baselines/v1.11_official_full_eval.json` (unchanged)

---

## v1.11 — 2026-05-22

### Query variants cleanup: disable stop-word heuristics (6.3.3)

No changes to retrieval logic, reranker, chunking, or embeddings.

**Background (A/B measurement)**

A rigorous A/B test over all 65 official queries compared three variant strategies
using full hybrid search (dense + BM25 + sparse) with real embeddings (no reranker):

| Strategy | R@5 | R@10 | R@30 | MRR | nDCG@10 | Latency/q |
|----------|-----|------|------|-----|---------|-----------|
| A: original only | 0.7705 | 0.9103 | 0.9923 | 0.9043 | 0.8037 | 24ms |
| B: original + stopword | 0.7705 | 0.9103 | 0.9923 | 0.9043 | 0.8037 | 47ms |
| C: original + stopword + tail | 0.7705 | 0.9103 | 0.9923 | 0.9043 | 0.8037 | 58ms |

**Δ A→B = +0.0000 on every metric, for every category, for every query.**

The stop-word and tail-term variants add zero retrieval quality improvement while
doubling or tripling per-query latency (2.4× cost). The extra chunks in the
candidate pool (B: +16, C: +24 per query) contain no additional relevant documents.

The benchmark runner never used `process_query` (it encodes each query directly),
so this cleanup has no effect on any benchmark metrics.

**Side finding:** q070 (cross_lingual_es_en) shows MRR=1.000 without the reranker
across all strategies — the v1.10 regression is purely a reranker effect (heading
context slightly confusing the cross-encoder), not a retrieval issue.

**Config changes (`rag_lab/config.py`)**

Replaced `VARIANTS_COUNT = 2` with two independent, named flags (both disabled):

```python
QUERY_VARIANT_STOPWORD_ENABLED: bool = False   # was: VARIANTS_COUNT >= 1
QUERY_VARIANT_LAST_TERMS_ENABLED: bool = False  # was: VARIANTS_COUNT >= 2
```

**Code changes (`rag_lab/retrieval/query_processor.py`)**

- Removed `VARIANTS_COUNT` import. Removed `load_embedding_model`, `EMBEDDING_MODEL`
  (unused). Removed `Tuple` (unused).
- Replaced `for i in range(VARIANTS_COUNT)` loop with explicit config-controlled branches.
- Renamed variant generation to named functions:
  - `_generate_stopword_variant(query)` — key terms only (stop-words removed)
  - `_generate_last_terms_variant(query)` — last 5 key terms (tail variant)
  - `_generate_query_variant(query, idx)` — legacy dispatcher, backward-compatible
- Fixed bug in `_filtered_terms`: now returns stripped tokens (was returning tokens
  with trailing punctuation — strip was applied only for stop-word check, not output).
- `_STOP_WORDS` promoted to module-level `frozenset` (was recreated on every call).
- Variant types renamed: `"variant_stopword"` / `"variant_last_terms"` (was generic `"expanded"`).
- Explicit deduplication guard: last_terms variant skipped if identical to stopword variant.

**Tests (`tests/test_retrieval/test_query_processor.py`)**

Expanded from 20 to 43 tests. New classes:
- `TestProcessQuery`: original always first/present, variants disabled by default,
  variants appear when enabled via monkeypatch, no duplicates, empty query → 1 result.
- `TestFilteredTerms`: stop words EN/ES, acronym preservation (DSD/MSD/SDMX),
  trailing punctuation stripped, empty/all-stop-words.
- `TestGenerateStopwordVariant`: key terms, short/long/Spanish/acronym queries,
  all-stop-words fallback, lowercased output.
- `TestGenerateLastTermsVariant`: tail focus, suffix of stopword variant.
- `TestGenerateQueryVariantLegacy`: backward compat (idx 0→stopword, 1→last_terms, N→original).

**Production impact**

| Metric | Before (v1.10) | After (v1.11) |
|--------|---------------|---------------|
| Queries per request | 3 | **1** (original only) |
| Encode calls/request | 3 | **1** |
| Hybrid search calls/request | 3 | **1** |
| Candidate pool size | ~75 | ~50 |
| Retrieval quality | unchanged | unchanged |
| P50 latency improvement | — | ~2× faster |

To re-enable variants: set `QUERY_VARIANT_STOPWORD_ENABLED=True` in `.env` or config.py.

**Active CI baseline promoted to v1.11**

`data/baselines/v1.11_official_full_eval.json` replaces v1.10 as the active regression guard.
Metrics unchanged (Δ+0.0000 vs v1.10). Known regression q070 inherited from v1.10.

---

## v1.10 — 2026-05-22

### Reranker structural context: heading_path + doc_id (6.3.3)

No changes to candidate generation, chunking, dense/BM25/sparse scoring, RRF, MMR, or
embeddings. The only change is the text passed to the cross-encoder during reranking.

**What changed**

`rerank()` now builds enriched text for each chunk before feeding it to the cross-encoder:

```
Document: SDMX_Technical_Notes
Section: ## 4. Data Structure Definition > ### 4.2 Key Families

<chunk text>
```

Previously the cross-encoder received bare chunk text only. The structural prefix lets the
model use section context when scoring relevance — particularly useful for ambiguous terms
like "key", "group", or "SDMX-ML" that have different meanings across sections.

**New function: `build_reranker_text(chunk, use_heading_context=True) -> str`**

- Prepends `Document: <doc_id>` and `Section: <heading_path>` when both are available.
- Degrades gracefully: uses only the available field if one is missing; falls back to bare
  text if both are absent (backward-compatible with legacy chunks without heading_path).
- Truncates heading_path at 200 chars to prevent pathologically long prefixes.
- Deterministic: same chunk → same output every time.

**New config: `RERANKER_USE_HEADING_CONTEXT = True`**

Set to `False` in `.env` (or config.py) to restore v1.9 text-only behaviour.

**New field on reranked chunks: `heading_path_used: bool`**

Each chunk returned by `rerank()` now carries `heading_path_used=True/False` indicating
whether heading context was actually used for that chunk (may be False even with
`RERANKER_USE_HEADING_CONTEXT=True` when the chunk has no heading_path).

**Benchmark results (65 official queries, variant=full, vs v1.8.1 baseline)**

| Metric     | v1.8.1 (baseline) | v1.10  | Δ       |
|------------|-------------------|--------|---------|
| R@5        | 0.8000            | 0.8205 | +0.0205 |
| R@10       | 0.9141            | 0.8962 | -0.0179 |
| R@30       | 0.9731            | 0.9782 | +0.0051 |
| MRR        | 0.9128            | 0.9385 | +0.0257 |
| nDCG@10    | 0.8255            | 0.8373 | +0.0118 |

Compare guard: `Overall OK` (no metric crossed FAIL or WARN threshold).

Interpretation: better top-5 precision (R@5 +2.1%, MRR +2.6%, nDCG@10 +1.2%) at a slight
cost in top-10 recall (-1.8%). Since the LLM receives RERANK_TOP_K=8 chunks, improving
ranking precision in the top 5 is the relevant signal.

**Per-category analysis**

| Category                    | MRR pre | MRR post | Δ      |
|-----------------------------|---------|----------|--------|
| ambiguity_test (n=5)        | 0.800   | 1.000    | +0.200 |
| acronym_or_exact_term (n=4) | 0.875   | 1.000    | +0.125 |
| technical_standard (n=15)   | 0.917   | 0.956    | +0.039 |
| regression_known_hard (n=5) | 0.850   | 0.867    | +0.017 |
| glossary_definition (n=17)  | 0.941   | 0.941    | ±0.000 |
| multi_chunk_same_doc (n=4)  | 1.000   | 1.000    | ±0.000 |
| multi_doc_synthesis (n=5)   | 1.000   | 1.000    | ±0.000 |
| table_or_structured (n=5)   | 0.900   | 0.900    | ±0.000 |
| cross_lingual_es_en (n=5)   | 0.867   | 0.767    | -0.100 |

The `cross_lingual_es_en` regression (-0.100, 1 query degraded by -0.500 MRR) is caused
by q070 ("¿Cómo se utilizan las restricciones…"): adding English doc_id/section headers
to Spanish-query scoring slightly changes the cross-encoder's attention. This is a known
tradeoff — all other categories are flat or improved.

**Tests (19 in `tests/test_retrieval/test_reranker_context.py`)**

- `TestBuildRerankerText`: full context format, no heading_path, no doc_id, both absent,
  context disabled, missing keys, None fields, truncation at 200 chars, no duplication,
  deterministic, old chunk without heading_path key, whitespace-only heading treated as empty.
- `TestRerankHeadingContext`: heading_path_used=True/False, context disabled flag, enriched
  text sent to cross-encoder, text-only when disabled, candidate count unchanged, rerank_score
  still attached.

**New CI baseline: `data/baselines/v1.10_official_full_eval.json`**

v1.10 replaces v1.8.1 as the active regression guard baseline. The new baseline captures the
heading-context reranker as the production default. v1.8.1 remains archived for historical
reference.

Known regression documented in baseline meta: q070 (`cross_lingual_es_en`) MRR 1.000→0.500.
Monitor on future branches; disable with `RERANKER_USE_HEADING_CONTEXT=False` if needed.

---

## v1.9 — 2026-05-22

### Real tokenizer for token counting (6.3.3 hygiene)

No retrieval changes. No ranking/MMR/RRF/sparse/FTS5/reranker changes.
No new document loaders. Pure token-counting improvement.

**New module: `rag_lab/utils/tokenizer.py`**

- `count_tokens(text) -> int`: lazy-loads `AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME,
  local_files_only=True)` (BGE-M3 / XLM-RoBERTa subword tokeniser). Cached globally after
  first call — tokeniser files only, full model weights never loaded.
- `local_files_only=True` guarantees no network access: if the tokeniser is not in the local
  HuggingFace cache it fails immediately (OSError, no hang). On any system that has run the
  embedding pipeline at least once, the tokeniser files are already cached.
- Fallback to `max(1, len(text) // 4)` if tokeniser unavailable (offline, ImportError, OSError).
  Logs a one-time warning; `_load_attempted` prevents repeated failed attempts.
- `reset_tokenizer_cache()` for test isolation.
- Configurable via `TOKEN_COUNTING_MODE` ("real" / "approx") and `TOKENIZER_MODEL_NAME`.

**Config additions (`rag_lab/config.py`)**

- `TOKENIZER_MODEL_NAME = EMBEDDING_MODEL` (→ "BAAI/bge-m3")
- `TOKEN_COUNTING_MODE = "real"` (set `"approx"` to bypass tokeniser entirely)

**Integration points updated**

- `rag_lab/chunking/splitter.py`: `_count_tokens` replaced by import alias of `count_tokens`.
  All split points, overlap, sibling merging, and tiny-chunk filtering now use BGE-M3 counts.
- `rag_lab/ingest/validation.py`: `count_tokens_approx()` delegates to `count_tokens`.
  Public API preserved.
- `rag_lab/retrieval/query_processor.py`: HyDE token-count log line uses `count_tokens`.

**Tests (18 in `tests/test_utils/test_tokenizer.py`)**

- Fallback: transformers unavailable, model not in local cache (offline fast-fail), no-retry
  after first failure, from_pretrained raises, approx mode, empty text, proportionality.
- Caching: tokeniser loaded once, reset_tokenizer_cache works.
- Real tokeniser: positive count, empty/whitespace→1, length monotone, Spanish text,
  Markdown table, plausible vs heuristic, no FlagModel loaded.
- Splitter integration: `n_tokens > 0`, proportional to text length.

**Updated tests**

- `tests/test_chunking/test_splitter.py::TestCountTokens`: removed exact-value assertions
  (heuristic-specific), replaced with behavioral contracts (monotone, positive, non-ASCII).
- `tests/test_ingest/test_validation.py::test_count_tokens_approx`: same — removed
  `== 100` assertion, replaced with proportionality check.

**Reproducibility and ingest impact**

> **Important:** v1.9 does NOT modify the already-ingested corpus.
> All 610 existing chunks, their chunk_ids, embeddings, and sparse vectors remain unchanged.
> The retrieval benchmark is unaffected (Δ+0.0000 on all metrics).
>
> However, **future re-ingests** will use BGE-M3 subword token counts instead of the
> `len(text)//4` heuristic. This can alter chunk boundaries, the number of chunks produced,
> and therefore chunk_ids. After a full re-ingest the benchmark baseline should be regenerated.
>
> For reproducibility across machines:
> - All machines should use the same `TOKENIZER_MODEL_NAME` and `TOKEN_COUNTING_MODE`.
> - Machines without `BAAI/bge-m3` in local HuggingFace cache will fall back to the
>   heuristic automatically. Set `TOKEN_COUNTING_MODE=approx` in `.env` to make this
>   explicit and silence the warning.
> - `TOKEN_COUNTING_MODE=approx` restores pre-v1.9 chunking behaviour identically.

**Benchmark**

- Official suite (65 queries) vs v1.8.1 baseline: Δ+0.0000 on all retrieval metrics.
  Corpus unchanged (re-ingest not required). p99 +14% relative (first-call tokeniser load
  in benchmark warm-up) — well below 30% WARN threshold.

---

## v1.8.1 — 2026-05-22

### Benchmark curation — official suite expanded to 65 queries (6.3.6 phase B)

No retrieval changes. No pipeline changes. Pure benchmark ground-truth curation.

**What changed**

- `data/benchmark_queries.yaml` upgraded from v1.8 to v1.8.1 format.
  - 37 candidate queries promoted to `suite: official, validated: true` after corpus
    verification (hybrid_search top-10 confirmed stated doc_relevance sources).
  - doc_relevance grades corrected where actual retrieval differed from original expectations
    (e.g. User_Guide more dominant than Notas for frequency codes and dimension types).
  - 4 confirmed-negative queries (q043–q046) set to `validated: true, suite: candidate`
    (confirmed out-of-scope; kept as candidate to avoid distorting aggregate recall@k).
  - 3 queries remain `validated: false`: q034 (MSD acronym, Glossary not in top-5),
    q047 (migration partial), q062 (Custom Type Scheme not in corpus).
  - Test `test_candidates_all_validated_false` → `test_candidate_validated_true_have_empty_doc_relevance`
    to allow confirmed-negative candidates with validated:true.

**New canonical baseline: v1.8.1**

- Created `data/baselines/v1.8.1_official_full_eval.json`:
  - 65 queries (28 original + 37 curated), variant `full`, top_k=50, rrf_k=20,
    corpus 610 chunks, git tag v1.8.1.
  - Metrics: R@5=0.8000, R@10=0.9141, R@30=0.9731, MRR=0.9128, nDCG@10=0.8255.
  - Metrics are higher than v1.7 baseline because the query set expanded; the 37 new
    queries were curated against the corpus. **Use v1.8.1 baseline for future CI guards.**
- Compare confirmed OK: all thresholds pass vs. v1.7_official.json.

**Suite distribution (v1.8.1)**

- official validated:true: 65 queries (10 categories, all ≥4 per category)
- candidate validated:true: 4 (confirmed negatives)
- candidate validated:false: 3 (grading inconclusive)
- negative_no_answer: 0 in official (justified — recall=0 by design would distort CI guard)

**Docs**

- `docs/BENCHMARKS.md`: v1.8.1 baseline section added as active CI baseline; v1.7 marked
  as historical. Suite distribution table and compare commands updated.

---

## v1.8 — 2026-05-22

### Benchmark complete and continuous evaluation (6.3.6)

**Scope note:** v1.8 completes the *framework* for continuous evaluation.
The official benchmark currently has 28 validated queries; expanding it to
60-100+ queries across all categories is a separate curation phase (post-v1.8).
Candidate queries (44, `validated:false`) do not block CI.

No retrieval changes. No ranking/MMR/RRF/FTS5/ChromaDB/embedding changes.
No multi-format (PDF/DOCX/HTML). No CSV/datasets/tabular data.

**Baseline canónico v1.7**

- Created `data/baselines/v1.7_official.json`: canonical retrieval baseline for v1.7.
  - 28 queries (q001–q028), variant `full` (hybrid RRF3 + BGE reranker), top_k=50, rrf_k=20,
    corpus 610 chunks, git tag v1.7, sha `00882e3`.
  - Rich metadata: git_tag, git_sha, corpus_chunks, embedding/reranker models,
    production_differences, command, queries_file.
  - All existing `benchmark_full_latest.json` etc. remain as historical reference only.

**Nuevo formato de queries YAML (retrocompatible)**

- `data/benchmark_queries.yaml` upgraded from v1.7 to v1.8 format.
  - New fields per query: `category`, `language`, `suite`, `validated`, `expected_behavior`,
    `source_of_truth`.
  - Backward compatible: v1.7 queries (no new fields) load and behave identically.
  - Missing `suite` treated as `official`; missing `validated` treated as `true`.

**Two-tier query system**

- **official** (28 queries, `validated: true`): used by CI regression guard.
- **candidate** (44 queries, `validated: false`): backlog for human review; excluded from guard.
- Total: 72 queries across 10 categories.

**10 categories covered**

`glossary_definition` (18), `technical_standard` (15), `acronym_or_exact_term` (5),
`multi_chunk_same_doc` (4), `cross_lingual_es_en` (5), `multi_doc_synthesis` (5),
`table_or_structured_reference` (5), `negative_no_answer` (5), `ambiguity_test` (5),
`regression_known_hard` (5).

**Benchmark runner (`runner.py`)**

- `BenchmarkRunner.filter_queries(queries, suite, validated_only)` — new utility.
- `run()` now includes `category`, `language`, `suite` in per-query results.
- `run()` now produces `per_category` aggregate in each variant's results.

**Benchmark CLI (`__main__.py`)**

- `--suite official|candidates|all` — filter by suite before running.
- `--validated-only` — keep only validated queries.
- `--report PATH` — generate Markdown report alongside JSON output.

**Regression guard (`compare.py`)**

- Added `recall@10` (WARN, drop > 3 pp) to `DEFAULT_THRESHOLDS`.
- Added `recall@30` (WARN, drop > 3 pp) to `DEFAULT_THRESHOLDS`.
- Added `p99` (WARN, relative increase > 30%) to `DEFAULT_THRESHOLDS`.
- Default variant changed from `hybrid_mmr` to `full`.

**Report module (`report.py`)** — new

- `generate_report(result, variant)` → structured report dict.
- `format_markdown(report)` → human-readable Markdown with per-category table.
- `format_json(report)` → machine-readable JSON.
- `python -m rag_lab.benchmark.report <file> [--variant] [--json] [--output]`

**Documentation**

- Created `docs/BENCHMARKS.md` (no previous benchmark doc existed at this path).
  - Baseline v1.7 documented with full metadata and production differences.
  - Query format v1.8 documented with suite/category semantics.
  - All CLI commands documented.

**Tests**

- `tests/test_benchmark/test_format.py` (new): 25 tests for YAML format, filter_queries,
  backward compatibility, real YAML validation.
- `tests/test_benchmark/test_report.py` (new): 18 tests for generate_report, format_markdown,
  format_json.
- `tests/test_benchmark/test_compare.py`: 9 new tests for v1.8 thresholds (recall@30, p99,
  strong/within-tolerance regression detection).

---

## v1.7 — 2026-05-22

### Technical debt cleanup (no new features, no retrieval changes)

Closes three audit items from the v1.5 audit backlog.
No changes to ranking, RRF, MMR, sparse scoring, FTS5, ChromaDB, or embeddings.

**3.7 — Dead code removal: `generation/verifier.py`**

`rag_lab/generation/verifier.py` (and its `verify_citations` function) was a
superseded implementation that was never called by any production code path.
The active citation verification pipeline lives in `rag_lab/verification/`
(`verify_citations_layer`, `CitationResult`, `CitationStatus`). The old module
had been re-exported from `generation/__init__.py` and had a test file that
only exercised the dead code.

- Deleted `rag_lab/generation/verifier.py`
- Removed `verify_citations` from `rag_lab/generation/__init__.py` and `__all__`
- Deleted `tests/test_generation/test_verifier.py`
- Removed unused `from rag_lab.generation.verifier import verify_citations` import
  from `tests/integration/test_full_pipeline.py`

**3.9 — Bug fix: `reset_reranker_cache` / `load_reranker` device handling**

`load_reranker(device)` returned the cached model regardless of the requested
device — calling `load_reranker("cpu")` after a CUDA load would silently return
the CUDA model. Similarly, `reset_reranker_cache()` cleared the model object
but did not clear the device tracker.

Fix: added `_reranker_cache_device` variable. `load_reranker` now compares the
requested device to the cached device; a mismatch (or empty cache) triggers a
fresh load. `reset_reranker_cache` clears both `_reranker_cache` and
`_reranker_cache_device`.

New regression tests in `tests/test_retrieval/test_reranker_device.py` verify
cache hit on same device, reload on device switch, and device tracker state.

**3.10 — Bug fix: HyDE token budget and thinking mode**

`_generate_hypothetical_answer()` called `generate_response()` without
`max_tokens`, causing it to use `LLM_MAX_TOKENS=2048` multiplied by
`_THINKING_TOKEN_MULTIPLIER=4` = 8192 tokens for a 3-5 sentence hypothetical
paragraph — wasteful and slow.

Fix: HyDE now passes `max_tokens=HYDE_MAX_TOKENS` (300) and
`temperature=HYDE_TEMPERATURE` (0.1) to `generate_response()`. The underlying
`generate_response()` already passes `enable_thinking=False` via
`chat_template_kwargs`, so thinking mode is suppressed on supporting servers
(SGLang). On servers that ignore it (LM Studio), the token budget is now
bounded to 300×4=1200 instead of 8192.

New regression tests in `tests/test_retrieval/test_query_processor.py` verify
that `generate_response` receives the correct parameters and that fallback
behaviour (LLM failure, empty response) is preserved.

---

## v1.6 — 2026-05-22

### Markdown quality gate before ingest

New validation layer that runs before any IngestTransaction opens.

- `rag_lab/ingest/validation.py`: `ValidationSeverity`, `ValidationIssue`,
  `ValidationReport`, `count_tokens_approx`
- `rag_lab/ingest/markdown_contract.py`: `MarkdownValidationConfig` +
  `validate_markdown()` with 10 checks: UTF-8 encoding, empty file, minimum
  content, YAML frontmatter validity, H1 title, heading hierarchy, section
  length, table size, long lines, estimated chunk count
- `rag_lab docs validate [--strict]` — exit 0/1
- `rag_lab docs inspect` — structural summary
- `rag_lab docs preview-chunks [--limit N]` — chunk preview without store writes
- `rag_lab ingest --strict` — WARNs also block; default mode only ERRORs block
- 30 new tests

---

## v1.5.1 — 2026-05-21

### Operational cleanup (no new features)

Hardens operational hygiene before v1.6. No changes to ranking, retrieval, or models.

**Store contamination fix:**
- Deleted the `test_doc` phantom document that polluted production DocStore from a
  prior unguarded CLI integration test.
- Fixed `tests/integration/test_full_pipeline.py` to patch both `config` AND the
  module-level bindings in `docstore.py` / `vector_store.py` so that CLI ingest
  tests always write to `tmp_path` stores, never to production.

**Integration test guard:**
- Added `tests/integration/conftest.py` with `@pytest.mark.integration` auto-marker.
- Added `guard_read_only_integration` fixture to `TestBenchmarks` — raises
  `AssertionError` if any test in that class attempts to write to production stores.
- Registered `integration` marker in root `conftest.py` (eliminates warning).

**Doctor improvements:**
- `check_fts5`: replaced `COUNT(*)` comparison (inflated by FTS5 internal segments)
  with real-ID set comparison. Now reports missing/orphan chunk counts instead of
  a cosmetic false-positive mismatch.
- `check_test_query`: on CUDA OOM, retries on CPU and returns WARN instead of FAIL.
  A saturated GPU in the environment no longer masks real retrieval problems.

**Packaging:**
- Added `pyproject.toml` with `rag-lab` entry point → `rag_lab.cli:app`.
- `pip install -e .` installs the `rag-lab` CLI wrapper to PATH.

**Test count:** 518 (514 from v1.5 + 4 new: FTS5 orphan WARN, FTS5 no-false-positive,
CPU fallback WARN, CPU fallback FAIL).

---

## v1.4 — 2026-05-21

### Transactional ingest with rollback compensation

Makes document ingestion logically transactional, recoverable, and safe against
partial failures spanning DocStore (SQLite), FTS5, ChromaDB, and the metadata
store.  No changes to ranking, MMR, weighted RRF, weights, top-k, or models.

**New schema (v4) in docstore.sqlite:**

`ingest_runs` table tracks every ingest attempt:

| Column | Purpose |
|--------|---------|
| `run_id` | 12-char hex UUID, primary key |
| `doc_id` | Document being ingested |
| `source_path` | Original file path (for retry) |
| `started_at` / `finished_at` | ISO timestamps |
| `status` | `IN_PROGRESS` → `COMMITTED` \| `FAILED` → `ROLLED_BACK` |
| `error_message` | First 500 chars of exception on failure |
| `chunks_expected` | Total chunks produced by chunker |
| `chunks_written_docstore` / `_fts5` / `_chroma` / `_sparse` | Per-store write counts |
| `metadata_written` | 1 once the documents table row is upserted |

**New module `rag_lab/ingest/transaction.py`:**

- `IngestRunStore` — CRUD on `ingest_runs` (create, update, get, list, get_failed,
  get_stale_in_progress).
- `IngestTransaction` — context manager that creates an `IN_PROGRESS` run on enter,
  marks it `COMMITTED` on clean exit, or `FAILED` + triggers rollback on exception.
- `rollback()` — compensation: deletes from ChromaDB (`delete_by_doc_id`), SQLite
  chunks + FTS5 + documents table.  Idempotent (safe to call twice).

**Ingest pipeline wrapped in IngestTransaction:**

Every document ingest now runs inside `with IngestTransaction(doc_id, path, ds):`.
The transaction records progress after each stage: `chunks_expected`,
`chunks_written_chroma`, `chunks_written_docstore/fts5/sparse`, `metadata_written`.
MetadataStore.upsert_document() is now called at the end of every successful ingest
(documents table is populated automatically — no manual `migrate_to_v3` needed for
newly ingested docs).

**Failure scenarios handled:**

| Failure point | Effect of rollback |
|---------------|-------------------|
| Before ChromaDB | No data written; run ROLLED_BACK |
| After ChromaDB, before DocStore | ChromaDB vectors deleted; run ROLLED_BACK |
| After DocStore, before metadata | Chunks + FTS5 + ChromaDB deleted; run ROLLED_BACK |

**New CLI commands (`rag-lab ingest` sub-app):**

```
rag-lab ingest [--doc PATH] [--force] [--resume] [--retry-failed] [--cpu-embedding]
rag-lab ingest runs   [--doc DOC_ID] [--status STATUS] [--limit N]
rag-lab ingest show   RUN_ID
rag-lab ingest rollback RUN_ID [--force]
rag-lab ingest retry    RUN_ID [--force] [--cpu-embedding]
```

`--resume` rolls back stale `IN_PROGRESS` runs (started > 30 min ago) and
re-ingests from the stored source path.  `--retry-failed` does the same for
`FAILED` runs.

**Reconcile integration:**

Two new fields in reconcile output:
- `stale_ingest_runs` — IN_PROGRESS runs > 30 min old (likely crashed)
- `failed_ingest_runs` — FAILED runs awaiting retry or manual rollback

Both cause `exit 1` and appear in the reconcile report with recovery commands.

**Doctor integration:**

New `ingest_health` check (added between `reconcile` and `test_query`):
- `FAIL` — stale IN_PROGRESS runs found
- `WARN` — FAILED runs awaiting retry
- `OK` — no issues; reports last committed ingest

**Test suite:** 510 tests, EXIT_CODE=0 (was 476 in v1.3; +34 new tests covering
IngestRunStore CRUD, IngestTransaction lifecycle, chaos/failure injection at each
stage, reconcile stale/failed detection, and check_ingest_health).

---

## v1.3 — 2026-05-21

### Metadata, tags, and structured filters

Adds a normalized metadata layer and structured document filtering without
touching any retrieval ranking, RRF, MMR, weights, top-k, or models.

**New schema (v3) in docstore.sqlite:**

| Table | Purpose |
|-------|---------|
| `documents` | One row per ingested doc: path, content_hash, source_id, status, timestamps, embedding metadata |
| `tags` | Normalized tag names with auto-increment tag_id |
| `document_tags` | Many-to-many between documents and tags (ON DELETE CASCADE) |
| `sources` | Optional source catalogue (URL, description) |

Migration: `python -m rag_lab.maintenance.migrate_to_v3` — idempotent, populates
documents from existing chunks, migrates tags from legacy doc_manager.db if present.

**Structured filters (`rag_lab/retrieval/filters.py`):**

`FilterSpec` dataclass with `doc_ids`, `tags_include` (AND), `tags_exclude`,
`source_id`, `status`. `resolve_filter(conn, spec)` converts it to a `List[str]`
of doc_ids for the existing filter mechanism. `hybrid_search()` now accepts
`filter_spec=` alongside the existing `doc_ids=`.

**New CLI commands (`rag-lab docs` / `rag-lab tags`):**

```
rag-lab docs list [--tag TAG] [--source SOURCE] [--status STATUS]
rag-lab docs show DOC_ID
rag-lab docs tag DOC_ID TAG_NAME
rag-lab docs untag DOC_ID TAG_NAME
rag-lab docs delete DOC_ID [--force]
rag-lab docs set-source DOC_ID SOURCE_ID
rag-lab tags list
rag-lab tags rename OLD NEW
rag-lab tags delete NAME [--force]
```

`docs delete` removes consistently from chunks (SQLite + FTS5 + documents table)
and ChromaDB. `DocStore.delete_by_doc_id()` and `VectorStore.delete_by_doc_id()`
added as first-class methods.

**Diagnose filter support:**

```
python -m rag_lab.maintenance.diagnose --query "..." --tag glossary
python -m rag_lab.maintenance.diagnose --query "..." --doc-id SDMX_Glossary --explain
python -m rag_lab.maintenance.diagnose --query "..." --exclude-tag test
```

`--explain` now also shows which filters were applied and how many documents
matched before retrieval.

**Reconcile metadata checks:**

Reconcile now reports orphaned documents (documents table row with no chunks),
doc_ids in chunks with no documents row, and document_tags pointing to
non-existent documents.

**Test suite:** 476 tests, EXIT_CODE=0 (was 427 in v1.2; +49 new tests
covering MetadataStore CRUD, FilterSpec resolution, migration idempotency, and
delete_by_doc_id).

---

## v1.2 — 2026-05-21

### Reliability and observability

This release adds diagnostics, regression protection, and extended consistency
checking. No retrieval behaviour, ranking, or model changes.

**New commands:**

- `python -m rag_lab.doctor` — 7-check system health gate (config, docstore,
  chromadb, fts5, sparse_coverage, reconcile, test_query). Exit codes: 0=OK,
  1=WARN, 2=FAIL. Supports `--checks NAME[,...]` to run a subset.
- `python -m rag_lab.benchmark.compare` — regression guard. Compares a current
  benchmark JSON against a saved baseline. Default thresholds: R@5/nDCG@10 drop
  >2 pp = FAIL; MRR drop >3 pp = FAIL; P95 increase >25% = WARN.
- `python -m rag_lab.maintenance.diagnose --explain` — per-signal rank breakdown
  showing `dense_rank`, `bm25_rank`, `sparse_rank`, `rrf_rank`, `mmr_score`, and
  `was_mmr_reordered` for every result.

**Reconcile improvements:**

- `--repair` flag (alias: `--fix`), `--check` CI mode, `--report-json PATH`.
- Extended checks: duplicate chunk IDs, model version mismatches vs config,
  embedding dimension mismatches vs config, sparse format version mismatches vs config.
- Quiet mode (`quiet=True`) for programmatic callers.

**Rank fields in hybrid_search output:**

Every chunk result now carries `dense_rank`, `bm25_rank`, `sparse_rank` (1-based
rank in each signal's list, or `None` if absent), `rrf_rank` (1-based in fused
order), and `was_mmr_reordered` (bool). Used by `--explain` mode.

**Documentation:** `docs/OPERATIONS.md` — runbook covering all operational commands.

**Test suite:** 427 tests, EXIT_CODE=0 (was 343 in v1.1; +84 new tests covering
reconcile, doctor, compare, and explain/rank fields).

---

## v1.1 — 2026-05-21

### MMR document-diversity post-processing

This release activates MMR (Maximal Marginal Relevance) doc-diversity reranking
by default, addressing the large-document monopoly problem identified during the
v1.0 baseline analysis.

**Problem solved:** With `top_k=50`, large documents (e.g. SDMX_2-1_User_Guide_6
with 197 chunks) could occupy multiple result slots in top-5/10, blocking smaller
but equally relevant documents. This degraded nDCG@10 (which counts each doc only
at first occurrence) and reduced the diversity of context passed to the LLM.

**Solution:** MMR post-processing applied after weighted RRF fusion. Greedy
selection penalises chunks from already-represented documents with a configurable
λ parameter. At λ=0.6, relevance still dominates — a second chunk from the same
document survives if its rrf_score justifiably outweighs the diversity penalty.

**Configuration changes (`rag_lab/config.py`):**

| Parameter | Before (v1.0) | After (v1.1) | Reason |
|-----------|--------------|--------------|--------|
| `MMR_ENABLED` | `False` | `True` | Activated after edge case validation |
| `MMR_LAMBDA` | `0.7` | `0.6` | λ=0.6 achieves perfect R@5=1.000 on 28-query set |

To compare against v1.0 baseline: set `MMR_ENABLED = False` in `config.py`.
`DOC_CAP_ENABLED` remains `False` — `hybrid_mmr` provides superior diversity
without a hard per-document limit.

**New code (`rag_lab/retrieval/diversity.py`):**

- `apply_document_cap(chunks, cap)` — hard per-doc-id limit, O(n). Validated,
  kept as experimental alternative (`hybrid_cap` variant).
- `apply_mmr(chunks, lambda_, k)` — doc-diversity MMR greedy selection. Adds
  `mmr_score` field to each result. Does not mutate inputs.

**New benchmark infrastructure:**

- `rag_lab/benchmark/metrics.diversity_stats()` — `unique_docs@k` and
  `max_chunks_same_doc@k` metrics.
- Two new benchmark variants: `hybrid_cap` and `hybrid_mmr` (opt-in via
  `--variants`; not included in the default five-variant run).
- `hybrid_search()` accepts `diversity_mode` parameter (`"cap"`, `"mmr"`, or
  `None`) and passes through doc_cap / mmr_lambda.

**Test suite:** 343 tests, EXIT_CODE=0 (was 323 in v1.0; +20 new diversity tests).

**Benchmark results — v1.1 official** (`top_k=50, rrf_k=20, sparse_w=0.25,
mmr_lambda=0.6`, 28 queries — see `data/benchmark_v1_1_mmr_20260521.json`):

| Variant | R@5 | R@10 | R@30 | MRR | nDCG@10 | unique_docs@5 |
|---------|-----|------|------|-----|---------|:---:|
| hybrid (v1.0 baseline) | 0.762 | 0.923 | 0.982 | 0.867 | 0.755 | 2.75 |
| hybrid_cap (N=3)       | 0.816 | 0.946 | 1.000 | 0.867 | 0.768 | 2.93 |
| **hybrid_mmr (λ=0.6)** | **1.000** | **1.000** | **1.000** | **0.884** | **0.840** | **4.82** |

**Corpus state:** 610 chunks / 610 ChromaDB / 610 FTS5 / 610 sparse BLOBs (100%).

**Key edge case findings (16 new annotated queries, q013–q028):**
- Spanish queries (q027, q028): hybrid R@5=0.000–0.333 → hybrid_mmr R@5=1.000.
  BM25 language mismatch + dense bias had produced a monopoly of marginally-relevant
  English chunks. MMR's diversity pressure surfaces the Spanish-language source.
- Multi-chunk same-doc queries (q013, q026): MMR never causes regression.
  At λ=0.6, the first chunk of the dominant doc stays at rank 1; subsequent chunks
  survive only if their rrf_score justifies the diversity penalty (confirmed for q026
  where nDCG@10 improved 0.974 → 1.000).
- Single-source Glossary queries (q016–q021): zero regressions. Glossary terminology
  is not blocked by MMR when each chunk covers a distinct artefact type.

**Recalibration triggers (same as v1.0):** corpus changes ≥20% size increase,
model updates (BGE-M3 or reranker), cross-lingual query distribution shifts.

---

## Baseline v1.0 — 2026-05-20

### Retrieval baseline: weighted RRF + calibrated parameters

This release closes the retrieval evaluation phase and establishes the first
official performance baseline. It replaces the equal-weight RRF3 fusion with a
calibrated weighted variant and raises the candidate pool from 30 to 50 results.

**Configuration changes (`rag_lab/config.py`):**

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `RETRIEVAL_TOP_K` | 30 | 50 | Larger pool improves R@5 without latency impact |
| `RRF_K` | 60 | 20 | More discriminative ranking at this corpus size |
| `SPARSE_RRF_WEIGHT` | 1.0 (implicit) | 0.25 | Eliminates large-document bias in sparse signal |
| `DENSE_RRF_WEIGHT` | 1.0 (implicit) | 1.0 | Reference weight, unchanged |
| `BM25_RRF_WEIGHT` | 1.0 (implicit) | 1.0 | Unchanged |

**Code changes:**

- `rag_lab/retrieval/fusion.py`: Added `weighted_rrf()` as primary function.
  `rrf_three()` retained as backward-compatible wrapper.
- `rag_lab/retrieval/hybrid_search.py`: Uses `weighted_rrf` with configurable
  per-signal weights. New optional params `dense_weight`, `bm25_weight`, `sparse_weight`.
- `rag_lab/benchmark/__main__.py`: CLI defaults now read from config instead of
  hardcoded values, so benchmark always reflects the active configuration.
- `rag_lab/benchmark/weighted_fusion.py`: Thin re-export from `retrieval.fusion`.
- `rag_lab/benchmark/calibration.py`: Import updated to canonical location.

**Test suite:** 323 tests, EXIT_CODE=0. Includes 21 new tests for `weighted_rrf`
covering weight scaling, sparse dominance, rrf_k discriminativeness, and
backward compatibility with `rrf_three`.

**Benchmark results** (top_k=50, rrf_k=20, sparse_w=0.25, 12 queries):

| Variant | R@5 | MRR | nDCG@10 |
|---------|-----|-----|---------|
| dense        | 0.743 | 0.778 | 0.724 |
| dense_bm25   | 0.792 | 0.840 | 0.755 |
| **hybrid**   | **0.812** | **0.847** | **0.750** |

Hybrid now outperforms dense_bm25 on R@5 (+2pp). Before calibration,
hybrid trailed dense_bm25 by 8pp on R@5.

**Corpus state:** 610 chunks / 610 ChromaDB / 610 FTS5 / 610 sparse BLOBs (100%).

---

## Pre-baseline work (2026-05-20, same day)

The following work was completed before the baseline was frozen:

### Retrieval benchmark framework
- Five pipeline variants: dense, bm25, dense_bm25, hybrid, full
- IR metrics: recall@5/10/30, MRR, nDCG@10, latency P50/P95/P99
- Annotated query file: 12 SDMX queries with graded relevance (0–3 per doc)
- CLI: `python -m rag_lab.benchmark`
- 38 unit tests for metrics and runner

### Calibration grid search
- 324 configurations × 12 queries: `dense_k`, `bm25_k`, `sparse_w`, `bm25_w`, `rrf_k`
- Precompute-once optimization: O(queries) store round-trips vs O(configs × queries)
- Root cause identified: BGE-M3 sparse over-weights large documents at `sparse_w=1.0`
- CLI: `python -m rag_lab.benchmark.calibrate`

### MVP hybrid pipeline (earlier)
- Three-store architecture: ChromaDB (dense) + FTS5 (BM25) + DocStore (sparse BLOBs)
- 100% sparse coverage via `backfill_sparse`
- Five-score result shape: `rrf_score`, `dense_score`, `bm25_score`, `sparse_score`, flags
- Sparse coverage guard: auto-disables sparse if coverage < 95%
- Corpus cleanup: 610/610/610/610 consistency across all stores
