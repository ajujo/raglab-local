# RAG-Lab Benchmark Documentation

## Overview

RAG-Lab uses a retrieval benchmark to measure and guard against regressions in the
retrieval pipeline. The benchmark evaluates Recall@k, MRR, and nDCG@10 across a curated
set of SDMX queries with ground-truth relevance grades.

---

## Baseline oficial: v1.11 (activo para CI)

**Archivo:** `data/baselines/v1.11_official_full_eval.json`  
**Generado:** 2026-05-22  
**Queries:** 65 · **Variante:** `full` · **Corpus:** 610 chunks (tag `v1.11` @ `b2e9594`)  
**Configuración:** `top_k=50`, `rrf_k=20`, `RERANKER_USE_HEADING_CONTEXT=True`, `QUERY_VARIANT_STOPWORD_ENABLED=False`, `QUERY_VARIANT_LAST_TERMS_ENABLED=False`

### Métricas de referencia

| Métrica    | Valor  |
|-----------|--------|
| Recall@5  | 0.8205 |
| Recall@10 | 0.8962 |
| Recall@30 | 0.9782 |
| MRR       | 0.9385 |
| nDCG@10   | 0.8373 |
| P50 (ms)  | 334    |
| P95 (ms)  | 384    |
| P99 (ms)  | 412    |

> **Nota:** v1.11 no cambia ninguna métrica de calidad respecto a v1.10 (Δ+0.0000 en todas).
> El incremento de latencia en benchmarking se debe a la GPU RTX 5090 en modo cold-start;
> en producción la latencia de candidate generation mejora ~2× al eliminar las variantes.

### Comparar contra este baseline

```bash
python -m rag_lab.benchmark --suite official --variants full --output /tmp/current.json
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current  /tmp/current.json
```

### Regresión conocida — q070 (`cross_lingual_es_en`) — heredada de v1.10

> **Query:** "¿Cómo se utilizan las restricciones en SDMX para limitar los valores permitidos?"  
> **MRR antes (v1.9):** 1.000 · **MRR después (v1.10+):** 0.500 · **Δ:** −0.500

El prefijo inglés (doc_id + heading_path) afecta ligeramente la atención del cross-encoder
en consultas en español. Pre-reranker MRR=1.000 — la regresión es efecto puro del reranker.
Para desactivarlo: `RERANKER_USE_HEADING_CONTEXT=False` en config.

---

## Baseline anterior: v1.10 (histórico)

**Archivo:** `data/baselines/v1.10_official_full_eval.json`  
**Generado:** 2026-05-22  
**Queries:** 65 · **Variante:** `full` · **Corpus:** 610 chunks (tag `v1.10` @ `adb1a5a`)  
**Configuración:** `top_k=50`, `rrf_k=20`, `RERANKER_USE_HEADING_CONTEXT=True`

> **Nota:** v1.10 fue el baseline activo hasta v1.11. Conservado como referencia histórica.
> **Usa `v1.11_official_full_eval.json` para los regression guards actuales.**

| Métrica    | Valor  |
|-----------|--------|
| Recall@5  | 0.8205 |
| Recall@10 | 0.8962 |
| Recall@30 | 0.9782 |
| MRR       | 0.9385 |
| nDCG@10   | 0.8373 |

---

## Baseline histórico: v1.8.1

**Archivo:** `data/baselines/v1.8.1_official_full_eval.json`  
**Generado:** 2026-05-22  
**Queries:** 65 (28 originales + 37 curadas en v1.8.1)  
**Variante:** `full`  
**Corpus:** 610 chunks (tag `v1.8.1` @ `614a836`)

> **Nota:** v1.8.1 fue el baseline activo hasta v1.10. Conservado como referencia histórica.
> **Usa `v1.11_official_full_eval.json` para los regression guards actuales.**

### Métricas de referencia (v1.8.1)

| Métrica    | Valor  |
|-----------|--------|
| Recall@5  | 0.8000 |
| Recall@10 | 0.9141 |
| Recall@30 | 0.9731 |
| MRR       | 0.9128 |
| nDCG@10   | 0.8255 |
| P50 (ms)  | 250.92 |
| P95 (ms)  | 257.89 |
| P99 (ms)  | 270.03 |

---

## Baseline histórico: v1.7

**Archivo:** `data/baselines/v1.7_official.json`  
**Generado:** 2026-05-22  
**Queries:** 28 (q001–q028, todas válidas)  
**Variante:** `full`  
**Corpus:** 610 chunks (v1.7 @ tag `00882e3`)

### Métricas de referencia

| Métrica    | Valor  |
|-----------|--------|
| Recall@5  | 0.7708 |
| Recall@10 | 0.9256 |
| Recall@30 | 0.9613 |
| MRR       | 0.8780 |
| nDCG@10   | 0.8067 |
| P50 (ms)  | 240.18 |
| P95 (ms)  | 246.95 |
| P99 (ms)  | 248.21 |

Configuración: `top_k=50`, `rrf_k=20`, dispositivos desde `.env`.

### Qué representa `full` y qué no

`full` es el **proxy de evaluación** más cercano a la pipeline productiva que puede ejecutarse
de forma reproducible y determinista sobre ground-truth fijo. Incluye el reranker BGE
cross-encoder, que es el paso más impactante para el orden final de los resultados.

**Este baseline NO es una réplica exacta de la pipeline interactiva de producción.**
Es el baseline oficial de evaluación para regresiones de retrieval, no el único baseline
operacional posible.

La pipeline productiva real (CLI `rag-lab query`, sin flags opcionales) hace:

1. `process_query()` → genera 1 consulta (original only, variantes desactivadas por defecto desde v1.11)
2. `hybrid_search(diversity_mode=None)` → aplica MMR porque `MMR_ENABLED=True` en config
3. `rerank()` sobre el pool combinado deduplicado

La variante `full` en el benchmark hace:

1. **Consulta única** (sin expansión de query)
2. `hybrid_search(diversity_mode="off")` → **sin MMR pre-reranker**
3. `rerank()` sobre todos los candidatos

**Diferencias documentadas respecto a producción (desde v1.11):**

| Aspecto              | Producción              | Baseline `full`      |
|----------------------|------------------------|----------------------|
| Expansión de query   | 1 consulta (variantes off) | **1 consulta**   |
| MMR antes de rerank  | Sí (MMR_ENABLED=True)  | **No (off)**         |
| top_k de búsqueda    | CLI default: 50        | **50 (RETRIEVAL_TOP_K)** |
| Reranker             | Sí                     | Sí                   |

**Por qué `full` es el proxy correcto para el regression guard:**
- El reranker es el paso más impactante; `full` lo incluye.
- El MMR previo al reranker tiene efecto mínimo (el reranker re-ordena toda la lista).
- La expansión de consulta no es modelable en benchmark con relevance grades fijos por query.
- `full` es completamente reproducible y determinista.

---

## Por qué `data/benchmark_full_latest.json` queda como histórico

`data/benchmark_full_latest.json` (y sus análogos `benchmark_all_20260520.json`, etc.) son
resultados de sesiones de calibración previas a v1.8. Tienen tres limitaciones que los
excluyen como baseline oficial:

1. **Solo 12 queries evaluadas** (q001–q012) — el subconjunto inicial, no el set completo.
2. **Sin trazabilidad de versión** — no documentan el tag de git ni el corpus exacto.
3. **Configuración desactualizada** — generados con `rrf_k=60` vs. el `rrf_k=20` actual
   (calibrado como óptimo según `project_calibration_findings`).

Estos archivos se conservan en `data/` como referencia histórica pero **no deben usarse
como baseline de CI**.

---

## Suite distribution (v1.8.1)

| Suite | validated | n | Uso |
|-------|-----------|---|-----|
| `official` | `true` | 65 | CI regression guard |
| `candidate` | `true` | 4 | Negativos confirmados (no relevant docs) — excluidos del guard |
| `candidate` | `false` | 3 | Grading inconcluso — excluidos del guard |

Categorías en la suite oficial: todos los 10 tipos cubren ≥4 queries, excepto
`negative_no_answer` (0 en official — mantenidos como candidate para no distorsionar recall@k).

---

## Ejecutar el benchmark (v1.8.1+)

```bash
# Suite oficial (65 queries validadas) — equivalente a "rag-lab benchmark run --suite official"
python -m rag_lab.benchmark \
  --suite official \
  --variants full \
  --top-k 50 --rrf-k 20 \
  --output data/baselines/run_$(date +%Y%m%d).json

# Suite candidatas (7 queries)
python -m rag_lab.benchmark \
  --suite candidates \
  --variants full \
  --output /tmp/candidates_run.json

# Todas las queries (sin filtro de suite)
python -m rag_lab.benchmark --variants full --output /tmp/all_run.json

# Todas las variantes (comparación completa)
python -m rag_lab.benchmark --variants dense bm25 dense_bm25 hybrid full

# Con reporte Markdown
python -m rag_lab.benchmark \
  --suite official --variants full \
  --output data/baselines/run.json \
  --report data/baselines/run_report.md
```

## Comparar contra el baseline oficial

```bash
# Comparar con el baseline canónico v1.11 — equivalente a "rag-lab benchmark compare ..."
python -m rag_lab.benchmark.compare \
  --baseline data/baselines/v1.11_official_full_eval.json \
  --current  data/baselines/run_YYYYMMDD.json \
  --variant  full

# Guardar reporte JSON de regresión
python -m rag_lab.benchmark.compare \
  --baseline data/baselines/v1.11_official_full_eval.json \
  --current  data/baselines/run_YYYYMMDD.json \
  --variant  full \
  --output   data/baselines/regression_YYYYMMDD.json
```

Códigos de salida: `0` = OK, `1` = WARN, `2` = FAIL.

## Generar reporte por categorías

```bash
# Markdown — equivalente a "rag-lab benchmark report ..."
python -m rag_lab.benchmark.report data/baselines/run.json --variant full

# JSON
python -m rag_lab.benchmark.report data/baselines/run.json --variant full --json

# Guardar a archivo
python -m rag_lab.benchmark.report data/baselines/run.json --output report.md
```

---

## Estructura de los archivos de benchmark

### `data/benchmark_queries.yaml`

Formato v1.8:

```yaml
queries:
  - id: q001
    text: "What is SDMX?"
    category: glossary_definition       # one of 10 categories
    language: en                         # en | es
    suite: official                      # official | candidate
    validated: true                      # true = included in CI guard
    expected_behavior: "Return definition chunks from Glossary and Training"
    source_of_truth: "SDMX_Glossary, SDMX-Training-introduction-2015"
    doc_relevance:
      SDMX-Training-introduction-2015: 3
      SDMX_Glossary: 3
    notes: >
      Basic definition. Training intro and Glossary have explicit definition sections.
```

**Backward compatible:** el runner acepta el formato v1.7 (sin nuevos campos).
Las queries sin `suite` se tratan como `suite: official`; sin `validated` como `validated: true`.

### Suites y categorías (v1.8)

| Suite | Validated | Uso |
|-------|-----------|-----|
| `official` | `true` | Incluidas en CI regression guard |
| `candidate` | `false` | Backlog para revisión humana; excluidas del guard |

Categorías disponibles: `glossary_definition`, `technical_standard`, `cross_lingual_es_en`,
`multi_chunk_same_doc`, `multi_doc_synthesis`, `acronym_or_exact_term`,
`table_or_structured_reference`, `negative_no_answer`, `ambiguity_test`, `regression_known_hard`.

**Estado actual (v1.8.1):** 65 queries official + 7 queries candidate = 72 total, 10 categorías.

### Archivos JSON de baseline

Generados por `BenchmarkRunner.save()`. Estructura:

```json
{
  "config": { "top_k": 50, "rrf_k": 20, "n_queries": 28, "variants": ["full"] },
  "results": {
    "full": {
      "aggregate": { "recall@5": ..., "ndcg@10": ..., ... },
      "per_query": [ { "query_id": "q001", "recall@5": ..., ... } ]
    }
  }
}
```

---

## Thresholds de regresión (compare.py)

| Métrica   | Umbral             | Severidad |
|-----------|-------------------|-----------|
| Recall@5  | caída > 2 pp       | FAIL      |
| nDCG@10   | caída > 2 pp       | FAIL      |
| MRR       | caída > 3 pp       | FAIL      |
| P95       | aumento > 25% rel. | WARN      |

---

## Variantes disponibles

| Variante     | Descripción                                      | Default | Requiere LLM |
|-------------|--------------------------------------------------|---------|:---:|
| `dense`     | Solo dense (ChromaDB cosine)                     | Sí      | No |
| `bm25`      | Solo BM25/FTS5                                   | Sí      | No |
| `dense_bm25`| RRF2: dense + BM25                               | Sí      | No |
| `hybrid`    | RRF3: dense + BM25 + BGE-M3 sparse               | Sí      | No |
| `full`      | hybrid + BGE cross-encoder reranker (= producción) | Sí    | No |
| `hybrid_cap`| hybrid + doc_cap (opt-in)                        | No      | No |
| `hybrid_mmr`| hybrid + MMR diversity (opt-in)                  | No      | No |
| `full_hyde` | full + HyDE dense augmentation (opt-in, experimental) | No | Sí |

---

## Experimento HyDE (v1.12) — no activado por defecto

**Resultado del A/B sobre 65 queries oficiales (2026-05-22):**

| Métrica | full (v1.11 baseline) | full_hyde | Δ | Veredicto |
|---------|----------------------|-----------|---|-----------|
| R@5 | 0.8205 | 0.7821 | −0.038 | ❌ FAIL (>2pp) |
| R@10 | 0.8962 | 0.8577 | −0.038 | ❌ |
| R@30 | 0.9782 | 0.9756 | −0.003 | ✓ |
| MRR | 0.9385 | 0.9385 | 0.000 | ✓ |
| nDCG@10 | 0.8373 | 0.8187 | −0.019 | ⚠ |
| P50 (ms) | 237 | 2966 | +2729 | ❌ 12.5× slower |

**Queries que mejoran con HyDE (R@5):** q010 (+0.500), q035 (+0.333), q071 (+0.333), q013 (+0.333)

**Queries que empeoran con HyDE (R@5):** q027/q041/q051/q069 (−0.500), q026/q031/q036/q056/q065 (−0.333), q057 (−0.333)

**q070 cross_lingual_es_en:** R@5=0.500 tanto en `full` como en `full_hyde` — HyDE no empeora ni mejora la regresión conocida del reranker context.

**Interpretación:**
El embedding BGE-M3 ya es suficientemente fuerte en este corpus SDMX. El texto hipotético
desplaza la búsqueda densa hacia vocabulario ligeramente diferente, causando más misses que hits.
La latencia extra (llamada LLM por query) es inaceptable para producción.

**Cuándo reconsiderar HyDE:**
- Si el corpus crece significativamente con documentos de vocabulario más técnico o especializado
- Si el modelo LLM local mejora en calidad y velocidad de generación
- Si se encuentra un corpus donde la embedding de la pregunta original es sistemáticamente débil

Para ejecutar el experimento:
```bash
python -m rag_lab.benchmark --suite official --variants full_hyde \
    --top-k 50 --rrf-k 20 --output /tmp/hyde_experiment.json
python -m rag_lab.benchmark.compare \
    --baseline data/baselines/v1.11_official_full_eval.json \
    --current /tmp/hyde_experiment.json --variant full_hyde
```

**Configuración actual:** `HYDE_ENABLED = False` en `rag_lab/config.py`.
Para habilitar temporalmente: `HYDE_ENABLED = True` (no recomendado salvo evidencia nueva).
