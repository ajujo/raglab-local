# Documentación de benchmarks — RAG-Lab

## 1. Propósito del benchmark

El benchmark de RAG-Lab mide la **calidad del retrieval**: qué tan bien el pipeline recupera los chunks relevantes para cada consulta, en qué posición aparecen y a qué velocidad.

El benchmark **no mide** la calidad de la respuesta generada por el LLM. Para eso, existe la auditoría del verificador de respuestas (véase `docs/ANSWER_VERIFICATION.md`).

El benchmark sirve para dos usos principales:

1. **Regression guard:** detectar degradaciones de calidad de retrieval antes de fusionar cambios.
2. **Experimentación:** comparar variantes del pipeline (dense vs. BM25 vs. hybrid vs. full) de forma reproducible.

---

## 2. Qué mide el benchmark

### Métricas de calidad

| Métrica | Definición |
|---|---|
| **Recall@k** | Fracción de documentos relevantes anotados que aparecen en el top-k del retrieval. Valor entre 0 y 1. |
| **MRR** | Mean Reciprocal Rank: 1 dividido por el rango del primer resultado relevante, promediado sobre todas las queries. Premia que el primer hit relevante aparezca lo más arriba posible. |
| **nDCG@10** | Normalised Discounted Cumulative Gain. Considera la posición y el grado de relevancia (0–3 por documento). Un resultado relevante en posición 1 puntúa más que en posición 10. Normalizado respecto al ranking ideal. |

Los grados de relevancia por documento (`doc_relevance`) son:

- **3:** documento muy relevante, contiene la respuesta directa.
- **2:** documento relevante, contiene información relacionada.
- **1:** documento marginalmente relevante.
- **0:** no relevante (o ausente).

La primera aparición de cualquier chunk de un documento determina su rango para MRR y nDCG.

### Métricas de latencia

| Métrica | Definición |
|---|---|
| **P50 (ms)** | Mediana de la latencia de retrieval (sin contar generación LLM). |
| **P95 (ms)** | Percentil 95 de latencia. |
| **P99 (ms)** | Percentil 99 de latencia. Cola de distribución. |
| **pool** | Tamaño medio del candidate pool antes del reranker. |

---

## 3. Qué NO mide el benchmark

El benchmark de retrieval no evalúa:

- **Calidad de la respuesta del LLM:** el texto generado no se analiza.
- **Calidad de las citas:** si las citas del formato `[[N] Fuente: ...]` son correctas o completas.
- **Alucinaciones:** si el LLM inventa información no presente en los chunks.
- **Satisfacción del usuario:** no hay juicio humano sobre la utilidad de la respuesta.

Para evaluación E2E de la respuesta, usar la auditoría del verificador (script `scripts/run_e2e_audit.py`).

---

## 4. Suite oficial

- **Archivo:** `data/benchmark_queries.yaml`
- **Queries:** 65 validadas (suite `official`, `validated: true`)
- **Idiomas:** inglés y español
- **Categorías (10):**

| Categoría | Descripción |
|---|---|
| `glossary_definition` | Definiciones de términos SDMX |
| `technical_standard` | Preguntas sobre estándares técnicos |
| `cross_lingual_es_en` | Consulta en español, documentos en inglés (o viceversa) |
| `multi_chunk_same_doc` | La respuesta requiere varios chunks del mismo documento |
| `multi_doc_synthesis` | La respuesta requiere chunks de varios documentos |
| `acronym_or_exact_term` | Búsqueda por sigla o término exacto |
| `table_or_structured_reference` | Referencias a tablas o datos estructurados |
| `negative_no_answer` | No existe respuesta en el corpus (mantenidas como `candidate`) |
| `ambiguity_test` | Consultas ambiguas que pueden interpretarse de varias formas |
| `regression_known_hard` | Casos conocidamente difíciles para el pipeline |

Adicionalmente, existen 7 queries en suite `candidate` (4 negativos confirmados + 3 con grading inconcluso) que no se incluyen en el regression guard.

---

## 5. Variantes del pipeline

El benchmark puede evaluar varias configuraciones del pipeline:

| Variante | Descripción | Requiere LLM |
|---|---|---|
| `dense` | Solo búsqueda densa (ChromaDB HNSW coseno) | No |
| `bm25` | Solo BM25 / FTS5 (búsqueda léxica) | No |
| `dense_bm25` | RRF2: fusión de dense + BM25 | No |
| `hybrid` | RRF3: dense + BM25 + sparse rescore BGE-M3 | No |
| `full` | hybrid + reranker BGE cross-encoder (variante de producción) | No |
| `full_hyde` | full + HyDE (experimental, desactivado por defecto) | Sí |

La variante **`full` es el proxy de evaluación más cercano a la pipeline productiva** y la que se usa para el regression guard.

---

## 6. Baseline activo: v1.11

**Archivo:** `data/baselines/v1.11_official_full_eval.json`
**Generado:** 2026-05-22
**Queries:** 65 · **Variante:** `full` · **Corpus:** 610 chunks
**Configuración:** `top_k=50`, `rrf_k=20`, `RERANKER_USE_HEADING_CONTEXT=True`, variantes de consulta desactivadas

### Métricas de referencia

| Métrica | Valor |
|---|---|
| Recall@5 | 0.821 |
| Recall@10 | 0.896 |
| Recall@30 | 0.978 |
| MRR | 0.939 |
| nDCG@10 | 0.837 |
| P50 (ms) | 334 |
| P95 (ms) | 384 |
| P99 (ms) | 412 |

> **Nota:** el incremento de latencia en benchmarking se debe a la GPU RTX 5090 en modo cold-start. En producción, la latencia de candidate generation mejora aproximadamente ×2 al eliminar las variantes de consulta.

---

## 7. Cómo ejecutar el benchmark

### Ejecución estándar (variante de producción)

```bash
rag-lab benchmark run --suite official --variants full --no-cache
```

### Ejecución con parámetros explícitos

```bash
python -m rag_lab.benchmark \
  --suite official \
  --variants full \
  --top-k 50 --rrf-k 20 \
  --output data/baselines/run_$(date +%Y%m%d).json
```

### Comparar todas las variantes

```bash
python -m rag_lab.benchmark \
  --suite official \
  --variants dense bm25 dense_bm25 hybrid full
```

### Con reporte Markdown

```bash
python -m rag_lab.benchmark \
  --suite official --variants full \
  --output data/baselines/run.json \
  --report data/baselines/run_report.md
```

---

## 8. Regression guard: comparar contra el baseline

```bash
# Ejecutar y guardar resultado
rag-lab benchmark run --suite official --variants full --no-cache \
  --output /tmp/current.json

# Comparar contra el baseline activo
python -m rag_lab.benchmark.compare \
  --baseline data/baselines/v1.11_official_full_eval.json \
  --current  /tmp/current.json \
  --variant  full
```

Códigos de salida: `0` = OK, `1` = WARN, `2` = FAIL.

### Umbrales de regresión

| Métrica | Umbral | Severidad |
|---|---|---|
| Recall@5 | caída > 2 pp | FAIL |
| nDCG@10 | caída > 2 pp | FAIL |
| MRR | caída > 3 pp | FAIL |
| P95 | aumento > 25% relativo | WARN |

---

## 9. Reporte por categorías

```bash
# Reporte Markdown en consola
python -m rag_lab.benchmark.report data/baselines/run.json --variant full

# Reporte en JSON
python -m rag_lab.benchmark.report data/baselines/run.json --variant full --json

# Guardar a archivo
python -m rag_lab.benchmark.report data/baselines/run.json --output report.md
```

---

## 10. Historial de mejoras

| Versión | Cambio principal | Impacto en métricas |
|---|---|---|
| v1.7 | Pipeline base: hybrid + reranker | R@5=0.771, MRR=0.878, nDCG@10=0.807 (28 queries) |
| v1.8.1 | Suite oficial ampliada a 65 queries | R@5=0.800, R@10=0.914, MRR=0.913, nDCG@10=0.826 |
| v1.10 | Reranker con heading context (`RERANKER_USE_HEADING_CONTEXT=True`) | R@5=0.821, MRR=0.939, nDCG@10=0.837 (+1.5pp nDCG) |
| v1.11 | Variantes de consulta desactivadas (0 beneficio, 2× latencia) | Métricas iguales a v1.10; latencia mejorada en producción |
| v1.12 | Experimento HyDE — resultado negativo | −3.8pp R@5, ×12.5 latencia → HYDE_ENABLED=False |
| v1.13 | Experimento HNSW — sin beneficio a 610 chunks | Mantener M=16, ef=100 |

### Regresión conocida: q070 (`cross_lingual_es_en`)

> **Query:** "¿Cómo se utilizan las restricciones en SDMX para limitar los valores permitidos?"
> **MRR antes (v1.9):** 1.000 · **MRR después (v1.10+):** 0.500 · **Δ:** −0.500

El prefijo en inglés (`doc_id` + `heading_path`) del formato de entrada al reranker afecta ligeramente la atención del cross-encoder en consultas en español. Pre-reranker MRR=1.000 — la regresión es efecto puro del reranker con heading context.

Para desactivarlo: `RERANKER_USE_HEADING_CONTEXT=False` en config (pero implica perder la mejora general de ~1.5pp en nDCG@10).

---

## 11. Experimento HyDE (v1.12)

A/B sobre 65 queries oficiales (2026-05-22):

| Métrica | full (baseline v1.11) | full_hyde | Diferencia |
|---|---|---|---|
| R@5 | 0.821 | 0.782 | −0.038 (FAIL) |
| R@10 | 0.896 | 0.858 | −0.038 |
| R@30 | 0.978 | 0.976 | −0.003 |
| MRR | 0.939 | 0.939 | 0.000 |
| nDCG@10 | 0.837 | 0.819 | −0.019 |
| P50 (ms) | 237 | 2966 | +2729 ms (×12.5) |

**Interpretación:** BGE-M3 es suficientemente potente en el corpus SDMX. El texto hipotético desplaza la búsqueda densa hacia vocabulario ligeramente diferente, causando más misses que hits. La latencia extra (una llamada LLM adicional por consulta) es inaceptable para producción.

**Estado:** `HYDE_ENABLED = False` en `rag_lab/config.py`.

---

## 12. Estructura de los archivos de benchmark

### `data/benchmark_queries.yaml`

```yaml
queries:
  - id: q001
    text: "What is SDMX?"
    category: glossary_definition
    language: en
    suite: official
    validated: true
    expected_behavior: "Return definition chunks from Glossary and Training"
    source_of_truth: "SDMX_Glossary, SDMX-Training-introduction-2015"
    doc_relevance:
      SDMX-Training-introduction-2015: 3
      SDMX_Glossary: 3
    notes: >
      Basic definition. Training intro and Glossary have explicit definition sections.
```

### Archivos JSON de baseline

Generados por `BenchmarkRunner.save()`. Estructura:

```json
{
  "config": { "top_k": 50, "rrf_k": 20, "n_queries": 65, "variants": ["full"] },
  "results": {
    "full": {
      "aggregate": { "recall@5": 0.821, "ndcg@10": 0.837, "mrr": 0.939 },
      "per_query": [ { "query_id": "q001", "recall@5": 1.0, ... } ]
    }
  }
}
```

---

## 13. Aviso importante

Los benchmarks descritos en este documento miden exclusivamente la calidad del **retrieval**: si los chunks correctos aparecen entre los resultados y en qué posición.

**No miden:**
- Si la respuesta generada por el LLM es correcta, completa o útil.
- Si las citas son válidas o están bien formadas.
- Si el LLM alucina información no presente en los chunks.

Para evaluación E2E de la calidad de la respuesta, consultar la auditoría del verificador de respuestas (`docs/ANSWER_VERIFICATION.md`), que evalúa la pipeline completa incluyendo generación, citas y trust score sobre consultas reales.
