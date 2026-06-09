# RAG-Lab — Guía de uso de RAGAS

Cómo ejecutar la evaluación de calidad de respuesta con RAGAS sobre el corpus SDMX.

---

## Arquitectura de dos entornos

RAGAS y sus dependencias (langchain, datasets, torch) son incompatibles con el entorno
`rag-lab`. La evaluación se divide en dos pasos independientes que se comunican por fichero:

```
[env rag-lab]                        [env ragas]
rag-lab eval run  →  JSONL  →  ragas_eval.py  →  resultados JSON
```

Nunca instalar ragas, langchain ni datasets en el entorno `rag-lab`.

---

## Paso 1 — Capturar salida del pipeline (env rag-lab)

```bash
conda activate rag-lab

rag-lab eval run \
  --suite official \
  --output data/eval_runs/v1.21_baseline.jsonl
```

### Opciones de `rag-lab eval run`

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--suite SUITE` | `official` | Suite de queries. `official` = 65 queries validadas sobre SDMX. |
| `--output PATH` | `data/eval_runs/<suite>_<timestamp>.jsonl` | Fichero de salida. Usar nombre con versión para comparaciones. |
| `--limit N` | — | Evalúa solo las primeras N queries. Útil para smoke tests rápidos. |
| `--queries q001,q002` | — | IDs concretos separados por comas. |
| `--top-k N` | `50` | Pool de candidatos del retriever antes del reranking. |
| `--rerank-top-k N` | `8` | Chunks pasados al LLM. Reducir a 4-5 mejora latencia y posiblemente answer_relevancy. |
| `--temperature F` | `0.0` | Temperatura del LLM. Mantener en 0.0 para reproducibilidad. |

### Schema del JSONL producido

Cada línea es un JSON con este esquema:

```json
{
  "query_id":        "q001",
  "question":        "What is SDMX?",
  "language":        "en",
  "category":        "glossary_definition",
  "answer":          "SDMX es un estándar [[1] Fuente: SDMX_Glossary | Sección: Intro | Líneas: 1-10].",
  "answer_for_eval": "SDMX es un estándar.",
  "contexts":        ["chunk text 1", "chunk text 2", "..."],
  "context_metadata": [
    {"chunk_id": "...", "doc_id": "SDMX_Glossary", "heading_path": "## Glossary", "rerank_score": 0.87}
  ],
  "citations":       [{"chunk_id": "...", "doc_id": "...", "lines": "10-25", "status": "valid"}],
  "trust_score":     0.87,
  "trust_level":     "HIGH",
  "latency_ms":      5800,
  "expected_answer": null,
  "expected_doc_ids": ["SDMX_Glossary"],
  "doc_relevance":   {"SDMX_Glossary": 3},
  "error":           null
}
```

**`answer` vs `answer_for_eval`:**

- `answer` — respuesta completa tal como se muestra al usuario, incluyendo las citas inline
  del tipo `[[N] Fuente: doc_id | Sección: ... | Líneas: X-Y]`. No se modifica.
- `answer_for_eval` — respuesta sustantiva limpia, sin anotaciones de cita, usada
  por `ragas_eval.py` para calcular métricas. Las citas inline suponen ~33% del texto
  de una respuesta típica y pueden contaminar `answer_relevancy` porque RAGAS genera
  preguntas sintéticas desde el texto de la respuesta — si ese texto incluye metadatos
  de fuente, las preguntas se desvían de la pregunta original.

`answer_for_eval` **no elimina las citas del usuario** — el usuario sigue viendo `answer`
completo con todas sus fuentes. Solo afecta a la evaluación interna con RAGAS.

`error` es `null` si la query fue correcta. Si falla (LLM caído, timeout), se registra
el error y el runner continúa con la siguiente query — el fichero queda parcialmente lleno.

---

## Paso 2 — Evaluación RAGAS (env ragas)

```bash
conda activate ragas

python scripts/ragas_eval.py \
  --input  data/eval_runs/v1.21_baseline.jsonl \
  --metrics faithfulness,answer_relevancy \
  --output data/eval_runs/v1.21_baseline_ragas.json
```

### Opciones de `scripts/ragas_eval.py`

| Opción | Por defecto | Descripción |
|--------|-------------|-------------|
| `--input PATH` | requerido | Fichero JSONL producido por `rag-lab eval run`. |
| `--metrics` | `faithfulness` | Métricas separadas por comas. Ver tabla de métricas. |
| `--output PATH` | — | Si se especifica, guarda los resultados en JSON. |
| `--answer-field` | `answer_for_eval` | Campo a pasar a RAGAS como respuesta. `answer_for_eval` (defecto) usa la respuesta limpia sin citas. `answer` usa la respuesta completa. Ver sección "answer vs answer_for_eval" arriba. |

**Compatibilidad con JSONL antiguos:** si el fichero no contiene `answer_for_eval` (generado
antes de v1.21 eval), `ragas_eval.py` cae automáticamente a usar `answer`. No hace falta
pasar ningún flag extra.

### Métricas disponibles

| Métrica | Tipo | Necesita ground truth | Qué mide |
|---------|------|-----------------------|----------|
| `faithfulness` | Reference-free | No | Fracción de afirmaciones en la respuesta que están respaldadas por los contextos recuperados. Mide alucinación: si el LLM inventa algo que no está en los chunks, baja. |
| `answer_relevancy` | Reference-free | No | Grado en que la respuesta aborda directamente la pregunta formulada. Genera preguntas sintéticas desde la respuesta y mide similitud con la original. |
| `context_recall` | Reference-based | Sí (`expected_answer`) | Fracción de la información de la respuesta de referencia que aparece en los contextos recuperados. Requiere respuestas anotadas — **no disponible todavía**. |
| `context_precision` | Reference-based | Sí (`expected_answer`) | Precisión de los contextos recuperados respecto a la respuesta de referencia. Requiere respuestas anotadas — **no disponible todavía**. |

### Configuración del juez LLM

RAGAS usa un LLM externo para evaluar las métricas que requieren razonamiento
(`faithfulness`, `answer_relevancy`). El juez está configurado en `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_JUDGE_MODEL=deepseek/deepseek-v4-flash
```

El juez es deliberadamente **distinto** al LLM generador (Qwen3.6-27B local) para
evitar *self-preference bias* — los modelos tienden a puntuar más alto sus propias
respuestas cuando actúan como juez.

**Coste estimado:** 65 queries × 2 métricas ≈ $0.10–0.30 por corrida completa
con DeepSeek v4 Flash. Verificar en [openrouter.ai](https://openrouter.ai).

---

## Smoke test rápido (5 queries, ~5 min)

```bash
# Paso 1
conda activate rag-lab
rag-lab eval run --suite official --limit 5 --output /tmp/smoke.jsonl

# Paso 2
conda activate ragas
python scripts/ragas_eval.py --input /tmp/smoke.jsonl --metrics faithfulness
```

Útil para verificar que el pipeline funciona antes de lanzar la suite completa.

---

## Ciclo completo de evaluación (~25 min)

```bash
# Definir versión
VERSION="v1.22"

# Paso 1: captura de pipeline (env rag-lab, ~13 min)
conda activate rag-lab
rag-lab eval run \
  --suite official \
  --output data/eval_runs/${VERSION}_baseline.jsonl

# Paso 2: evaluación RAGAS (env ragas, ~10 min)
conda activate ragas
python scripts/ragas_eval.py \
  --input  data/eval_runs/${VERSION}_baseline.jsonl \
  --metrics faithfulness,answer_relevancy \
  --output data/eval_runs/${VERSION}_baseline_ragas.json
```

Añadir el resultado a `docs/BENCHMARK_HISTORY.md` y `docs/BENCHMARK_STATUS.md`.

---

## Notas de mantenimiento del entorno ragas

```bash
# Versiones instaladas (2026-06-08)
# ragas==0.1.21  (0.4.x tiene bug de importación con langchain-community>=0.2)
# langchain-openai==0.1.25
# sentence-transformers==5.5.1  (para answer_relevancy, embeddings en CPU)

# Si el entorno se rompe, recrear con:
conda create -n ragas python=3.11 -y
conda activate ragas
pip install "ragas==0.1.21" langchain-openai sentence-transformers langchain-google-vertexai
```

**Por qué ragas 0.1.21 y no 0.4.x:** la versión 0.4.x tiene un import en tiempo de carga
`from langchain_community.chat_models.vertexai import ChatVertexAI` que falla con
`langchain-community>=0.2` (el módulo se movió a `langchain-google-vertexai`).
La 0.1.21 importa limpio y tiene todas las métricas reference-free que necesitamos.

**Testset generation no disponible:** RAGAS 0.1.21 no incluye generación de testsets sintéticos.
Esta funcionalidad fue añadida en 0.2.x. No se actualiza RAGAS en esta rama para evitar
romper la compatibilidad existente.

---

## Diagnóstico de `answer_relevancy`

### Distribución real (v1.21 baseline, 65 queries)

La distribución es bimodal — **no** es una calidad media baja uniforme:

```
score=0.000     ███████████  11 queries (17%)  ← outliers que hunden la media
score=0.6–0.8  ████████████  12 queries (18%)
score=0.8–0.9  ████████       8 queries (12%)
score=0.9–1.0  ████████████████████████████████  34 queries (52%)
```

Sin los 11 outliers: **media = 0.906** (por encima del objetivo de 0.85).

### Causas de los 11 scores = 0.000 (diagnóstico v1.21)

| Causa | Queries | Descripción |
|-------|---------|-------------|
| Corpus incompleto | q039, q041, q042 | LLM responde "No encuentro esta información". RAGAS no puede generar preguntas relevantes desde "No sé". |
| Contaminación por citas | q056 | Sin citas: 0.000 → 0.831. Las citas dominan el texto y RAGAS genera preguntas sobre metadatos de fuente. |
| Preguntas ambiguas | q048, q050 | `ambiguity_test` — diseñadas para ser polisémicas. El LLM cubre múltiples conceptos → RAGAS no converge. |
| Incompatibilidad RAGAS | q013, q032, q038, q054, q065 | Preguntas meta, de síntesis amplia o de tablas. Respuestas correctas pero RAGAS no puede generar una pregunta sintética convergente. |

q056 se resolvió con `answer_for_eval`. Los 10 restantes son **no aplicables estructuralmente**
y están clasificados como tal en `data/benchmark_queries.yaml` desde v1.21.1.

### Por qué `answer_for_eval` mejora la métrica

`answer_relevancy` funciona así: RAGAS genera N preguntas sintéticas desde el texto de la
respuesta y mide su similitud con la pregunta original. Si la respuesta contiene texto de
citas como `[[3] Fuente: SDMX_Glossary | Sección: SDMX Information Model | Líneas: 6283-6321]`,
RAGAS puede generar preguntas como "¿Qué sección del SDMX_Glossary describe el Information
Model?" — completamente irrelevante para la pregunta original "What is the SDMX information
model and what are its layers?".

Con `answer_for_eval` (citas eliminadas): +0.028 de media global, y algunos casos como q056
mejoran +0.83.

---

## Applicability reporting (v1.21.1)

`answer_relevancy` no es una métrica válida para todas las queries de la suite oficial.
**10 queries** tienen `ragas.answer_relevancy_applicable: false` en
`data/benchmark_queries.yaml`. La métrica global (65 queries) es útil para comparaciones
históricas, pero la **métrica aplicable (55 queries)** es el indicador principal de calidad.

### Categorías de no-aplicabilidad

| `applicability_reason` | Queries | Significado | `decision` |
|------------------------|---------|-------------|------------|
| `meta_synthesis` | q013, q032, q054, q065 | Pregunta de síntesis amplia o meta-pregunta; RAGAS no converge | `evaluator_limitation` |
| `ragas_evaluator_limitation` | q038 | Respuesta correcta pero RAGAS no genera pregunta sintética convergente para tablas/enumeraciones | `evaluator_limitation` |
| `structured_reference_missing_corpus` | q039, q041, q042 | La referencia estructurada (codelist, tabla de valores) no está accesible en el corpus Markdown | `needs_corpus_expansion` |
| `ambiguity_test` | q048, q050 | Diseñadas para ser polisémicas; el LLM cubre múltiples sentidos → RAGAS no converge | `keep_as_stress_test` |

### Métricas recomendadas

| Métrica | Descripción | Cuándo usar |
|---------|-------------|-------------|
| `answer_relevancy_all` | Media sobre las 65 queries | Comparación histórica cross-version |
| `answer_relevancy_applicable` | Media sobre las 55 queries aplicables | **Indicador principal de calidad** |
| `faithfulness_all` | Media sobre las 65 queries | Faithfulness no tiene restricción de aplicabilidad |

### Acciones futuras por grupo (no implementadas)

- **`needs_corpus_expansion` (q039, q041, q042):** añadir documentos Markdown con codelists
  SDMX estructurados (OBS_STATUS, header elements, namespace prefixes). Sin reingestión de
  documentos existentes.
- **`keep_as_stress_test` (q048, q050):** mantener como stress test de ambigüedad. Crear
  variantes `candidate` con preguntas sin ambigüedad para medir esa dimensión limpiamente.
- **`evaluator_limitation` (q013, q032, q038, q054, q065):** evaluar con rúbrica de
  synthesis/completeness, no con `answer_relevancy`. Pendiente de diseño de rúbrica.

### Ejecutar con informe de aplicabilidad

```bash
conda activate ragas
python scripts/ragas_eval.py \
  --input  data/eval_runs/v1.21.1_applicability.jsonl \
  --metrics faithfulness,answer_relevancy \
  --output data/eval_runs/v1.21.1_applicability_ragas.json
```

El script carga automáticamente `data/benchmark_queries.yaml` y muestra:
- Tabla `ALL queries` (65)
- Tabla `APPLICABLE only` (55) — **métrica principal**
- Tabla `NOT APPLICABLE` (10) — scores preservados, visibles para referencia
- Lista detallada de no-aplicables con `reason` y `decision`

Para deshabilitar el splitting: `--queries-yaml none`.
