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
  "answer":          "SDMX (Statistical Data and Metadata eXchange) is...",
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
