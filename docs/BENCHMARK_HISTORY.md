# RAG-Lab — Benchmark History

Registro histórico de las métricas de evaluación por versión.
Cada entrada es reproducible con los comandos al pie.

---

## Cómo reproducir un benchmark completo

```bash
# 1. Capturar salida del pipeline (env rag-lab)
conda activate rag-lab
rag-lab eval run --suite official --output data/eval_runs/<version>_baseline.jsonl

# 2. Evaluar con RAGAS (env ragas)
conda activate ragas
python scripts/ragas_eval.py \
  --input  data/eval_runs/<version>_baseline.jsonl \
  --metrics faithfulness,answer_relevancy \
  --output data/eval_runs/<version>_baseline_ragas.json
```

Tiempo estimado: ~13 min pipeline + ~10 min RAGAS = ~25 min total.

---

## Métricas de retrieval (benchmark propio)

Suite oficial, 65 queries, variante `full`, sin caché. Baseline: v1.11.

| Versión | R@5 | R@10 | R@30 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| v1.11 (baseline) | 0.821 | 0.896 | 0.978 | 0.939 | 0.837 |

Reproducir:
```bash
rag-lab benchmark --suite official --variants full --no-cache
```

---

## Métricas de calidad de respuesta (RAGAS)

Suite oficial, 65 queries, juez externo DeepSeek v4 Flash vía OpenRouter.
**Reference-free** — no requiere respuestas de referencia.

| Versión | faithfulness | answer_relevancy | n queries | juez | fecha | notas |
|---|---|---|---|---|---|---|
| **v1.21** | **0.9123** | **0.7624** | 65 | deepseek/deepseek-v4-flash | 2026-06-08 | Baseline |
| v1.22-E1 | 0.9250 ↑ | 0.7673 ↑ | 65 | deepseek/deepseek-v4-flash | 2026-06-09 | RERANK_TOP_K=4. Leve mejora en ambas, latencia p50 empeora. |
| v1.22-E2 | 0.9032 ↓ | 0.7408 ↓ | 65 | deepseek/deepseek-v4-flash | 2026-06-09 | System prompt directivo. Latencia p50 −22% pero ambas métricas RAGAS empeoran. |
| v1.22-E3 | 0.8357 ↓↓ | 0.7192 ↓↓ | 65 | deepseek/deepseek-v4-flash | 2026-06-09 | User prompt concisión. Faithfulness colapsa, descartado. |

### Guía de interpretación

| Métrica | Qué mide | Escala orientativa |
|---|---|---|
| `faithfulness` | Fracción de statements respaldados por los contextos recuperados. Mide alucinación. | <0.80 preocupante · 0.80–0.90 aceptable · >0.90 bueno |
| `answer_relevancy` | Grado en que la respuesta aborda directamente la pregunta formulada. | <0.70 débil · 0.70–0.85 aceptable · >0.85 bueno · >0.90 excelente |

### Métricas de pipeline (internas, v1.21 baseline)

| Métrica | Valor |
|---|---|
| avg trust_score | 0.909 |
| HIGH | 61 / 65 |
| MEDIUM | 4 / 65 |
| LOW | 0 / 65 |
| avg latency (E2E) | 6209 ms |
| p50 latency | 5594 ms |
| p95 latency | 10259 ms |

> **Nota latencia:** el modelo Qwen3.6-27B con modo thinking activo añade 3–8 s de razonamiento
> interno antes de generar la respuesta visible. La latencia de retrieval puro (sin LLM) es <200 ms.

---

## Notas metodológicas

- Los benchmarks de retrieval (R@5, MRR, nDCG@10) y los de calidad de respuesta (RAGAS)
  miden dimensiones distintas y no son directamente comparables.
- `faithfulness` y `answer_relevancy` son **reference-free** — no requieren respuestas
  de referencia anotadas. Esto los hace más fáciles de mantener pero ligeramente
  más ruidosos que métricas supervisadas.
- El juez LLM introduce varianza. Comparar versiones siempre con el mismo modelo de juez.
- `context_recall` y `answer_correctness` (métricas con ground truth) quedan pendientes
  hasta que exista un dataset anotado de pares (pregunta, respuesta correcta).
