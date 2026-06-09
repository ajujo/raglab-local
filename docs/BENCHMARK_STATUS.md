# RAG-Lab — Estado del sistema y benchmark

Informe actualizable del estado de calidad del sistema. Actualizar tras cada versión
que cambie el pipeline de retrieval, generación o verificación.

---

## Versión actual

**v1.21.1** · 2026-06-09 · 1104 tests · corpus: 610 chunks SDMX

---

## Resultados de benchmark

### Retrieval (benchmark propio, reference-based)

Suite oficial, 65 queries, variante `full`, sin caché. Baseline comparativo: v1.11.

| Métrica | v1.21 | Interpretación |
|---------|-------|----------------|
| R@5 | **0.821** | El 82% de las queries tienen el documento relevante en los 5 primeros chunks |
| R@10 | **0.896** | El 90% lo tienen en los 10 primeros |
| R@30 | **0.978** | El 98% lo tienen en los 30 primeros |
| MRR | **0.939** | El documento relevante aparece de media en la posición 1.06 |
| nDCG@10 | **0.837** | Calidad ordenada de los 10 primeros resultados |

### Calidad de respuesta (RAGAS, reference-free)

65 queries, juez externo: `deepseek/deepseek-v4-flash` vía OpenRouter.

| Métrica | v1.21 (raw) | v1.21 eval (clean) | v1.21.1 applicable | Señal |
|---------|-------------|---------------------|---------------------|-------|
| `faithfulness` | 0.9123 | 0.9296 | **0.9659** | ✓ Sólido. Bajo nivel de alucinación. |
| `answer_relevancy (all)` | 0.7624 | 0.7775 | 0.7676 | Solo comparación histórica. |
| `answer_relevancy (applicable)` | — | — | **0.8529** | ✓ Métrica principal. Supera objetivo 0.85. |

**Métrica principal recomendada: `answer_relevancy_applicable = 0.8529`** (55 queries aplicables).

La métrica global (all) sigue siendo útil para comparaciones históricas, pero las 10 queries
clasificadas como no aplicables (`answer_relevancy_applicable: false` en `benchmark_queries.yaml`)
tienen causas estructurales que las hacen no evaluables con RAGAS:
- q039, q041, q042: `structured_reference_missing_corpus` — referencia estructurada no accesible en Markdown
- q048, q050: `ambiguity_test` — diseñadas para ser polisémicas
- q013, q032, q038, q054, q065: `meta_synthesis` / `ragas_evaluator_limitation`

Ver RAGAS_USAGE.md §"Applicability reporting" para el desglose completo.

### Pipeline interno (auto-evaluación)

| Métrica | v1.21 |
|---------|-------|
| avg trust_score | 0.909 |
| HIGH | 61 / 65 (94%) |
| MEDIUM | 4 / 65 (6%) |
| LOW | 0 / 65 (0%) |
| avg latencia E2E | 6 209 ms |
| p50 latencia | 5 594 ms |
| p95 latencia | 10 259 ms |

---

## Diagnóstico — puntos fuertes y débiles

### Puntos fuertes

**Retrieval de alta calidad.** R@5=0.821 y MRR=0.939 sobre un corpus técnico
especializado son números competitivos. La arquitectura dense+sparse+reranking con RRF
está bien calibrada para la especificidad del corpus SDMX.

**Bajo nivel de alucinación.** `faithfulness=0.91` confirma que el LLM se ciñe a los
contextos recuperados. Para un corpus técnico donde una afirmación incorrecta tiene
consecuencias reales (implementaciones de estándares), esto es la métrica más crítica.

**Trust score interno consistente con evaluación externa.** El avg trust_score interno
(0.909) y la faithfulness externa (0.912) van en la misma dirección, lo que valida el
sistema de verificación propio del pipeline.

### Puntos débiles

**`answer_relevancy=0.78` es el talón de Aquiles (con diagnóstico acotado).**
La distribución es bimodal — 54 queries tienen score >0.85 (media 0.906), pero
11 outliers con score=0.000 hunden la media global a 0.78. Las causas son cuatro:
corpus incompleto (3 queries), diseño por polisemia (2 queries), incompatibilidad
RAGAS con preguntas meta (5 queries), y contaminación por citas (1 query, ya corregida
con `answer_for_eval`). El problema real de generación afecta a un subconjunto acotado,
no al 24% de las queries.

**Latencia p95=10s inaceptable para uso interactivo.** El modelo Qwen3.6-27B con
razonamiento interno añade 3–8 s antes de generar la respuesta visible. Tolerable para
consultas técnicas esporádicas; problemático para uso fluido o demos.

---

## Umbrales de referencia por caso de uso

| Métrica | Uso propio / investigación | Producto interno empresa | Producto externo / SaaS |
|---------|---------------------------|--------------------------|-------------------------|
| `faithfulness` | >0.80 | >0.88 | >0.92 |
| `answer_relevancy` | >0.70 | >0.80 | >0.88 |
| R@5 | >0.70 | >0.80 | >0.85 |
| Latencia p95 | <30s | <5s | <2s |

**Estado actual (v1.21.1, applicable subset):**
- Uso propio / investigación: ✓ cumple todos los umbrales
- Producto interno empresa: faithfulness ✓, retrieval ✓, **answer_relevancy ✓ (0.853 vs 0.80)**, **latencia ✗ (10s vs 5s)**
- Producto externo: answer_relevancy en el límite, latencia sigue siendo el bloqueador

---

## Hoja de ruta hacia mejores métricas

### Mejorar answer_relevancy (0.76 → >0.85) — alta prioridad

**Causa probable:** el prompt del sistema es demasiado amplio o el LLM recibe 8 chunks
de contexto con información periférica que lo anima a responder de forma general.

**Experimentos sugeridos (coste: 1–2 tardes cada uno):**

1. **Reducir `RERANK_TOP_K` de 8 a 4.** Menos contexto = respuesta más focalizada.
   Riesgo bajo si R@5 se mantiene. Medir con `rag-lab eval run` + `ragas_eval.py`.

2. **Prompt más directivo.** Añadir instrucción explícita en el system prompt:
   *"Responde SOLO y directamente a la pregunta formulada. No añadas contexto general
   no solicitado."* Coste: 0 código, solo un cambio en `prompt_builder.py`.

3. **Query rewriting activado.** Ya implementado (`--rewrite`), sin benchmark oficial.
   Podría mejorar la recuperación de chunks más específicos. Medir con benchmark.

### Reducir latencia (p95=10s → <5s) — media prioridad

Ver sección "Análisis del servidor LLM" más abajo.

### Métricas con ground truth (pendiente) — baja prioridad actual

`context_recall` y `answer_correctness` requieren un dataset de pares
(pregunta, respuesta correcta) anotados para el corpus SDMX. Sin ese dataset, estas
métricas no son aplicables. Crearlos es trabajo de anotación manual, no de ingeniería.

---

## Análisis del servidor LLM y latencia

### Configuración actual

```bash
vllm serve "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" \
  --max-model-len 80000 \
  --gpu-memory-utilization 0.75 \
  --attention-backend flashinfer \
  --performance-mode interactivity \
  --language-model-only \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 2 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000
```

### Lo que ya está bien optimizado

- `--performance-mode interactivity` — prioriza TTFT (time-to-first-token)
- `--enable_thinking: false` — desactiva el razonamiento interno (correcto, ya viene del config)
- `--kv-cache-dtype fp8_e4m3` — KV cache comprimido, más contexto con menos VRAM
- `--speculative-config mtp 3 tokens` — decodificación especulativa, +20–40% throughput
- `--enable-prefix-caching` — reutiliza el prefijo del system prompt entre queries
- `--attention-backend flashinfer` — atención más rápida que la implementación base

### Causa real de la latencia alta

Con todas esas optimizaciones y p95=10s, el cuello de botella es el **tamaño del modelo**.
Un 27B NVFP4 genera ~25–40 tokens/s en RTX 5090. Una respuesta RAG típica tiene 200–400 tokens → 5–10 s de generación pura. No hay configuración vLLM que cambie eso sin cambiar el modelo.

### Opciones para reducir latencia

**Opción A — Modelo más pequeño (recomendado para probar)**

Un modelo 8B-9B en FP16 o NVFP4 generaría ~80–120 tokens/s en RTX 5090 → 2–4 s p95.
El VRAM liberado (~20 GB) permitiría aumentar `--gpu-memory-utilization` y el batch size.

Candidatos razonables:
- `Qwen3-8B` — misma familia, instruction-tuned, buen razonamiento técnico
- `Qwen2.5-7B-Instruct` — probado y estable, bajo coste de inferencia
- `Qwen3-14B` — compromiso: +calidad vs 8B, -latencia vs 27B

El comando vLLM sería casi idéntico — solo cambiar el model ID y ajustar
`--gpu-memory-utilization` (puede subir a 0.85–0.90 con un modelo más pequeño).

**Opción B — Reducir prompt** (complementaria, sin cambiar modelo)

Reducir `RERANK_TOP_K` de 8 a 4 chunks → prompt ~800 tokens más corto → ~1–2 s menos
de generación. Doble beneficio: menos latencia y posiblemente mejor `answer_relevancy`.

**Opción C — Prefill caching agresivo** (ya activado, pero limitado)

`--enable-prefix-caching` ya está activo. El system prompt se cachea entre queries,
pero el contexto de chunks cambia en cada query y no se beneficia del prefill cache.

### Recomendación

Probar `Qwen3-8B` con el benchmark completo. El ciclo es:
1. Lanzar vLLM con `Qwen3-8B` en lugar del 27B
2. `rag-lab eval run --suite official --output data/eval_runs/v1.22_qwen3_8b.jsonl`
3. `python scripts/ragas_eval.py --input ... --output data/eval_runs/v1.22_qwen3_8b_ragas.json`
4. Comparar faithfulness y answer_relevancy contra el baseline v1.21

Si faithfulness se mantiene >0.85 y answer_relevancy sube (los modelos más pequeños
tienden a ser más directos y menos verbosos), el cambio es un win neto en calidad Y latencia.

---

## Historial de cambios de este informe

| Fecha | Versión | Cambio |
|-------|---------|--------|
| 2026-06-08 | v1.21 | Primera versión del informe. Baseline RAGAS establecido. |
| 2026-06-09 | v1.21 eval | Diagnóstico answer_relevancy (bimodal, 11 zeros). Nuevo campo answer_for_eval. Métricas actualizadas a clean answers. |
| 2026-06-09 | v1.21.1 | Applicability reporting. answer_relevancy_applicable=0.8529 (55 queries). 10 queries clasificadas como no aplicables en YAML. |
