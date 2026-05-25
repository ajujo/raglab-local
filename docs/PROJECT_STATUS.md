# RAG-Lab — Estado del Proyecto

**Versión actual:** v1.18.1  
**Estado:** Estable — cerrado — apto para uso real controlado  
**Fecha de cierre:** 2026-05-23  

---

## Resumen ejecutivo

RAG-Lab es un sistema RAG (Retrieval-Augmented Generation) en producción local para
consultar documentación técnica SDMX en Markdown. A partir de v1.18.1 el sistema está
completo para uso real controlado: retrieval híbrido calibrado, verificador de citas
activo, capa de confianza auditada E2E, y feedback recogido (pero no usado aún para
re-ranking, por decisión explícita).

No se abren nuevas funcionalidades hasta que el uso real genere problemas concretos
que justifiquen nuevas fases.

---

## Corpus actual

| Documento | Chunks | Sparse | Embedding model |
|-----------|--------|--------|----------------|
| Notas_Tecnicas_SDMX_2.1 | 115 | ✓ | BAAI/bge-m3 v2024-09 |
| SDMX-Training-introduction-2015 | 18 | ✓ | BAAI/bge-m3 v2024-09 |
| SDMX_2-1_User_Guide_6 | 197 | ✓ | BAAI/bge-m3 v2024-09 |
| SDMX_Glossary | 277 | ✓ | BAAI/bge-m3 v2024-09 |
| SDMX_Glossary_Test | 3 | ✓ | BAAI/bge-m3 v2024-09 |
| **Total** | **610** | **100%** | |

Stores sincronizados: DocStore (SQLite) = ChromaDB = FTS5 index = Sparse BLOBs = 610.

---

## Retrieval activo

| Componente | Configuración | Estado |
|-----------|---------------|--------|
| Embedding | BAAI/bge-m3 (dense + sparse) | activo |
| Dense search | ChromaDB cosine, HNSW M=16 ef=100 | activo |
| BM25 | FTS5 SQLite | activo |
| Sparse | BGE-M3 sparse vectors (SQLite BLOB, `sparse_scorer.py`) | activo |
| Fusión | RRF (k=60), top_k=50 candidates | activo |
| Diversidad | MMR post-RRF | activo |
| Reranker | BAAI/bge-reranker-v2-m3, heading context | activo |
| HyDE | implementado | **desactivado** (`HYDE_ENABLED=False`) |
| Query rewriting | implementado | **desactivado** por defecto |
| Query variants | implementado | **desactivados** por defecto |

---

## Verificador de respuestas y citas (v1.18+)

| Capa | Estado |
|------|--------|
| Citation verifier | activo — regex `[[N] Fuente: … | Sección: … | Líneas: …]`, VALID/PARTIAL/INVALID |
| Consistency check | activo — segundo LLM call, detección de hallucinations |
| Scoring | activo — citation 35% + retrieval 30% + consistency 25% + coverage 10% |
| `evidence_map` | disponible — `VerificationResult.evidence_map` (computed property) |
| `format_verification_block(verbose=True)` | disponible — trazabilidad por cita |

Auditoría E2E v1.18.1: **10/10 PASS** (3 easy, 3 technical, 2 Spanish, 1 ambiguous,
1 out-of-corpus). Informe: `data/audits/v1.18.1_answer_verifier_e2e.json`.

---

## Benchmark oficial (baseline activo: v1.11)

Suite: `official`, variante `full`, 65 queries, no-cache.

| Métrica | Valor |
|---------|-------|
| R@5 | 0.821 |
| R@10 | 0.896 |
| R@30 | 0.978 |
| MRR | 0.939 |
| nDCG@10 | 0.837 |
| P50ms | ~250 |

Sin regresión desde v1.11. Baseline: `data/baselines/v1.11_official_full_eval.json`.

---

## Feedback

| Aspecto | Estado |
|---------|--------|
| Recogida | activa — `FeedbackStore`, SQLite, 8 tipos de evento |
| Análisis | `python -m rag_lab.feedback.analyze_feedback` |
| Uso como señal | **NO activo** — recogida pero sin efecto sobre ranking |

El feedback acumulado servirá de base para decisiones de re-ranking cuando haya
suficiente volumen y análisis real. Ver sección "Qué NO está activado".

---

## Tests

| Suite | Tests | Estado |
|-------|-------|--------|
| `pytest tests/ -q` | 972 | ✓ |
| `tests/test_verification/` | 47 | ✓ |
| `tests/test_cli/test_audit_script.py` | 21 | ✓ |
| `tests/test_cli/test_store_isolation_guard.py` | 7 | ✓ |

---

## Qué NO está activado todavía

Las siguientes funcionalidades están **explícitamente congeladas** hasta nueva orden:

1. **Feedback como señal de re-ranking.**  
   El `FeedbackStore` recoge eventos, pero ningún componente del pipeline consulta
   ese feedback para modificar scores o ranking. Activar esto requiere un benchmark
   A/B completo antes de mergearlo.

2. **Ingesta directa de PDF, DOCX, HTML.**  
   No existen loaders para estos formatos. Toda ingesta debe pasar por Markdown limpio
   y validado. Añadir loaders requiere análisis de calidad de la extracción (heading
   structure, tables, formulas) antes de tocar el chunker.

3. **Cualquier automatismo que modifique el ranking basándose en feedback.**  
   Boost/penalización por `query_hash` match, clustering de feedback, etc.
   Ninguno de estos mecanismos está implementado ni parcialmente activo.

4. **Cualquier loader que ingeste documentos sin pasar antes por Markdown limpio y
   validado.** El gate de calidad (`--strict`) debe ejecutarse antes de ingestar.

5. **HyDE activado por defecto.**  
   HyDE está implementado y funciona, pero los benchmarks mostraron R@5 -0.038 y
   latencia ×12.5 respecto al baseline. Queda disponible vía `--hyde` para casos
   concretos donde se justifique.

6. **Query rewriting / query variants por defecto.**  
   Ambos disponibles vía flags (`--rewrite`, `QUERY_VARIANT_*`), pero desactivados
   en producción hasta que haya un caso de uso que justifique la latencia adicional.

---

## Protocolo de uso real controlado

### Añadir un documento nuevo

```bash
# 1. Validar calidad del Markdown antes de ingestar
rag-lab docs validate path/to/doc.md --strict

# 2. Previsualizar la estructura de chunks resultante
rag-lab docs preview-chunks path/to/doc.md

# 3. Añadir al catálogo de documentos
rag-lab docs add path/to/doc.md --tag sdmx

# 4. Ingestar
python -m rag_lab.cli ingest --doc path/to/doc.md

# 5. Verificar integridad después del ingesto
rag-lab doctor
rag-lab reconcile --check
```

### Después de cualquier cambio en stores o configuración

```bash
rag-lab doctor          # health check completo
rag-lab reconcile --check   # confirmar DocStore=ChromaDB=FTS5=Sparse
```

### Registrar feedback cuando una respuesta falle

```bash
# Feedback sobre un chunk concreto (chunk_id visible en el bloque de verificación)
rag-lab feedback add --chunk-id <id> --type wrong_doc
rag-lab feedback add --chunk-id <id> --type bad_citation

# Ver estadísticas acumuladas
rag-lab feedback stats
python -m rag_lab.feedback.analyze_feedback
```

### Auditar la capa de verificación periódicamente

```bash
python scripts/audit_answer_verifier.py --suite answer_e2e
```

### Antes de cualquier cambio en retrieval

```bash
# 1. Ejecutar benchmark de referencia
rag-lab benchmark run --suite official --variants full --no-cache

# 2. Aplicar el cambio

# 3. Ejecutar benchmark de nuevo
rag-lab benchmark run --suite official --variants full --no-cache

# 4. Comparar. Aceptar solo si Δ ≥ 0 en R@5, MRR y nDCG@10.
```

### Nunca hacer

- Ingestar PDF/DOCX/HTML directamente sin conversión a Markdown limpio.
- Activar feedback como señal de re-ranking sin benchmark A/B previo.
- Modificar pesos RRF/MMR/scoring sin documentar el baseline antes.
- Ignorar warnings del verificador de citas en producción.

---

## Historial de versiones relevante

| Versión | Cambio principal |
|---------|-----------------|
| v1.18.2 | Remove legacy SparseStore (JSON) — SQLite BLOBs son canónicos |
| v1.18.1 | E2E audit verificador citas — 10/10 PASS; script + docs |
| v1.18 | Verification hardening — 4 bugs, evidence_map, verbose trace, prompt hardening |
| v1.17 | Release candidate audit — guard tests, docs, smoke E2E |
| v1.16.3 | `--repair-metadata`, `benchmark run` alias, diagnose metadata display |
| v1.16.2 | Exponer doctor/benchmark/reconcile/diagnose en CLI (`rag-lab <cmd>`) |
| v1.16.1 | FTS5 idempotency fix (DELETE-then-INSERT) |
| v1.16 | Parallel + resumable ingest (`--workers N`) |
| v1.15 | Feedback capture (recogida, sin efecto en ranking) |
| v1.14.1 | Cache metadata invalidation (tags + deletes invalidan caché) |
| v1.14 | Caché persistente de retrieval/reranking (SQLite, TTL 7 días) |
| v1.13.1 | Fix HNSW false mismatch warning |
| v1.13 | HNSW configurable, perfiles de benchmark |
| v1.12 | HyDE + query rewriter (implementados, HyDE desactivado por benchmark) |
| v1.11 | Query variants cleanup — **baseline activo de CI** |
| v1.10 | Reranker structural context (heading path prefix) |

---

## Próximo trabajo recomendado

El sprint v1.18.1 cierra la fase de infraestructura y verificación.

**Recomendación:** pausa activa de desarrollo de features. Usar el sistema en producción
real, observar fallos, recoger feedback, y dejar que los problemas reales dicten la
siguiente fase.

Señales que justificarían abrir v1.19:

- Feedback acumulado suficiente (>50 eventos) con patrón claro de fallos evitables.
- Corpus nuevo (documento real a ingestar) que revele un problema de chunking o calidad.
- Una pregunta de usuario que el sistema falle sistemáticamente con HIGH confidence
  (falso positivo del verificador).
- Necesidad real de ingestar PDF/DOCX (no anticipada).

Hasta entonces: usar, observar, registrar feedback.
