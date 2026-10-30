# Cierre P0-C — propiedad documental e integridad de datos

> Fecha: 2026-10-30 · Rama: `p0c-closure-20261030`
> Estado: **CERRADO con dictamen VERDE** de revisión independiente (ronda 2).
> Base: auditoría adversarial del 18-10-2026 (dictamen ROJO, 26 hallazgos) y
> candidato P0-C del 20-10-2026 (ROJO por R2-F1 y R2-F2).

## 1. Resumen

El repositorio estaba **dos iteraciones por detrás** del trabajo auditado. El
working tree contenía el estado de P0 ronda 1 —el que la revisión independiente
dictaminó ROJO con un hallazgo CRÍTICO (`R3-01`: un intento `FAILED` de otra ruta
concedía propiedad sobre un documento ajeno)— mientras las correcciones
posteriores vivían congeladas fuera del repo, en `~/audits/rag-lab/`.

Este cierre: aplica el linaje completo al repositorio, corrige los tres bloqueantes
que quedaban abiertos, los fija con regresiones, los somete a revisión adversarial
independiente y deja el corpus real verificado y en uso.

| Estado | Tests | Dictamen |
|---|---|---|
| P0 r1 (estado heredado) | 1176 | ROJO — `R3-01` CRITICAL |
| P0-B r3 (congelado) | 1229 | ROJO — `R3-F1..F4` |
| P0-C (congelado) | 1264 | ROJO — `R2-F1`, `R2-F2` |
| **Este cierre** | **1289** | **VERDE** |

## 2. Qué se ha corregido

### R2-F1 — Contradicción de hashes sin filas de batch

`ownership.decide()` comparaba `document_versions.content_hash` con
`documents.content_hash` solo a través de `committed_hashes`, que sale de las
filas de `ingest_documents`. La ruta `retry` crea versión y metadatos **sin**
esas filas: el estado normal es `committed_hashes == []`, de modo que ambos
hashes nunca se contrastaban y una contradicción permitía publicar una generación
nueva.

**Corrección:** comparación directa cuando ambos hashes existen, con
independencia de las filas de batch. Discrepancia ⇒ `DENY_INCOHERENT`.

### R2-F2 — CREATE sobre índices huérfanos

La rama CREATE retornaba antes de mirar FTS y Chroma: una fila fantasma en
`chunks_fts` o un vector huérfano en Chroma permitían publicar sobre un índice
incoherente.

**Corrección:** CREATE exige `fts_matches`, `chroma_verifiable` y
`chroma_matches`. «Documento nuevo» significa que no existe nada de él en
**ningún** almacén, no solo en SQLite.

### CLOSURE-F1 — El legacy no corroboraba el hash documental

Hallazgo ALTO de la primera ronda de revisión, **preexistente** (no introducido
por los dos anteriores). En la rama legacy (`has_version=False`) se rechazaban
hashes COMMITTED distintos entre sí, pero `documents.content_hash` no se
contrastaba con ellos: con hashes de batch internamente uniformes, una
contradicción documental se publicaba en lugar de rechazarse.

**Corrección:** el hash documental debe estar corroborado por el run propietario
COMMITTED antes de conceder `REPLACE`/`NOOP`.

### Dos defectos colaterales descubiertos durante la integración

- **RAG-18 (cosmético, visible):** el bloque `if __name__ == "__main__": app()`
  estaba a mitad de `cli.py`, por encima de `_print_feedback_hints`. Con
  `python -m rag_lab.cli` el `NameError` se capturaba y se imprimía como «Error
  en la capa de verificación» al final de **cada** consulta. Movido al final.
- **Aislamiento de tests (integridad):** `manifest.py` y `cleaner.py` resolvían
  `DATA_DIR` al importarse, así que la suite escribía entradas de prueba en
  `data/ingested.jsonl` y ficheros en `data/cleaned` **reales**. Ahora se resuelve
  en tiempo de llamada desde `rag_lab.config`, con el atributo de módulo como
  override prioritario. Verificado: `data/ingested.jsonl` y
  `storage/docstore.sqlite` conservan su sha256 antes y después de la suite.

## 3. Contrato de propiedad aplicado

Derivado de `document_versions.active_run_id → ingest_runs COMMITTED →
source_path`; para datos anteriores al versionado, del último run COMMITTED y, en
su defecto, de `documents.path`.

| Caso | Situación | Decisión |
|---|---|---|
| A | `doc_id` libre y sin restos en ningún almacén | `CREATE` |
| B | Propietario + resultado persistido idéntico | `NOOP_OWNER` |
| C | Propietario + resultado distinto | `REPLACE` (versionado transaccional) |
| D | Otra ruta + equivalencia exacta persistida | `NOOP_EQUIVALENT_OTHER_SOURCE` — **sin ninguna escritura documental, sin cambio de propietario** |
| E | Otra ruta + contenido distinto | `DENY_NOT_OWNER` |
| F | Legacy: equivalencia exacta | `NOOP`; evidencia insuficiente ⇒ `DENY` |
| G | Transferencia de propiedad | fuera del camino automático; operación administrativa explícita |

Contradicciones de hash, incoherencia SQL/FTS/Chroma, hash aislado sin
corroboración, operación pendiente de recuperación o intento `FAILED` ⇒ `DENY`.
El manifiesto es advisory, nunca autoridad de propiedad.

## 4. Archivos modificados

| Archivo | Tipo | Motivo |
|---|---|---|
| `rag_lab/ingest/ownership.py` | nuevo | núcleo `decide()` + adaptador SQLite/FTS/Chroma |
| `rag_lab/cli_ingest.py` | mod. | decide en batch y retry; SKIPPED por idempotencia; revalidación TOCTOU |
| `rag_lab/ingest/versioning.py` | nuevo | versionado con pre-imagen, journal y recovery |
| `rag_lab/ingest/transaction.py` | mod. | `content_hash` en `commit_replace`; fail-closed |
| `rag_lab/storage/docstore.py` | mod. | migración v7 (`content_hash` ligado a la versión) |
| `rag_lab/storage/{fts_store,vector_store}.py` | mod. | invariante `None` vs `[]`, borrado por versión |
| `rag_lab/ingest/{manifest,cleaner}.py` | mod. | `DATA_DIR` resuelto al llamar |
| `rag_lab/utils/tokenizer.py` | mod. | cota por bytes (no puede subestimar) |
| `rag_lab/generation/prompt_builder.py` | mod. | presupuesto de contexto (RAG-08) |
| `rag_lab/generation/llm_client.py` | mod. | presupuesto en todas las entradas |
| `rag_lab/config.py` | mod. | parámetros de ventana y presupuesto |
| `rag_lab/cli.py` | mod. | RAG-18 + chunks usados en la verificación |
| `tests/test_ingest/test_p0c_ownership.py` | nuevo | 60 tests del contrato (A–F, CLOSURE-F1, núcleo puro) |
| `tests/test_ingest/test_p0b_hardening.py`, `test_p0b_round2.py` | nuevos | regresiones P0-B |
| `tests/test_ingest/{test_identity_p0,test_versioning_p0,test_p0_hardening}.py` | nuevos | regresiones P0 |
| `tests/test_retrieval/test_filter_invariants.py` | nuevo | invariante de filtros (RAG-03) |
| `tests/test_generation/test_context_budget_p0.py` | nuevo | presupuesto de contexto (RAG-08) |

## 5. Matriz hallazgo → corrección → prueba → resultado

| ID | Corrección | Prueba | Resultado |
|---|---|---|---|
| RAG-01 | versionado con pre-imagen; restore en rollback | `test_versioning_p0`, `test_transaction`, `test_p0_hardening` | cerrado |
| RAG-02 | rollback de run no vigente restaura, no borra | `TestRollbackVersionSafety` | cerrado |
| RAG-03 | `None` = sin filtro, `[]` = conjunto vacío | `test_filter_invariants` | cerrado |
| RAG-04 | reemplazo exacto retira obsoletos | `TestReplacementDropsObsolete` | cerrado |
| RAG-05 | identidad por frontmatter doc_id + ownership | `test_identity_p0`, `test_p0c_ownership` | cerrado |
| RAG-08 | presupuesto de contexto y salida | `test_context_budget_p0` | cerrado |
| SOL-07 | contrato de propiedad (`ownership.decide`) | `test_p0c_ownership` (60) | cerrado |
| R3-F1..F4 | agregación sin `LIMIT`, huellas exactas, sin hash aislado, sin manifiesto | `test_p0b_hardening`, `test_p0b_round2` | cerrado |
| R2-F1 | hashes persistidos comparados directamente | `test_contradictory_persisted_hashes_denied_without_batch_rows` (4) | **cerrado** |
| R2-F2 | CREATE valida índices | `test_orphan_index_entry_blocks_create` (8) | **cerrado** |
| CLOSURE-F1 | legacy corrobora hash documental | `test_legacy_contradictory_documents_hash_denied` (4) + control positivo | **cerrado** |
| RAG-18 | guard `__main__` al final | `tests/test_cli` (71) | cerrado |

## 6. Verificación

**Suite completa** (env `rag-lab`, CPU, 2026-10-30):

```
1289 passed, 0 failed, 0 errors — 665.35 s
```

**No vacuidad de las regresiones nuevas:** contra el código anterior a la
corrección, **13 de 15** tests de R2 fallan (los 2 que pasan son controles
positivos por diseño).

**Sondas independientes propias** (`scripts/independent_probe.py`): **30/30 PASS**
—R2-F1 y R2-F2 en batch y retry, estado documental idéntico antes/después,
generación intacta, y CREATE legítimo funcionando.

**Sondas históricas P0-B intactas:** 116 passed / 11 failed. Los 11 fallos son
expectativas antiguas de materialización que el contrato nuevo sustituye por
NO-OP sin escritura; documentados en `p0c-20261020/evidence/CONTRACT-CHANGE.md` y
**no** alterados para «hacerlos pasar».

**Integridad de stores reales** (`reconcile --check`):

```
DocStore (SQLite): 610 chunks · ChromaDB: 610 · FTS5: 610 · Sparse BLOBs: 610 (100%)
✓ DocStore and ChromaDB are consistent.
```

**Comportamiento del corpus real bajo el contrato** (5 documentos, sin escribir):

| Documento | Chunks | SQL/FTS/Chroma | Misma ruta | Otra ruta |
|---|---|---|---|---|
| Notas_Tecnicas_SDMX_2.1 | 115 | coherente | `NOOP_OWNER` | `NOOP_EQUIVALENT_OTHER_SOURCE` |
| SDMX_Glossary | 277 | coherente | `NOOP_OWNER` | `NOOP_EQUIVALENT_OTHER_SOURCE` |
| SDMX_2-1_User_Guide_6 | 197 | coherente | `NOOP_OWNER` | `NOOP_EQUIVALENT_OTHER_SOURCE` |
| SDMX-Training-introduction-2015 | 18 | coherente | `NOOP_OWNER` | `NOOP_EQUIVALENT_OTHER_SOURCE` |
| SDMX_Glossary_Test | 3 | coherente | `NOOP_OWNER` | `NOOP_EQUIVALENT_OTHER_SOURCE` |

**Punta a punta con LLM real** (`qwen3-14b` Q6_K, ctx 8192, LM Studio), corpus de
producción: **6/6 sin fallo**.

| Pregunta | Tiempo | Resultado |
|---|---|---|
| L1 lookup | 17.5 s | responde, 2/2 citas |
| L2 definición técnica | 21.9 s | responde |
| L3 distinción conceptual | 18.2 s | responde |
| L4 razonamiento multi-chunk | 16.8 s | responde |
| L5 límite del corpus | 23.9 s | responde reconociendo el límite |
| Fuera de corpus | 15.3 s | **abstiene**: «No encuentro esta información en los documentos proporcionados» |

La última es la pregunta que en la auditoría original producía `HTTP 400` por
prompt de 9794 tokens contra ventana de 8192: hoy no desborda y además se abstiene
en lugar de inventar.

## 7. Revisión adversarial independiente

Revisor: `openai-codex/gpt-6.1-sol`, aislado con `bwrap` (`/` en solo lectura,
red desconectada), sin acceso de escritura al candidato.

| Ronda | Objeto | Dictamen | Hallazgos |
|---|---|---|---|
| 1 | `a523049a…` | **ROJO** | R2-F1 y R2-F2 **cerrados**; 1 ALTO preexistente (`CLOSURE-F1`) |
| 2 | `00dedc73…` | **VERDE** | ninguno crítico, alto ni medio abierto |

En la ronda 2 el revisor verificó 172/172 ficheros antes y después,
re-ejecutó sus propias sondas de R2 (24 passed), probó **16 publicaciones
legítimas** del legacy (hash corroborado, sin filas de batch, sin runs, solo fila
documental) y **12 controles de NO-OP** con snapshot completo, más 4 casos de
revalidación TOCTOU. Confirmó que la corrección no bloquea la ruta legítima ni
exige inventar evidencia inexistente.

Candidato final congelado y revisado:

```
Manifiesto: candidate-v2/MANIFEST.sha256 (172 ficheros)
SHA combinado: 00dedc73d9e68f91cbba7b437069eeab8ac1bd58e080a54d29a924b10f584cea
```

## 8. Riesgos y limitaciones

1. **`Notas_Tecnicas_SDMX_2.1` — RESUELTO (2026-10-30).** Estaba congelado para
   actualizaciones: su propietario registrado apuntaba a una ruta inexistente y el
   fichero producía 86 chunks frente a los 115 indexados. Se exportaron los 115
   trozos a `evidence/pre-delete-Notas_chunks.jsonl`, se borró con
   `rag-lab docs delete` (sin restos de versión ni tags) y se reingerió desde el
   fichero actual. Resultado: 86 chunks, generación 1, `content_hash` ligado a la
   versión y `documents.path` apuntando a la ruta real. Corpus: 581 chunks
   coherentes en SQLite/FTS5/Chroma, sparse 100 %.
2. **Aislamiento de stores en tests — RESUELTO (2026-10-30).** La migración v7 se
   había aplicado a producción como efecto colateral de un test que abre
   `DocStore()` por defecto. Nuevo fixture autouse en `tests/conftest.py` que
   redirige los *bindings de módulo* a un temporal de sesión (eximiendo por
   marcador los tests `integration`, que leen el corpus a propósito). Verificado:
   sha256 de `storage/docstore.sqlite` y `data/ingested.jsonl` idéntico antes y
   después de la suite completa, y la suite baja de 665 s a 305 s.

### Limitaciones que persisten

1. **Concurrencia multiproceso no cubierta.** El contrato asume escritor único;
   la revalidación TOCTOU protege dentro de un proceso.
2. **Embeddings legacy desconocidos** se tratan como no comparables.
3. **Filas de control `SKIPPED`** se escriben en `ingest_documents` para registrar
   un NO-OP; están fuera del estado documental.
4. **P1 sin abordar** (calidad, no pérdida de datos): `RAG-06`/`RAG-07` (caché no
   invalida por run real ni distingue modos), `RAG-13` (48/62 preguntas sin hits
   BM25 por AND implícito), `RAG-14` (33 chunks >1024 tokens truncados para el
   embedding), `RAG-15` (citas fuera de rango aceptadas).

## 9. Uso

```bash
conda activate rag-lab

rag-lab query "¿Qué es un DataFlow en SDMX?"     # consulta
rag-lab chat                                      # interactivo
rag-lab ingest                                    # SOURCES de config.py
rag-lab ingest --doc ruta/documento.md            # un documento
rag-lab doctor                                    # salud del sistema
rag-lab reconcile --check                         # coherencia de stores
```

Respuestas posibles de `ingest` sobre un documento existente:

| Salida | Significado |
|---|---|
| `SKIPPED` / NO-OP | ya está igual: no se escribe nada |
| `NO-OP (equivalent, other source)` | otra ruta con el mismo contenido: no se escribe nada, no cambia el dueño |
| `REPLACE` | eres el dueño y el contenido cambió: se versiona |
| `DENY_NOT_OWNER` | otra ruta con contenido distinto: rechazado |
| `DENY_INCOHERENT` | los almacenes se contradicen: rechazado |

## 10. Continuación

1. Decidir sobre `Notas_Tecnicas_SDMX_2.1` (adoptar ruta o reindexar).
2. Cerrar el aislamiento de stores en tests (`DocStore()` por defecto apunta a
   producción).
3. Abordar los P1 como proyecto propio, empezando por `RAG-06`/`RAG-07` (caché),
   que son los de mayor riesgo de respuestas obsoletas.

## Referencias

- Auditoría original: `~/audits/rag-lab/rag-lab-audit-20261018-xlxeun/INFORME_FINAL.md`
- P0: `~/audits/rag-lab/p0-20261018/INFORME_P0_FINAL.md`
- P0-B: `~/audits/rag-lab/p0b-20261018/INFORME_P0B_FINAL.md`
- P0-C: `~/audits/rag-lab/p0c-20261020/INFORME_P0C_FINAL.md`
- Plan de este cierre: `docs/P0C_CLOSURE_PLAN.md`
- Evidencia completa: `~/audits/rag-lab/closure-20261030/`
