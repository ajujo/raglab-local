# P0-C — Plan de cierre (R2-F1 / R2-F2)

> Rama: `p0c-closure-20261030` · Inicio: 2026-10-30
> Base: candidato P0-B r3 + P0-C congelado (`d9a8e23b…`, 172 ficheros verificados)
> Estado de entrada: **ROJO** por dos bloqueantes abiertos (informe
> `~/audits/rag-lab/p0c-20261020/INFORME_P0C_FINAL.md`).

## 0. Punto de partida real

El repositorio estaba **dos iteraciones por detrás** del trabajo auditado:

| Estado | Dónde estaba | Tests | Dictamen |
|---|---|---|---|
| P0 r1 | working tree del repo | 1176 | ROJO (`R3-01` CRITICAL) |
| P0-B r3 | `audits/…/p0c-20261020/baseline/` | 1229 | ROJO (R3-F1..F4) |
| P0-C | `audits/…/p0c-20261020/candidate/` | 1264 | ROJO (R2-F1, R2-F2) |

Ya aplicado a la rama (commits `b367966` y `64ffd25`): P0-B r3 + P0-C completos,
verificados fichero a fichero contra el manifiesto congelado. El delta fue
exactamente 6 ficheros de código + 5 de tests, sin cambios ajenos.

## 1. Los dos bloqueantes

Ambos están en el núcleo puro `rag_lab/ingest/ownership.py::decide()`, que es
quien autoriza o deniega escribir sobre un documento.

### R2-F1 — Contradicción de hashes no detectada cuando faltan filas de batch

En `decide()`, rama `state.has_version`, las dos comprobaciones de hash exigen
que `committed_hashes` no esté vacío:

```python
if state.active_content_hash and state.committed_hashes and \
        state.active_content_hash not in state.committed_hashes:      # línea 487
if state.documents_content_hash and state.committed_hashes and \
        state.documents_content_hash not in state.committed_hashes:   # línea 495
```

`committed_hashes` sale de `ingest_documents` (filas de batch). Pero la ruta
`retry` (`_ingest_one`) crea versión y metadatos **sin crear filas de batch**,
así que el estado normal es `committed_hashes == []`. En ese caso **nunca se
comparan entre sí** `document_versions.content_hash` y `documents.content_hash`:
una contradicción pasa desapercibida y se publica una nueva generación.

Reproducido por el revisor (SQLite/FTS/Chroma reales temporales): ingestar A por
`retry` → 15 chunks, generación 1, cero filas `ingest_documents`; alterar solo
`documents.content_hash`; reingestar contenido distinto → **publica**, generación
1→2, cambios documentales.

**Corrección exigida:** comparación directa entre los dos hashes persistidos
cuando ambos existen, con independencia de `committed_hashes`; discrepancia ⇒
`DENY_INCOHERENT`. Las comprobaciones de corroboración contra las filas de batch
se conservan tal cual cuando sí hay filas.

**Invariante:** *dos hashes persistidos que describen la misma versión confirmada
no pueden ser distintos. Si lo son, el estado es contradictorio y no se escribe.*

### R2-F2 — CREATE antes de validar índices huérfanos

```python
if not state.has_any_chunks and not state.has_documents:
    return Decision(DecisionKind.CREATE, "new document")     # líneas 536-538
```

La rama CREATE retorna **antes** de mirar `fts_matches` / `chroma_matches`. El
adaptador ya detecta la incoherencia (las calcula comparando el conjunto de
chunks SQL contra FTS y contra Chroma), pero CREATE no las consulta.

Reproducido por el revisor en entorno temporal sin versión, sin chunks SQL y sin
fila `documents`, insertando una fila fantasma en `chunks_fts` (o un vector
huérfano en Chroma): `fts_matches=false`, y aun así batch `committed=1` / retry
`15`, generación 0→1, y el fantasma sobrevive en el índice.

**Corrección exigida:** antes de devolver CREATE, exigir que los índices estén
coherentes con SQL (ambos vacíos para ese `doc_id`); si hay entradas huérfanas o
la coherencia no es verificable ⇒ `DENY_INCOHERENT`.

**Invariante:** *"documento nuevo" significa que no existe nada de él en ningún
almacén, no solo en SQLite.*

## 2. Regresiones obligatorias

En `tests/test_ingest/`, versionadas en el repo (no solo sondas del revisor):

1. R2-F1 en **batch**, en **retry** y con **`--force`**: contradicción
   `documents.content_hash` vs `document_versions.content_hash` con
   `committed_hashes` vacío ⇒ DENY, sin cambios documentales, generación intacta.
2. R2-F2 en **batch**, **retry** y **`--force`**, dos variantes: fila fantasma en
   `chunks_fts` y vector huérfano en Chroma ⇒ DENY, sin cambios documentales.
3. Control positivo: documento realmente nuevo en stores vacíos ⇒ CREATE sigue
   funcionando (que el fail-closed no rompa el caso legítimo).

Cada test compara estado antes/después (chunks, `chunks_fts`,
`document_versions`, `documents`, `document_tags`, snapshots, `version_ops`, ids y
textos de Chroma, manifiesto).

## 3. Criterios de aceptación

| # | Criterio | Cómo se comprueba |
|---|---|---|
| 1 | R2-F1 cerrado en las 3 rutas | regresiones nuevas + sonda del revisor re-ejecutada |
| 2 | R2-F2 cerrado en las 3 rutas | idem |
| 3 | Suite completa verde | `pytest tests/` en env `rag-lab`, CPU |
| 4 | Sin regresiones funcionales | CREATE/REPLACE/NO-OP/legacy siguen decidiendo igual |
| 5 | Sondas históricas conservadas | 116 pass / 11 fail documentados (`CONTRACT-CHANGE.md`), sin tocarlas |
| 6 | Producción intacta durante el desarrollo | `storage/` y `data/` sin cambios hasta la fase de migración |
| 7 | E2E con LLM real | `rag-lab query` contra `qwen3-14b` en `:8000`, sin HTTP 400 |
| 8 | Revisión independiente | `openai-codex/gpt-6.1-sol`, una ronda, VERDE |

## 4. Fases

| Fase | Qué | Estado |
|---|---|---|
| 0 | Commit de seguridad + tag + backup verificado de stores | ✅ hecho |
| 1 | Traer P0-B r3 + P0-C al repo y verificar contra manifiesto | ✅ hecho |
| 2 | Corregir R2-F1 y R2-F2 + regresiones | en curso |
| 3 | Suite completa, sondas históricas, E2E con LLM real | pendiente |
| 4 | Revisión independiente GPT-6.1 SOL | pendiente |
| 5 | Migración v7 sobre datos reales (con backup) + reconcile + doctor | pendiente |
| 6 | Informe final + actualización de README/ESTADO | pendiente |

## 5. Restricciones (no negociables)

- **No** tablas nuevas, **no** migración v8, **no** columnas sin autorización.
- **No** se relaja ninguna garantía de P0/P0-B ya aprobada.
- **No** se toca `storage/` ni `data/` hasta la fase 5, y solo con backup previo.
- **No** se borran ni alteran sondas históricas para "hacerlas pasar".
- **No** commit, push ni despliegue sin cerrar la revisión independiente.
