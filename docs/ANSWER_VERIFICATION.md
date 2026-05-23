# Answer Verification Layer

This document describes the answer verification pipeline introduced in v1.18 and
validated end-to-end in v1.18.1.

---

## Overview

Every query response passes through three verification components before being shown
to the user:

```
LLM response
     │
     ├─► [1] Citation verifier    — checks every [[N] ...] citation against retrieved chunks
     │
     ├─► [2] Consistency check    — second LLM call: detects unsupported claims / hallucinations
     │
     └─► [3] Scoring              — weighted confidence score (0–1)
```

The result is a `VerificationResult` object that drives both the displayed block and
programmatic access via `evidence_map`.

---

## Citation format

The system prompt instructs the LLM to cite every factual claim using:

```
[[N] Fuente: <doc_id> | Sección: <heading_path> | Líneas: <line_start>-<line_end>]
```

where `N` is the fragment number as shown in the context.  Example:

> SDMX stands for Statistical Data and Metadata eXchange.
> [[1] Fuente: SDMX-Training-introduction-2015 | Sección: What is SDMX | Líneas: 18-21]

The verifier regex matches this exact format.  Citations are classified as:

| Status | Meaning |
|--------|---------|
| **VALID** | doc_id matches, heading fuzzy-matches, both line boundaries match |
| **PARTIAL** | doc_id + heading match, only one line boundary matches |
| **INVALID** | No retrieved chunk matches the citation |

---

## Confidence score

The final score is a weighted sum of four sub-scores:

| Component | Weight | Source |
|-----------|--------|--------|
| Citation score | 35% | fraction of VALID citations |
| Retrieval score | 30% | top-3 min-max-normalized reranker logits |
| Consistency score | 25% | second LLM call result |
| Coverage score | 10% | fraction of citations that matched any chunk |

Thresholds:

| Level | Score range |
|-------|-------------|
| HIGH | ≥ 0.75 |
| MEDIUM | 0.50–0.74 |
| LOW | < 0.50 |

---

## Verification block (standard output)

```
─────────────────────────────────────────────
Verificación de respuesta
  Fragmentos recuperados:
    [1] SDMX_Guide | Líneas 18-21   8.3/10 ████████░░
    [2] SDMX_Glossary | Líneas 6738-6767   6.1/10 ██████░░░░
  Citas verificadas : 2/2 ✓
  Consistencia      : OK ✓
  Score de confianza: 0.87 — HIGH ✓
─────────────────────────────────────────────
```

### v1.18 display fixes

| Situation | Before v1.18 | v1.18+ |
|-----------|-------------|--------|
| Response with zero citations | `0/0 ✓` (false positive) | `0/0 ✗` |
| Consistency parse failed | `N/A` | `DEGRADED ⚠` |
| Consistency check disabled | score=1.0 (boost) | score=0.75 (neutral) |

---

## Verbose mode (`format_verification_block(verbose=True)`)

Pass `verbose=True` to include a traceability section showing `chunk_id` and a text
snippet for each citation:

```
  Trazabilidad:
    [1] SDMX_Guide | Líneas 18-21  →  chunk_id: a71339880b1c…
        "This document provides some background information on the SDMX…"
```

This is available programmatically; the CLI currently uses `verbose=False` (default).

---

## `evidence_map` property

`VerificationResult.evidence_map` is a computed property that maps citation index
(1-based) to a dict:

```python
{
  1: {
    "chunk_id": "a71339880b1c…",
    "doc_id": "SDMX-Training-introduction-2015",
    "lines": (18, 21),
    "status": "VALID",
  },
  …
}
```

Returns `{}` when no citations were produced.

---

## Warnings

`VerificationResult.get_warnings()` returns a list of human-readable strings:

| Trigger | Warning |
|---------|---------|
| Zero citations | `⚠ Respuesta sin citas — no se puede trazar al documento fuente.` |
| INVALID citation | `Cita inválida: [[N] Fuente: …]` |
| Hallucination detected | `⚠ Se detectaron posibles alucinaciones en la respuesta.` |
| Unsupported / contradiction | `⚠ Algunas afirmaciones pueden no estar respaldadas…` |
| Consistency parse failure | `⚠ Consistency check no pudo ejecutarse correctamente.` |
| Low retrieval spread | `⚠ Algunos fragmentos tienen relevancia baja…` |

---

## E2E audit script

The script `scripts/audit_answer_verifier.py` runs the full pipeline end-to-end
against a pre-defined query suite and saves a JSON report.

```bash
# Full live run (LLM required)
python scripts/audit_answer_verifier.py --suite answer_e2e

# Dry-run (no LLM — uses synthetic response, useful in CI)
python scripts/audit_answer_verifier.py --dry-run

# Custom output path
python scripts/audit_answer_verifier.py --output path/to/report.json
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All entries PASS or WARN |
| 1 | At least one entry FAIL |
| 2 | Unrecoverable runtime error |

### Verdict criteria

| Category | PASS | WARN | FAIL |
|----------|------|------|------|
| `easy_direct`, `technical_sdmx`, `spanish` | All citations VALID, HIGH/MEDIUM confidence | Some PARTIAL, or LOW confidence | Zero citations, or all INVALID, or hallucination |
| `ambiguous` | ≥1 VALID citation | Zero VALID citations | Hallucination |
| `out_of_corpus` | System declares info not in corpus | Produced citations but didn't declare absence | — |

### v1.18 invariant checks (printed at end of each run)

The script verifies four invariants introduced in v1.18:

- **BUG-1**: No `0/0 ✓` false positives in any verification block.
- **BUG-2**: Zero-citation responses emit a warning.
- **BUG-3**: No `N/A` consistency status (replaced by `DEGRADED ⚠`).
- **FEAT**: `evidence_map` is populated for all entries with VALID citations.

---

## v1.18.1 E2E audit results

**Date:** 2026-05-23  
**Suite:** `answer_e2e` (10 queries)  
**Mode:** LIVE (real LLM)  
**Result: 10/10 PASS**

| ID | Category | Verdict | Citations | Confidence |
|----|----------|---------|-----------|-----------|
| e01 | easy_direct | PASS | 3/3 VALID | HIGH |
| e02 | easy_direct | PASS | 9/9 VALID | HIGH |
| e03 | easy_direct | PASS | 3/3 VALID | HIGH |
| t01 | technical_sdmx | PASS | 10/10 VALID | HIGH |
| t02 | technical_sdmx | PASS | 13/13 VALID | HIGH |
| t03 | technical_sdmx | PASS | 7/7 VALID | HIGH |
| s01 | spanish | PASS | 10/10 VALID | HIGH |
| s02 | spanish | PASS | 10/10 VALID | HIGH |
| a01 | ambiguous | PASS | 10/10 VALID | HIGH |
| o01 | out_of_corpus | PASS | system correctly declared info not in corpus + zero-citation ⚠ emitted |

All four v1.18 invariant checks passed.  
Full JSON report: `data/audits/v1.18.1_answer_verifier_e2e.json`

---

## Configuration

Verification behaviour is controlled by `rag_lab/config.py`:

| Config | Default | Effect |
|--------|---------|--------|
| `ENABLE_CONSISTENCY_CHECK` | `True` | Enables the second LLM call for faithfulness |
| `RERANK_TOP_K` | `8` | Number of chunks passed to LLM and verifier |

Disabling `ENABLE_CONSISTENCY_CHECK` sets consistency score to `0.75` (neutral,
not a boost).
