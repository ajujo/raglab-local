# Archive

This directory contains historical documents from early RAG-Lab development (approximately v1.0 through v1.5). They are preserved for reference but **should not be used as a source of truth for the current system state**. The architecture, configuration, and operational procedures described in these files have changed substantially.

For up-to-date documentation, see:
- `docs/ARCHITECTURE.en.md` / `docs/ARCHITECTURE.es.md` — current pipeline architecture
- `docs/BENCHMARKS.en.md` / `docs/BENCHMARKS.es.md` — current benchmark results and procedures
- `docs/OPERATIONS.es.md` — operational procedures
- `docs/FRONTMATTER.es.md` — YAML frontmatter contract

---

## Files in this archive

| File | Description |
|---|---|
| `AGENTS.md` | AI agent instructions from early versions (v1.0-era) |
| `auditoria.md` | Technical architecture audit (2026-05-20); key decision: 2-stage sparse retrieval |
| `BENCHMARKS.md` | Early benchmark experiments (v1.1 diversity experiments) |
| `CONFIG_PANEL.md` | Configuration parameter documentation (v1.0-era) |
| `GUIDE.md` | Usage guide (v1.0-era, pre-`rag-lab` wrapper) |
| `MULTI_DOC.md` | Multi-document support design notes (v1.0-era) |
| `QWEN.md` | Project context document written for the Qwen LLM (v1.0-era) |
| `rag-sdmx-plan.md` | Initial implementation plan (2026-04-21) |
| `Test_detallado.md` | Detailed test documentation (v1.0, 115 tests) |
| `update1-1.md` | v1.1 update notes |
| `update1-2.md` | v1.2 update notes |
| `update1-5.md` | v1.5 update notes |

---

> **Note:** The current test suite has over 1 000 tests. The storage architecture changed significantly in v1.16 (sparse vectors moved from `sparse_index.json` into DocStore SQLite BLOBs). Any file in this archive that refers to `sparse_index.json` or a `SparseStore` describes a design that no longer exists.
