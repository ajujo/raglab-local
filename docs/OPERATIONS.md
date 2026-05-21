# RAG-Lab Operations Guide

Operational reference for diagnosing, maintaining, and protecting the RAG-Lab system.

---

## Quick-reference commands

| Goal | Command |
|------|---------|
| Full health check | `python -m rag_lab.doctor` |
| Ingest a document (transactional) | `rag-lab ingest --doc path/to/doc.md` |
| Ingest all sources | `rag-lab ingest` |
| Resume crashed ingest runs | `rag-lab ingest --resume` |
| Retry all failed ingest runs | `rag-lab ingest --retry-failed` |
| List ingest run history | `rag-lab ingest runs` |
| Show a specific ingest run | `rag-lab ingest show <run_id>` |
| Roll back a failed run manually | `rag-lab ingest rollback <run_id>` |
| Retry a specific failed run | `rag-lab ingest retry <run_id>` |
| Health check with test query | `python -m rag_lab.doctor --query "What is SDMX?"` |
| Run specific checks only | `python -m rag_lab.doctor --checks config,docstore,chromadb` |
| Store consistency report | `python -m rag_lab.maintenance.reconcile` |
| Store consistency (CI mode) | `python -m rag_lab.maintenance.reconcile --check` |
| Remove ChromaDB orphans | `python -m rag_lab.maintenance.reconcile --repair` |
| Save reconcile report | `python -m rag_lab.maintenance.reconcile --report-json out.json` |
| Full system diagnostic | `python -m rag_lab.maintenance.diagnose` |
| Diagnostic with test query | `python -m rag_lab.maintenance.diagnose --query "What is SDMX?"` |
| Diagnostic with signal breakdown | `python -m rag_lab.maintenance.diagnose --query "..." --explain` |
| Diagnostic with filter | `python -m rag_lab.maintenance.diagnose --query "..." --tag glossary --explain` |
| Benchmark regression check | `python -m rag_lab.benchmark.compare --baseline data/benchmark_v1_1_mmr_20260521.json --current data/benchmark_latest.json` |
| Run benchmark | `python -m rag_lab.benchmark --variants hybrid hybrid_mmr --output data/benchmark_latest.json` |
| Ingest all documents | `python -m rag_lab.cli ingest` |
| Backfill sparse BLOBs | `python -m rag_lab.maintenance.backfill_sparse` |
| Migrate to schema v2 | `python -m rag_lab.maintenance.migrate_to_v2` |
| Migrate to schema v3 (metadata) | `python -m rag_lab.maintenance.migrate_to_v3` |
| List documents | `python -m rag_lab.cli docs list` |
| List documents by tag | `python -m rag_lab.cli docs list --tag glossary` |
| Show document details | `python -m rag_lab.cli docs show SDMX_Glossary` |
| Tag a document | `python -m rag_lab.cli docs tag SDMX_Glossary glossary` |
| Untag a document | `python -m rag_lab.cli docs untag SDMX_Glossary glossary` |
| Delete document (all stores) | `python -m rag_lab.cli docs delete SDMX_Glossary` |
| List all tags | `python -m rag_lab.cli tags list` |
| Rename a tag | `python -m rag_lab.cli tags rename old-name new-name` |

---

## Doctor command

`python -m rag_lab.doctor` runs 7 sequential health checks and exits with a clear status.

### Checks

| Check | What it verifies |
|-------|-----------------|
| `config` | Required config constants exist and have valid values |
| `docstore` | SQLite DocStore opens and contains chunks |
| `chromadb` | ChromaDB collection is reachable and non-empty |
| `fts5` | FTS5 virtual table exists and is fully populated |
| `sparse_coverage` | Fraction of chunks with sparse BLOBs meets `SPARSE_COVERAGE_THRESHOLD` |
| `reconcile` | Cross-store consistency (DocStore vs ChromaDB); calls reconcile internally |
| `test_query` | End-to-end retrieval returns at least one result for the test query |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All checks OK |
| `1` | At least one WARN, no FAIL |
| `2` | At least one FAIL |

### Options

```
--checks NAME[,NAME,...]   Run only the specified checks (comma-separated)
--query TEXT               Query used for test_query check (default: "What is SDMX?")
```

### Example output

```
───────────────────────────────────────────────────
RAG-Lab Doctor
───────────────────────────────────────────────────
  ✓ config               OK  — dim=1024, model_ver=2024-09, top_k=50
  ✓ docstore             OK  — 610 chunks
  ✓ chromadb             OK  — 610 vectors
  ✓ fts5                 OK  — 610/610 chunks indexed
  ✓ sparse_coverage      OK  — 610/610 (100%)
  ✓ reconcile            OK  — DocStore=610, ChromaDB=610
  ✓ test_query           OK  — 3 results — top: SDMX_Glossary rrf=0.0412

───────────────────────────────────────────────────
  ✓ Overall: OK
───────────────────────────────────────────────────
```

---

## Reconcile command

`python -m rag_lab.maintenance.reconcile` checks consistency between DocStore and ChromaDB.

### Modes

| Flag | Behaviour |
|------|-----------|
| _(none)_ | Print report, exit 0 if consistent, exit 1 if issues |
| `--check` | CI mode — same as default (explicit) |
| `--repair` | Remove orphaned entries from ChromaDB (destructive) |
| `--fix` | Alias for `--repair` (backward compatibility) |
| `--report-json PATH` | Save full JSON report to PATH |

### Extended checks (v1.2+)

Beyond orphan detection, reconcile now reports:

- **Duplicate chunk IDs** — should never occur (PK constraint), but reported if found
- **Model version mismatches** — chunks ingested with an older `embedding_model_version`
- **Embedding dim mismatches** — chunks with a different `embedding_dim` than current config
- **Sparse format version mismatches** — chunks with a stale `sparse_format_version`

Any of these conditions causes exit code 1 and is flagged in the report.

### JSON report format

```json
{
  "docstore_count": 610,
  "chroma_count": 610,
  "fts_count": 610,
  "sparse_blob_count": 610,
  "chroma_orphans": [],
  "missing_from_chroma": [],
  "duplicate_chunk_ids": [],
  "model_version_mismatches": [],
  "embedding_dim_mismatches": [],
  "sparse_format_version_mismatches": [],
  "repaired": false
}
```

### Recovery actions

| Issue | Command |
|-------|---------|
| ChromaDB orphans | `python -m rag_lab.maintenance.reconcile --repair` |
| Missing from ChromaDB | `python -m rag_lab.cli ingest --force` |
| FTS5 incomplete | `python -m rag_lab.maintenance.migrate_to_v2` |
| Sparse BLOBs missing | `python -m rag_lab.maintenance.backfill_sparse` |
| Model version mismatch | Re-ingest affected documents |

---

## Diagnose command

`python -m rag_lab.maintenance.diagnose` gives a detailed view of store counts and coverage.

### Options

```
--query TEXT     Run a test retrieval query
--explain        Show per-signal rank breakdown for each result (requires --query)
```

### Explain mode

With `--explain`, each result shows:

```
  [1] SDMX_Glossary | lines 142-160
       rrf=0.0412  dense=0.0231  bm25=12.50  sparse=0.0871
       in_dense=True  in_bm25=True  in_sparse=True
       ┌─ signal ranks: dense[rank=2]  bm25[rank=1]  sparse[rank=3]
       │  rrf_rank=1  chunk_id=abc123de…
       └─ mmr_score=0.7231 ← MMR reordered
```

This is the primary tool for diagnosing why a specific chunk appears or doesn't appear in results.

---

## Benchmark regression guard

`python -m rag_lab.benchmark.compare` compares a current benchmark run against a saved baseline.

### Usage

```bash
python -m rag_lab.benchmark.compare \
    --baseline data/benchmark_v1_1_mmr_20260521.json \
    --current  data/benchmark_latest.json \
    --variant  hybrid_mmr \
    --output   data/regression_report.json
```

### Default thresholds

| Metric | Threshold | Severity |
|--------|-----------|----------|
| R@5 | drop > 2 pp | FAIL |
| nDCG@10 | drop > 2 pp | FAIL |
| MRR | drop > 3 pp | FAIL |
| P95 latency | increase > 25% (relative) | WARN |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | No regressions |
| `1` | At least one WARN |
| `2` | At least one FAIL |

### Canonical baseline

The v1.1 baseline JSON is stored at `data/benchmark_v1_1_mmr_20260521.json`.

Key metrics (28 queries, `hybrid_mmr` variant, λ=0.6):

| Metric | Value |
|--------|-------|
| R@5 | 1.000 |
| MRR | 0.884 |
| nDCG@10 | 0.840 |
| unique_docs@5 | 4.82 |

---

## Ingest transactions (v1.4+)

Every document ingest is now wrapped in a logical transaction that tracks
progress and performs rollback compensation on failure.

### Status transitions

```
IN_PROGRESS → COMMITTED     (success)
IN_PROGRESS → FAILED        (exception raised mid-ingest)
FAILED      → ROLLED_BACK   (after compensation: delete from ChromaDB + DocStore + FTS5 + documents)
```

### Checking ingest history

```bash
rag-lab ingest runs                      # last 20 runs
rag-lab ingest runs --status FAILED      # only failed
rag-lab ingest runs --doc SDMX_Glossary  # runs for one doc
rag-lab ingest show abc123def456         # full details
```

### Recovering from failures

After a crash or partial ingest:

```bash
# Automatic: roll back stale IN_PROGRESS + FAILED, then re-ingest
rag-lab ingest --resume          # stale IN_PROGRESS only
rag-lab ingest --retry-failed    # FAILED only

# Manual: roll back a specific run, then re-ingest separately
rag-lab ingest rollback abc123def456
rag-lab ingest --doc path/to/doc.md --force
```

`rollback` is **idempotent** — safe to run multiple times.

### What rollback does

1. Delete from ChromaDB: `VectorStore.delete_by_doc_id(doc_id)`
2. Delete from SQLite chunks + FTS5 + documents table
3. Mark run status as `ROLLED_BACK`

### Stale IN_PROGRESS detection

Runs that have been `IN_PROGRESS` for more than 30 minutes are considered
stale (process likely crashed). Both `reconcile` and `doctor ingest_health`
detect and report them.

---

## Routine maintenance checklist

Run after any bulk ingest or schema change:

1. `python -m rag_lab.doctor` — quick health gate
2. `python -m rag_lab.maintenance.reconcile` — cross-store consistency
3. `python -m rag_lab.benchmark.compare --baseline data/benchmark_v1_1_mmr_20260521.json --current <new_run>` — regression guard
4. `pytest tests/ -v` — full test suite

---

## Troubleshooting

### "Sparse scoring disabled: coverage X% < threshold 95%"

Sparse BLOBs are below the configured threshold. Run:

```bash
python -m rag_lab.maintenance.backfill_sparse
```

### "FTS5 table is empty — run migrate_to_v2"

The FTS5 virtual table was not created or is empty. Run:

```bash
python -m rag_lab.maintenance.migrate_to_v2
```

### ChromaDB orphans after integration tests

Some integration tests insert temporary chunks that may not be cleaned up. Run:

```bash
python -m rag_lab.maintenance.reconcile --repair
```

### Stale model version / embedding dim mismatches

Chunks were ingested with an older config. Re-ingest the affected documents:

```bash
python -m rag_lab.cli ingest --doc path/to/document.md --force
```

---

## Document metadata and tags (v1.3+)

### Schema

v1.3 adds five metadata tables to docstore.sqlite (same file as chunks):

| Table | Purpose |
|-------|---------|
| `documents` | One row per doc: path, content_hash, source_id, status, timestamps |
| `tags` | Normalized tag names (tag_id, name UNIQUE) |
| `document_tags` | Many-to-many join, ON DELETE CASCADE |
| `sources` | Optional source catalogue |

### Initial migration

After upgrading to v1.3, run once to populate the documents table from chunks
and migrate any tags from the legacy doc_manager.db:

```bash
python -m rag_lab.maintenance.migrate_to_v3
```

Safe to re-run — fully idempotent.

### Tagging documents

```bash
rag-lab docs tag SDMX_Glossary glossary
rag-lab docs tag SDMX_Glossary sdmx-core
rag-lab docs tag SDMX_2-1_User_Guide_6 user-guide
rag-lab tags rename sdmx-core sdmx
```

### Filtering retrieval by tags

Tags are resolved to doc_ids before any retrieval call. The ranking pipeline
(RRF, MMR, etc.) is unchanged — tags only restrict the candidate pool.

Via diagnose (for debugging):
```bash
python -m rag_lab.maintenance.diagnose --query "code list" --tag glossary
python -m rag_lab.maintenance.diagnose --query "REST API" --exclude-tag test --explain
```

Via Python (`hybrid_search`):
```python
from rag_lab.retrieval.filters import FilterSpec
results = hybrid_search(
    query, vs, ds, fts,
    query_dense=emb,
    query_sparse=sparse,
    filter_spec=FilterSpec(tags_include=["glossary"]),
)
```

### tag include logic (AND)

`tags_include=["glossary", "sdmx-core"]` returns only documents that have
**all** listed tags. Use multiple `--tag` flags in diagnose for AND logic.

### Deleting a document consistently

```bash
rag-lab docs delete SDMX_Glossary_Test
```

Removes from: SQLite chunks + FTS5 + documents table + ChromaDB vectors.

### Reconcile metadata checks

After v3 migration, reconcile also checks:
- Documents in documents table with no chunks (possible after manual deletion)
- Chunk doc_ids with no documents row (migration not run yet)
- document_tags pointing to deleted documents (should be zero due to CASCADE)

```bash
python -m rag_lab.maintenance.reconcile
```
