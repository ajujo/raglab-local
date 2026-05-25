# RAG-Lab — Development History

This document explains how and why the system evolved from its first version to the current
state at v1.19.1. It covers design decisions, benchmark results, and lessons learned at each
sprint.

---

## MVP — v1.0: Working basic RAG

The MVP established the complete end-to-end pipeline over a single Markdown document.

**Components implemented:**
- Ingestion with basic cleanup (strip of base64 images)
- Heading-aware chunking (H2+ boundaries respected, tables kept intact)
- Dense embedding with BGE-M3 (dense + sparse vectors simultaneously)
- Storage in ChromaDB (vectors) and JSON (sparse index)
- Retrieval with RRF (Reciprocal Rank Fusion) over dense + sparse
- Generation with a local OpenAI-compatible client

**What was missing in v1.0:**
- Automated tests
- Ingest transactions (a mid-process failure left the system inconsistent)
- Input document quality validation
- Multi-document support
- Evaluation metrics

---

## v1.1: Document diversity and tests

The most important limitation of v1.0 for a real corpus was that large documents monopolized
all top-5 retrieval slots. With a single long document, all retrieved chunks came from the
same file.

**MMR (Maximal Marginal Relevance):** Added diversity to chunk selection so that the top-K
represented different sections and documents. The diversity A/B benchmark showed hybrid_mmr
+14.6pp R@5 compared to retrieval without diversity.

**Automated test suite:** First test suite with coverage of the main modules. From v1.1 onwards,
all changes are validated before merging.

**Multi-doc:** Support for ingesting and querying multiple documents simultaneously with
filtering by `ACTIVE_DOCS`.

---

## v1.2: Verification layer

The system generated answers but had no mechanism to evaluate their reliability.

**Citation check:** Regex verification of the `[[N] ...]` format in the generated answer.
Checks that the cited chunks exist in the context sent to the LLM.

**Self-consistency:** A second LLM call to detect hallucinations. Costly in latency but
detectable: if the LLM generates assertions that contradict the context, the score drops.

**Trust score:** Weighted score combining citation (35%), retrieval (30%), consistency (25%),
and coverage (10%). Min-max normalization over retrieval scores before display.

The goal was not to block answers but to give the user a confidence signal.

---

## v1.3: Metadata and tags

With multiple documents and a growing corpus, filtering by collection or document type was
needed without modifying the ranking algorithm.

**`documents` table:** Source of truth for ingested documents with classification fields.

**Tag system:** Persistent per-document tags, filterable in retrieval. The `FilterSpec`
mechanism restricts the candidate pool before vector search.

**Sources:** Association of documents to data sources for lifecycle management.

---

## v1.4: Ingest transactions

Before v1.4, a failure during ingestion (embedding error, disk full, interruption) left the
system in an inconsistent state: ChromaDB might have chunks with no corresponding entry in
DocStore, or vice versa.

**IngestTransaction:** A class that groups all writes of one ingestion and does a complete
rollback if any step fails. Ingestion becomes atomic: either everything is written, or nothing
is written.

This also made `rag-lab ingest rollback <run_id>` possible — cleanly undoing a complete
ingestion.

---

## v1.5: Dataset/CSV removal

During early development, references to tabular datasets (CSV, Parquet, DuckDB) had crept into
the code. RAG-Lab is a RAG system over Markdown documents — it has no semantics for tabular
data.

**Scope guard:** All tabular loaders, dataset references, and associated tests were removed.
Stores were isolated in tests to prevent one suite from contaminating another.

This decision was deliberate and is not reversed. Tabular data is not a use case for RAG-Lab.

---

## v1.6: Markdown validation and quality gate

An emerging problem as the corpus grew: documents with defective structure (empty headings,
malformed frontmatter, broken tables) generated low-quality chunks that degraded retrieval
without any warning signal.

**`rag-lab docs validate`:** Validation command with error and warning codes before ingesting.
Detects: invalid frontmatter, empty headings, tables without headers, excessively long sections.

**`--strict`:** Flag that treats WARNs as ERRORs. Useful for CI pipelines or bulk ingestion
where quality assurance is required.

**`rag-lab docs inspect`:** Shows document structure (frontmatter, estimated tokens, estimated
chunks) without ingesting.

**`rag-lab docs preview-chunks`:** Generates the chunks that would be created without writing
anything to the stores. Allows auditing the chunking result before committing.

---

## v1.7: Cleanup

Technical debt sprint. No new features, only fixes.

**Remove `generation/verifier.py`:** An orphaned duplicate verification file had been left after
the v1.2 reorganization. It was removed to avoid confusion.

**Fix reranker device cache:** The global reranker cache did not respect device changes
(CPU/GPU) between tests. Tests must always use CPU regardless of the environment variable
value. The fix ensures that `conftest.py` can reset the device cache.

---

## v1.8: Benchmark framework

Without reproducible metrics it was impossible to know whether a change improved or worsened
retrieval. The "benchmark" before v1.8 was ad-hoc: a few questions tested by hand.

**Eval queries:** Official set of questions with relevance judgments over the SDMX corpus.
Each query has relevant chunks identified by `doc_id` and position.

**Baseline JSON:** The result of each benchmarked variant is persisted in JSON for historical
comparison.

**Compare guard:** A test that fails if a proposed variant degrades the main metrics (R@5, MRR,
nDCG@10) beyond the configured threshold.

From v1.8 onwards, any change in retrieval is validated against the benchmark before merging.

---

## v1.9: Real tokenizer

Chunking up to v1.8 used a character-length token estimate (`len(text)/4`). This approximation
is inaccurate for technical text with many symbols, tables, and short English terms.

**BAAI/bge-m3 tokenizer:** The real tokenizer of the embedding model was integrated for token
counting. The `CHUNK_MAX_TOKENS=800` limit is now applied accurately against the real tokens
the model will see.

Impact: some chunks that previously passed the limit are now split; some that were split
unnecessarily now remain together. This improves the semantic coherence of chunks.

---

## v1.10: Heading context in the reranker

The reranker (BGE-reranker-v2-m3) receives (query, chunk) pairs to score relevance. Before
v1.10, the chunk text was sent as-is, without context about the document or section it
belonged to.

**Prefix added to the chunk:** `"Document: {title}\nSection: {heading_path}\n\n{text}"` before
sending to the reranker.

**Benchmark result:**
- R@5: +2.1pp
- MRR: +2.6pp

**Known regression:** Query q070 (Spanish/English mix) showed -0.5pp in MRR. The effect is
minor and the global benefit is clear.

This was the change with the highest return on investment in the entire project history.

---

## v1.11: Query variants cleanup and CI baseline

In v1.10, query expansion mechanisms (synonym variants, LLM-generated paraphrases) had been
left active. The A/B benchmark showed:

- Query variants: **0 benefit** in R@5, MRR, nDCG@10
- Latency: **×2** (each variant requires an additional embedding pass)

**Decision:** Disable variants by default. The code is kept but not executed unless an explicit
flag is set.

**v1.11 promoted as CI baseline:** From v1.11 onwards, the benchmark always compares against
the v1.11 state as the reference line. This makes historical comparison reproducible.

---

## v1.12: HyDE + query rewriting

HyDE (Hypothetical Document Embeddings) is a technique that generates a hypothetical text
fragment with the LLM ("what would the answer to this question look like?") and uses that
fragment as the embedding query instead of the original text.

**Implementation:** v1.12 implemented HyDE and query rewriting as opt-in options (`--hyde`,
`--rewrite`).

**HyDE benchmark on SDMX corpus:**
- R@5: -3.8pp compared to baseline
- Latency: ×12.5
- Verdict: **net negative**

**Explanation:** HyDE helps when queries are short or ambiguous and the corpus is heterogeneous.
The SDMX corpus is technical and specific. Queries are already sufficiently informative for
dense embedding without expansion.

**Query rewriting:** Implemented but without an official benchmark. Status: available as
`--rewrite` but not recommended in production.

---

## v1.13: HNSW configuration audit

ChromaDB uses HNSW for the vector index. HNSW parameters (M, ef_construction, ef_search)
control the trade-off between index quality and search speed.

**Finding:** In ChromaDB 1.x all HNSW parameters are **build-time** — they are set when the
collection is created and cannot be modified without rebuilding the index from scratch.

**Profile benchmark:**
- M=8: recall -12pp compared to M=16
- M=16 (default): baseline
- M=32: +0.3pp recall, +40% construction time

**Decision:** Keep M=16 (ChromaDB default). There is no benefit in increasing it, and M=8
degrades significantly. The parameter is documented but not exposed as configurable.

---

## v1.14: Query cache

Repeated queries (same text, same corpus) regenerated the embedding, ran the search, and called
the LLM every time. In interactive use this is unnecessary latency.

**SQLite cache:** Persistent cache in SQLite keyed by the fingerprint of query + configuration.
7-day TTL by default.

**Corpus fingerprint:** The fingerprint includes a hash of the corpus state (number of chunks,
IDs). If a document is ingested or deleted, the fingerprint changes and all cache entries are
automatically invalidated.

**v1.14.1:** Invalidation fix: tag operations and `docs delete` also invalidate the cache
(the corpus was being updated but the fingerprint was not being recalculated).

---

## v1.15: Feedback store

Without real usage data there was no way to know which queries failed, which chunks were being
retrieved unnecessarily, or which documents were not useful.

**FeedbackEntry:** Per-query record of: query text, hyde flag, retrieved chunk_ids, retrieval
score, feedback type (`useful`, `not_useful`, `relevant`, `irrelevant`, `wrong_doc`, `outdated`,
`duplicate`, `bad_citation`).

**Purely observational:** Does not affect ranking now or in the near future. The risk of
overfitting to few events is real. The threshold for activating it as a ranking signal is >50
events with a clear pattern.

**CLI:** `rag-lab feedback add/list/stats/export/clear`.

---

## v1.16: Batch/resumable ingest

With larger corpora or many documents being ingested at once, sequential ingestion could take
minutes. A failure would interrupt the whole process and require a restart.

**Parallel workers:** `--workers N` allows ingesting N documents in parallel.

**IngestTransaction per batch:** Each document has its own transaction. A failure in one
document does not affect the rest of the batch.

**Batch and run tracking:** `rag-lab ingest batches` and `rag-lab ingest runs` show the ingest
history with status (success, failed, partial). `rag-lab ingest retry <run_id>` retries only
the failed documents from a previous run.

**`--resume`:** Continues an interrupted ingestion from the last checkpoint.
**`--retry-failed`:** Automatically retries documents with `failed` status from the most
recent run.

---

## v1.17: Release candidate audit

Quality sprint before v1.18. No new features.

**Store isolation guard tests:** Tests that verify no test suite leaves state in shared stores.
They detect cases where a test creates chunks in ChromaDB or DocStore and does not clean them
up correctly.

**`rag-lab doctor`:** Health check command that runs a series of checks on the system state:
configuration, DocStore, ChromaDB, FTS5, sparse coverage, reconciliation, ingest health, and
a test query.

**`rag-lab reconcile`:** Verifies and repairs inconsistencies between stores (DocStore,
ChromaDB, FTS5, Sparse BLOBs). `--check` only reports; `--repair` fixes.

**`rag-lab diagnose`:** Runs a diagnostic query with a detailed pipeline trace: which chunks
were retrieved, what the reranker scored, which filters were applied.

---

## v1.18: Verification hardening

The v1.2 verification layer had four silent bugs that had gone unnoticed because tests did not
cover edge cases in citation parsing.

**4 bugs fixed:**
1. The citation regex did not recognize the `[[N]]` format (double brackets without a space).
2. The coverage score was always 1.0 if there were no citations (should be 0.0).
3. The `evidence_map` was not built if the answer was empty.
4. The consistency comparison failed silently if the second LLM call returned malformed JSON.

**evidence_map:** Explicit evidence map that associates each assertion in the answer with the
chunk that supports it. Visible in the verbose trace.

**Verbose trace:** `--profile` now shows the full verification trace: which chunks were cited,
what consistency scored, and why the trust score has that value.

---

## v1.18.1: E2E audit

After the v1.18 hardening, a full E2E audit was run with a real LLM (not a mock).

**10/10 PASS:** The 10 questions in the E2E evaluation set produced answers with correct
citation check, consistency check without detected hallucinations, and trust score > 0.7.

**Audit script:** `scripts/e2e_audit.py` — runs the full set and produces a results report.
Reproducible with `python scripts/e2e_audit.py`.

---

## v1.18.2: Legacy SparseStore JSON removal

The original sparse index (v1.0) used a JSON file to store sparse vectors. In v1.16 it was
migrated to SQLite BLOBs as the canonical format. In v1.18.2 the legacy read/write code for
the JSON sparse format was removed.

**SparseStore JSON:** Dead code removed. The canonical format is SQLite BLOBs in
`storage/docstore.sqlite`.

The migration `python -m rag_lab.maintenance.migrate_to_v2` remains available for installations
that still have the old JSON file.

---

## v1.19: Frontmatter contract

With the corpus growing in number of documents and domains, manual classification by filename
was insufficient. A structured mechanism was needed to associate classification metadata with
each document.

**YAML frontmatter contract:** The fields `doc_id`, `title`, `domain`, `source_type`,
`language`, `version`, and `tags` are read from YAML frontmatter and persisted in the
`documents` table.

**Derived tags:** Each classification field automatically generates a derived tag
(`domain:sdmx`, `source_type:manual`, `lang:en`, `version:2.1`). This allows filtering by
classification using the existing tag infrastructure without special code paths.

**FilterSpec:** `FilterSpec(domain="sdmx")` resolves internally to
`tags_include=["domain:sdmx"]` before retrieval. The ranking algorithm does not change.

**Prohibited fields:** `dataset` and `dataset_id` produce ERROR `frontmatter_scope_violation`.
RAG-Lab does not support tabular data.

---

## v1.19.1: Documentation

Documentation sprint. Complete reorganization of `/docs/`:

- `ROADMAP.es.md` / `ROADMAP.en.md` — Roadmap with explicit philosophy
- `DEVELOPMENT_HISTORY.es.md` / `DEVELOPMENT_HISTORY.en.md` — This document
- `API_REFERENCE.es.md` / `API_REFERENCE.en.md` — Complete CLI reference
- `FRONTMATTER.en.md` — English version of the existing frontmatter contract

---

## Discarded decisions and why

### HyDE disabled by default

Benchmarked in v1.12: R@5 -3.8pp, latency ×12.5. The SDMX corpus is technical and specific —
queries are already sufficiently informative for dense embedding. HyDE remains available as
`--hyde` for experimentation but is not activated by default.

### PDF deferred indefinitely

Text extraction from PDF loses document structure. Headings disappear or become plain text,
tables deform, text order can be scrambled. RAG-Lab's chunker depends on Markdown structure.
Without an audited conversion pipeline that produces quality Markdown, PDFs would generate
low-quality chunks.

### Feedback frozen as a ranking signal

With <50 events there is no statistical signal. Adjusting ranking weights from sparse data
produces overfitting. Feedback exists as an observation tool and will be activated as a ranking
signal when there is sufficient evidence.

### CSV/tabular data removed (v1.5)

RAG-Lab is a RAG system over Markdown documents. It has no semantics for tabular data.
CSV/Parquet/DuckDB loaders that had crept into early code were removed as a scope guard.

### Global sparse search discarded

A global sparse scan without WAND/early termination is O(N) over the full corpus. SQLite does
not have these optimization mechanisms for inverted indices. At 610 chunks it is tolerable, but
it does not scale. The correct solution requires a dedicated engine (Elasticsearch, Qdrant).
The current architecture (sparse only over the dense candidate pool) is the right compromise
for the current corpus size.

### Query variants disabled (v1.11)

A/B benchmark: 0 benefit in R@5/MRR/nDCG@10, latency ×2. The cost does not justify the zero
benefit. Variants are implemented but disabled by default.

---

*Last updated: v1.19.1*
