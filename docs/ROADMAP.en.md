# RAG-Lab — Roadmap (v1.19.1)

> **Philosophy:** No feature gets added out of momentum. Only with evidence, a benchmark result,
> or a real usage pain point. If there is no concrete problem to solve or metric to improve,
> the feature does not ship.

---

## Current state

**Version:** v1.19.1
**Status:** Stable, in active local use
**Corpus:** 610 chunks of SDMX documentation
**Tests:** 1031 tests, all passing
**Retrieval:** R@5=0.821, R@10=0.896, MRR=0.939, nDCG@10=0.837

The system is complete for its current use as a local CLI tool over a Markdown corpus of SDMX
technical documentation. There is no known active technical debt.

---

## Near-term reasonable work (based on actual usage)

These tasks have direct justification from how the system is used today.

### Add frontmatter to the existing SDMX corpus

Most documents in the corpus predate v1.19 and have no frontmatter. Adding the YAML contract
(`doc_id`, `domain`, `source_type`, `language`, `version`) is documentation work, not
engineering work — it has to be done manually, document by document.

- **Why now:** The system can already use metadata for retrieval filtering. Without frontmatter,
  `FilterSpec(domain="sdmx")` filters do not work on legacy documents.
- **Cost:** Manual effort. Can be done incrementally.
- **Risk:** Low. The workflow is `validate → inspect → ingest --force`.

### End-to-end response quality benchmark

The current benchmark measures retrieval quality (R@5, MRR, nDCG@10). It does not measure
whether the generated answer is correct, complete, or useful. Evaluating the full pipeline
requires a set of (question, expected answer) pairs and a text evaluation metric.

- **Why now:** Without this metric, retrieval improvements do not necessarily translate into
  better answers as perceived by the user.
- **Cost:** High. Requires creating the evaluation dataset and defining the metric (ROUGE,
  BERTScore, LLM-as-judge, or human evaluation).
- **Blocker:** There is no set of "correct answers" for the SDMX corpus. Creating that dataset
  is a prerequisite.

### Scale watch: sparse O(N) over the candidate pool

The current sparse scan is O(N) over the candidate pool (the top-K results from vector search).
With 610 chunks this is trivially fast. If the corpus grows by several orders of magnitude,
this design decision will need to be revisited.

- **Signal to act:** Retrieval latency > 2s in benchmark, or corpus > 50,000 chunks.
- **Future solution:** An engine with WAND / early termination (Elasticsearch, Qdrant). Not SQLite.
- **Today:** Do nothing. The problem does not exist.

---

## Deferred improvements (waiting for real data)

Features that are implemented or designed but deliberately not activated until there is evidence.

### Feedback as a re-ranking signal

The feedback store has existed since v1.15 and records events (useful, not useful, wrong, etc.).
It is currently **purely observational** — it does not affect ranking or chunk selection.

- **Why deferred:** With fewer than 50 events there is no statistical signal. Adjusting ranking
  weights from sparse data produces overfitting to noisy cases and degrades overall performance.
- **Condition to activate:** >50 events with a clear, reproducible pattern of avoidable failures.
  An A/B benchmark showing improvement is required before activating.
- **Risk of premature activation:** A few negative events on a legitimately relevant chunk can
  suppress that chunk in future queries.

### Query rewriting on by default

The query rewriter has been implemented since v1.12 as the `--rewrite` flag. No official
benchmark of its effect on retrieval has been run.

- **Why deferred:** Without a benchmark we do not know whether it helps or hurts. HyDE was
  implemented with the same expectations and turned out to be negative (-3.8pp R@5, latency ×12.5).
- **Condition to activate:** An official benchmark showing a net improvement in R@5 and MRR
  with acceptable latency.
- **Current status:** Available as `rag-lab query "..." --rewrite`. Not recommended in
  production until benchmarked.

---

## Frozen improvements (do not revisit without strong evidence)

Decisions made with data. Reopen only if the data changes.

### HyDE on by default

HyDE (Hypothetical Document Embeddings) generates a hypothetical passage with the LLM before
searching the corpus. It was implemented in v1.12 and benchmarked on the official SDMX corpus.

**Benchmark result:**
- R@5: -3.8pp compared to the baseline without HyDE
- Latency: ×12.5 (the LLM adds generation latency before retrieval)
- Verdict: net negative on this corpus

**Status:** Available as `rag-lab query "..." --hyde` for experimentation. Do not enable by
default unless a new corpus shows a clear benefit.

**Why it might work on other corpora:** HyDE helps when queries are very short or ambiguous and
the corpus is highly heterogeneous. The SDMX corpus is technical and specific: queries are
already sufficiently informative for dense retrieval.

### Global sparse search (without a dense candidate pool first)

The current architecture runs sparse search only over the candidate pool from vector search
(dense first, then sparse over that subset). There is no WAND or early termination in SQLite.

**Why frozen:** A global sparse scan over the full corpus in SQLite is O(N) over all documents.
At 610 chunks this is tolerable, but the architecture does not scale. Doing this correctly
requires a dedicated engine (Elasticsearch, Qdrant with native sparse support). SQLite is not
the right place for this.

---

## Possible product evolutions (long term)

These are not planned tasks — they are possible directions if usage of the system evolves.

### REST API / multi-user / authentication

Today RAG-Lab is a local CLI. If it needs to become a service accessible from other systems or
by multiple simultaneous users, the work would include:
- A FastAPI/Flask server over the retrieval and generation logic
- Authentication (API keys, OAuth)
- Session management and per-user context
- Conversation persistence

**Signal to act:** A real need for multi-user access or integration with other systems.

### Cloud LLM provider support

The system today assumes a local OpenAI-compatible endpoint (`http://localhost:8000/v1`).
Portability to OpenAI, Anthropic, Mistral AI, etc., would require:
- An abstraction layer over the LLM client (today it is a thin wrapper over the OpenAI API)
- Cost management and rate limiting
- Per-environment provider configuration

**Signal to act:** Need to use the system outside the local environment or in CI with an
external provider.

### PDF/DOCX/HTML loaders

Today RAG-Lab only ingests Markdown. The reason is that text extraction from PDF/DOCX loses
structure: headings disappear, tables deform, text order can be scrambled. Chunking quality
depends directly on the quality of the Markdown input.

To add these loaders, the prerequisite is an audited conversion pipeline that preserves
document structure (not just extracts plain text).

---

## Do not do yet

An explicit list of things that have been considered and ruled out for the near future.

### PDF/DOCX/HTML loaders

Extraction loses structure. Not until there is an audited conversion pipeline that preserves
headings, tables, and content order. A conversion pipeline that produces low-quality Markdown
would degrade retrieval.

### AutoML / tabular data

Out of scope. RAG-Lab is for documents, not datasets. CSV/Parquet/DuckDB loaders were removed
in v1.5 as a scope guard. This decision is not reversed.

### Embedding model fine-tuning

High computational cost, risk of regression on the existing corpus, and requires an annotated
training corpus (query-document relevance pairs). Without that corpus, fine-tuning is a blind
bet. The improvement with the highest return on investment so far (heading context in v1.10,
+2.1pp R@5) was a feature-engineering change, not fine-tuning.

### Incremental indexing without full re-ingestion

Today adding or modifying a document requires ingesting it (`rag-lab ingest --doc`). Making
granular per-chunk updates without re-ingesting is complex and fragile. The cost of re-ingesting
a document is low. There is no demonstrated need for incremental indexing.

---

## Signals that would justify opening v1.20

There is no planned sprint. v1.20 will be opened when at least one of these conditions is met:

1. **Feedback with a clear pattern:** >50 feedback events with a recurring, avoidable failure
   type that the system could correct with a change to ranking or retrieval.

2. **New corpus revealing quality problems:** A new corpus is added (documentation for another
   standard, technical manuals) and reveals chunking, frontmatter, or retrieval issues that the
   current system does not handle well.

3. **Systematically failing question class:** A class of questions that should have answers in
   the corpus and the system consistently cannot answer correctly. Identified by the E2E
   benchmark or by direct use.

4. **E2E benchmark available:** Once the (question, expected answer) dataset exists, the first
   E2E benchmark result will likely reveal concrete improvements.

5. **Scale requirement:** The corpus grows to a size where current latency is unacceptable
   (>2s in retrieval) and there is a clear benefit to changing the storage architecture.

---

*Last updated: v1.19.1*
