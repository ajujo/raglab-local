"""E2E evaluation runner.

Runs the full RAG pipeline for each EvalSample and writes JSONL output
incrementally. Each line is a self-contained EvalResult serialized as JSON.

Design rules (from V1.21_RAGAS_EVAL_PLAN.md):
- Reuses the same pipeline modules as rag_lab.cli — no duplicated logic.
- Cache is always disabled for reproducibility.
- temperature=0 for deterministic LLM output.
- Errors on individual queries are captured; the run continues.
- Output is written line-by-line so partial runs are preserved.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rag_lab.config import EMBEDDING_DEVICE, RERANKER_DEVICE
from rag_lab.embedding.encoder import encode_chunks
from rag_lab.evaluation.config import (
    EVAL_RERANK_TOP_K,
    EVAL_TEMPERATURE,
    EVAL_TOP_K,
)
from rag_lab.evaluation.eval_utils import strip_inline_citations_for_eval
from rag_lab.evaluation.types import EvalResult, EvalSample
from rag_lab.exceptions import LLMConnectionError, RAGLabError
from rag_lab.generation.llm_client import generate_response
from rag_lab.generation.prompt_builder import build_prompt
from rag_lab.retrieval.hybrid_search import hybrid_search
from rag_lab.retrieval.query_processor import process_query
from rag_lab.retrieval.reranker import rerank
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore
from rag_lab.verification.pipeline import verify_and_score

logger = logging.getLogger("rag_lab")


class EvalRunner:
    """Runs EvalSamples through the full RAG pipeline.

    Stores are initialized once and reused across all queries.
    """

    def __init__(
        self,
        embedding_device: str | None = None,
        rerank_device: str | None = None,
        top_k: int = EVAL_TOP_K,
        rerank_top_k: int = EVAL_RERANK_TOP_K,
        temperature: float = EVAL_TEMPERATURE,
    ):
        self.embedding_device = embedding_device or EMBEDDING_DEVICE
        self.rerank_device = rerank_device or RERANKER_DEVICE
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.temperature = temperature
        self._stores_initialized = False

    def _init_stores(self) -> None:
        if self._stores_initialized:
            return
        self._vector_store = VectorStore()
        self._vector_store.initialize()
        self._fts_store = FTSStore()
        self._fts_store.initialize()
        self._doc_store = DocStore()
        self._doc_store.initialize()
        self._stores_initialized = True
        logger.info("EvalRunner: stores initialized")

    def _close_stores(self) -> None:
        if hasattr(self, "_doc_store"):
            self._doc_store.close()
        if hasattr(self, "_fts_store"):
            self._fts_store.close()

    def run_single(self, sample: EvalSample) -> EvalResult:
        """Run one sample through the full pipeline.

        Never raises — errors are captured in EvalResult.error.
        """
        self._init_stores()
        t0 = time.monotonic()

        try:
            # 1. Query variants (no HyDE, no rewrite — deterministic baseline)
            queries = process_query(sample.question, use_hyde=False, use_rewriting=False)

            # 2. Embed all variants
            all_query_data: list[tuple] = []
            for q in queries:
                dense_emb, sparse_map = encode_chunks(
                    [{"text": q["text"]}],
                    batch_size=1,
                    device=self.embedding_device,
                )
                query_dense = dense_emb[0]
                query_sparse = (
                    next(iter(sparse_map.values()), {})
                    if q.get("use_for_sparse", True)
                    else {}
                )
                all_query_data.append((query_dense, query_sparse))

            # 3. Hybrid search
            all_results: list[dict] = []
            for query_dense, query_sparse in all_query_data:
                results = hybrid_search(
                    sample.question,
                    self._vector_store,
                    self._doc_store,
                    self._fts_store,
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    top_k=self.top_k,
                )
                all_results.extend(results)

            # Deduplicate by chunk_id
            seen: set[str] = set()
            unique_results: list[dict] = []
            for r in all_results:
                cid = r.get("chunk_id")
                if cid not in seen:
                    seen.add(cid)
                    unique_results.append(r)

            # 4. Rerank
            if unique_results:
                unique_results = rerank(
                    sample.question,
                    unique_results[:20],
                    top_k=self.rerank_top_k,
                    device=self.rerank_device,
                )

            top_chunks = unique_results[:self.rerank_top_k]

            # 5. Generate
            answer = ""
            if top_chunks:
                system_prompt, user_prompt = build_prompt(sample.question, top_chunks)
                answer = generate_response(
                    system_prompt, user_prompt, temperature=self.temperature
                )

            # 6. Verify
            retrieval_scores = [
                c.get("rerank_score", c.get("rrf_score", c.get("score", 0.5)))
                for c in top_chunks
            ]
            verification = verify_and_score(
                answer,
                top_chunks,
                retrieval_scores,
                enable_consistency_check=True,
            )

            latency_ms = int((time.monotonic() - t0) * 1000)

            contexts = [c.get("text", "") for c in top_chunks]
            context_metadata = [
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "doc_id": c.get("doc_id", ""),
                    "heading_path": c.get("heading_path", ""),
                    "rerank_score": c.get("rerank_score", c.get("rrf_score", 0.0)),
                }
                for c in top_chunks
            ]
            citations = [
                {
                    "chunk_id": cr.chunk_id,
                    "doc_id": cr.matched_chunk.get("doc_id") if cr.matched_chunk else None,
                    "lines": (
                        f"{cr.matched_chunk.get('line_start')}-{cr.matched_chunk.get('line_end')}"
                        if cr.matched_chunk else None
                    ),
                    "status": cr.status.value,
                }
                for cr in verification.citation_results
            ]

            return EvalResult(
                sample=sample,
                answer=answer,
                answer_for_eval=strip_inline_citations_for_eval(answer),
                contexts=contexts,
                context_metadata=context_metadata,
                citations=citations,
                trust_score=verification.score_result.final_score,
                trust_level=verification.score_result.confidence_level.value,
                latency_ms=latency_ms,
                error=None,
            )

        except (LLMConnectionError, RAGLabError, Exception) as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.error("EvalRunner: query %s failed: %s", sample.query_id, exc)
            return EvalResult(
                sample=sample,
                answer="",
                contexts=[],
                context_metadata=[],
                citations=[],
                trust_score=0.0,
                trust_level="LOW",
                latency_ms=latency_ms,
                error=str(exc),
            )


def run_eval(
    samples: list[EvalSample],
    output_path: Path,
    embedding_device: str | None = None,
    rerank_device: str | None = None,
    top_k: int = EVAL_TOP_K,
    rerank_top_k: int = EVAL_RERANK_TOP_K,
    temperature: float = EVAL_TEMPERATURE,
) -> Path:
    """Run the full E2E evaluation and write results to a JSONL file.

    Writes one JSON line per query immediately — partial runs are preserved
    if the process is interrupted.

    Args:
        samples: Queries to evaluate.
        output_path: Destination JSONL file.
        embedding_device: Override embedding device.
        rerank_device: Override reranker device.
        top_k: Retrieval pool size before reranking.
        rerank_top_k: Top-K chunks passed to LLM.
        temperature: LLM temperature (default 0.0 for determinism).

    Returns:
        The output_path that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runner = EvalRunner(
        embedding_device=embedding_device,
        rerank_device=rerank_device,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        temperature=temperature,
    )

    logger.info("EvalRunner: starting %d queries → %s", len(samples), output_path)

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            for i, sample in enumerate(samples, 1):
                logger.info("EvalRunner: [%d/%d] %s", i, len(samples), sample.query_id)
                result = runner.run_single(sample)
                fh.write(result.to_jsonl_line() + "\n")
                fh.flush()
    finally:
        runner._close_stores()

    logger.info("EvalRunner: done — %d lines written to %s", len(samples), output_path)
    return output_path
