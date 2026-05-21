"""BenchmarkRunner: orchestrates multi-variant retrieval evaluation.

Usage
-----
    from rag_lab.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner()
    queries = BenchmarkRunner.load_queries("data/benchmark_queries.yaml")
    results = runner.run(queries, variants=["dense", "hybrid", "full"])
    print(BenchmarkRunner.to_markdown(results))
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from rag_lab.benchmark.metrics import (
    aggregate_metrics,
    diversity_stats,
    latency_percentiles,
    mrr,
    ndcg_at_k,
    recall_at_k,
    signal_stats,
)
from rag_lab.benchmark.pipeline_variants import ALL_VARIANT_NAMES, VARIANT_NAMES, run_variant
from rag_lab.config import (
    DOC_CAP_N,
    EMBEDDING_DEVICE,
    MMR_LAMBDA,
    RERANKER_DEVICE,
    RETRIEVAL_TOP_K,
    RRF_K,
)
from rag_lab.embedding.encoder import encode_chunks
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")


class BenchmarkRunner:
    """Run all (or a subset of) retrieval pipeline variants against a query set."""

    def __init__(
        self,
        top_k: int = RETRIEVAL_TOP_K,
        rrf_k: int = RRF_K,
        embedding_device: str = None,
        rerank_device: str = None,
        doc_cap: int = DOC_CAP_N,
        mmr_lambda: float = MMR_LAMBDA,
    ):
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.embedding_device = embedding_device or EMBEDDING_DEVICE
        self.rerank_device = rerank_device or RERANKER_DEVICE
        self.doc_cap = doc_cap
        self.mmr_lambda = mmr_lambda
        self._stores_initialized = False

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def _init_stores(self) -> None:
        if self._stores_initialized:
            return
        self.vector_store = VectorStore()
        self.vector_store.initialize()
        self.doc_store = DocStore()
        self.doc_store.initialize()
        self.fts_store = FTSStore()
        self.fts_store.initialize()
        self._stores_initialized = True
        logger.info("Benchmark stores initialized")

    def _close_stores(self) -> None:
        if hasattr(self, "doc_store"):
            self.doc_store.close()
        if hasattr(self, "fts_store"):
            self.fts_store.close()

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> tuple:
        """Encode query text. Returns (dense_vec, sparse_dict)."""
        dense, sparse_map = encode_chunks(
            [{"text": text, "chunk_id": "__query__"}],
            batch_size=1,
            device=self.embedding_device,
        )
        query_dense = dense[0]
        query_sparse = next(iter(sparse_map.values()), {})
        return query_dense, query_sparse

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self, variants: List[str]) -> None:
        """Run one dummy query to pre-load models and warm HNSW index."""
        logger.info("Warming up models and index…")
        self._init_stores()
        q_dense, q_sparse = self._encode("warmup query for model preloading")
        for name in variants:
            try:
                run_variant(
                    name, "warmup", q_dense, q_sparse,
                    self.vector_store, self.doc_store, self.fts_store,
                    top_k=3, rrf_k=self.rrf_k,
                    rerank_device=self.rerank_device,
                )
            except Exception:
                pass
        logger.info("Warmup complete")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        queries: List[dict],
        variants: Optional[List[str]] = None,
        warmup: bool = True,
    ) -> dict:
        """Run all variants on all queries.

        Args:
            queries: List of query dicts (from YAML).
            variants: Subset of VARIANT_NAMES to run (default: all).
            warmup: Whether to run a warmup pass before timing.

        Returns:
            Full result dict with per-query and aggregate metrics.
        """
        variants = variants or VARIANT_NAMES
        unknown = [v for v in variants if v not in ALL_VARIANT_NAMES]
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}. Valid: {ALL_VARIANT_NAMES}")

        self._init_stores()

        if warmup:
            self.warmup(variants)

        result = {
            "config": {
                "top_k": self.top_k,
                "rrf_k": self.rrf_k,
                "doc_cap": self.doc_cap,
                "mmr_lambda": self.mmr_lambda,
                "embedding_device": self.embedding_device,
                "rerank_device": self.rerank_device,
                "n_queries": len(queries),
                "variants": variants,
            },
            "results": {},
        }

        for variant in variants:
            logger.info(f"Running variant: {variant} on {len(queries)} queries")
            per_query = []
            latencies = []

            for qitem in queries:
                qid = qitem.get("id", "?")
                qtext = qitem.get("text", qitem.get("query", ""))

                # Encode (excluded from retrieval latency)
                q_dense, q_sparse = self._encode(qtext)

                # Run variant
                try:
                    chunks, stats = run_variant(
                        variant, qtext, q_dense, q_sparse,
                        self.vector_store, self.doc_store, self.fts_store,
                        top_k=self.top_k, rrf_k=self.rrf_k,
                        rerank_device=self.rerank_device,
                        doc_cap=self.doc_cap,
                        mmr_lambda=self.mmr_lambda,
                    )
                except Exception as exc:
                    logger.error(f"Variant {variant} failed on query {qid}: {exc}")
                    chunks, stats = [], {"latency_ms": 0, "candidate_pool_size": 0,
                                        "n_dense": 0, "n_bm25": 0, "n_sparse": 0}

                lat = stats.get("latency_ms", 0.0)
                latencies.append(lat)

                # Per-query metrics
                pq = {
                    "query_id": qid,
                    "query": qtext,
                    "n_results": len(chunks),
                    "latency_ms": round(lat, 2),
                    "candidate_pool_size": stats.get("candidate_pool_size", 0),
                    "n_dense": stats.get("n_dense", 0),
                    "n_bm25": stats.get("n_bm25", 0),
                    "n_sparse": stats.get("n_sparse", 0),
                    "recall@5": recall_at_k(chunks, qitem, 5),
                    "recall@10": recall_at_k(chunks, qitem, 10),
                    "recall@30": recall_at_k(chunks, qitem, 30),
                    "mrr": mrr(chunks, qitem),
                    "ndcg@10": ndcg_at_k(chunks, qitem, 10),
                    **signal_stats(chunks),
                    **diversity_stats(chunks),
                    "top5_doc_ids": [c.get("doc_id", "") for c in chunks[:5]],
                }
                per_query.append(pq)

            # Aggregate
            agg = aggregate_metrics(per_query)
            agg.update(latency_percentiles(latencies))

            result["results"][variant] = {
                "aggregate": agg,
                "per_query": per_query,
            }

        self._close_stores()
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_queries(path: Union[str, Path]) -> List[dict]:
        """Load queries from a YAML or JSON file."""
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix} (use .yaml or .json)")

        queries = data if isinstance(data, list) else data.get("queries", data)
        if not queries:
            raise ValueError("No queries found in file")
        return queries

    @staticmethod
    def to_markdown(result: dict) -> str:
        """Render aggregate metrics as a markdown table."""
        variants = result["config"]["variants"]
        cfg = result["config"]

        lines = [
            "## Retrieval Benchmark Results",
            "",
            f"- Corpus: {cfg.get('n_queries', '?')} queries evaluated",
            f"- top_k={cfg.get('top_k')}  rrf_k={cfg.get('rrf_k')}",
            "",
        ]

        # Table header
        cols = ["R@5", "R@10", "R@30", "MRR", "nDCG@10", "P50ms", "P95ms", "P99ms", "Pool"]
        lines.append("| Variant | " + " | ".join(cols) + " |")
        lines.append("|---------|" + "|".join(["-------"] * len(cols)) + "|")

        for v in variants:
            agg = result["results"][v]["aggregate"]
            row = [
                f"{agg.get('recall@5', 0):.3f}",
                f"{agg.get('recall@10', 0):.3f}",
                f"{agg.get('recall@30', 0):.3f}",
                f"{agg.get('mrr', 0):.3f}",
                f"{agg.get('ndcg@10', 0):.3f}",
                f"{agg.get('p50', 0):.0f}",
                f"{agg.get('p95', 0):.0f}",
                f"{agg.get('p99', 0):.0f}",
                f"{agg.get('candidate_pool_size', 0):.0f}",
            ]
            lines.append(f"| {v:<12} | " + " | ".join(row) + " |")

        lines.append("")
        lines.append("### Signal coverage (mean fraction of results with each signal)")
        lines.append("")
        lines.append("| Variant | Dense | BM25 | Sparse |")
        lines.append("|---------|-------|------|--------|")
        for v in variants:
            agg = result["results"][v]["aggregate"]
            row = [
                f"{agg.get('dense_coverage', 0):.2f}",
                f"{agg.get('bm25_coverage', 0):.2f}",
                f"{agg.get('sparse_coverage', 0):.2f}",
            ]
            lines.append(f"| {v:<12} | " + " | ".join(row) + " |")

        # Diversity table — only when any diversity metric is present
        first_agg = result["results"][variants[0]]["aggregate"]
        if "unique_docs@5" in first_agg:
            lines.append("")
            lines.append("### Document diversity (mean per query)")
            lines.append("")
            lines.append("| Variant | unique_docs@5 | max_same@5 | unique_docs@10 | max_same@10 |")
            lines.append("|---------|:---:|:---:|:---:|:---:|")
            for v in variants:
                agg = result["results"][v]["aggregate"]
                row = [
                    f"{agg.get('unique_docs@5', 0):.2f}",
                    f"{agg.get('max_chunks_same_doc@5', 0):.2f}",
                    f"{agg.get('unique_docs@10', 0):.2f}",
                    f"{agg.get('max_chunks_same_doc@10', 0):.2f}",
                ]
                lines.append(f"| {v:<12} | " + " | ".join(row) + " |")

        return "\n".join(lines)

    @staticmethod
    def save(result: dict, path: Union[str, Path]) -> None:
        """Save result dict as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Benchmark results saved to {path}")
