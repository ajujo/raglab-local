"""Calibration runner: grid-search over retrieval parameters.

Efficiently tests 324+ parameter combinations by precomputing candidates
once at max-k per query, then applying all weight/rrf_k combinations as
pure in-memory operations (no repeated store round-trips).

Execution model
---------------
For each query (one pass over stores):
  1. Encode query → dense + sparse embeddings
  2. Fetch dense candidates at max(dense_k values) from ChromaDB  [once]
  3. Fetch BM25  candidates at max(bm25_k values)  from FTS5      [once]
  4. Compute sparse scores for the full candidate pool              [once]
  5. Preload chunk metadata for all candidates                      [once]

Then for each (dense_k, bm25_k, sparse_w, bm25_w, rrf_k) combination:
  - Slice precomputed rankings to (dense_k, bm25_k)
  - Filter sparse ranking to current pool
  - Apply weighted_rrf → compute metrics (all in-memory, <1 ms each)

This reduces store round-trips from O(|grid| × N_queries) to O(N_queries).
"""

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from rag_lab.benchmark.metrics import (
    aggregate_metrics,
    latency_percentiles,
    mrr,
    ndcg_at_k,
    recall_at_k,
    signal_stats,
)
from rag_lab.retrieval.fusion import weighted_rrf
from rag_lab.config import EMBEDDING_DEVICE, RERANKER_DEVICE, RETRIEVAL_TOP_K, RRF_K
from rag_lab.embedding.encoder import encode_chunks
from rag_lab.retrieval.sparse_scorer import load_sparse_for_chunks, rank_candidates_by_sparse
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")

DEFAULT_GRID = {
    "dense_k":       [30, 50, 100],
    "bm25_k":        [30, 50, 100],
    "sparse_weight": [0.25, 0.5, 0.75, 1.0],
    "bm25_weight":   [0.5, 0.75, 1.0],
    "rrf_k":         [20, 60, 100],
}


class CalibrationRunner:
    """Parameter grid-search for hybrid retrieval."""

    def __init__(
        self,
        top_k: int = RETRIEVAL_TOP_K,
        embedding_device: str = None,
        rerank_device: str = None,
    ):
        self.top_k = top_k
        self.embedding_device = embedding_device or EMBEDDING_DEVICE
        self.rerank_device = rerank_device or RERANKER_DEVICE

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def _init_stores(self) -> None:
        self.vs = VectorStore(); self.vs.initialize()
        self.ds = DocStore();    self.ds.initialize()
        self.fts = FTSStore();   self.fts.initialize()

    def _close_stores(self) -> None:
        if hasattr(self, "ds"): self.ds.close()
        if hasattr(self, "fts"): self.fts.close()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(
        self,
        queries: List[dict],
        grid: Optional[dict] = None,
        reranker_top_n: int = 5,
    ) -> dict:
        """Run the calibration grid.

        Args:
            queries: List of annotated query dicts.
            grid: Parameter grid dict (defaults to DEFAULT_GRID).
            reranker_top_n: After grid search, re-run top-N configs with reranker.

        Returns:
            Dict with 'grid_results' (all configs sorted), 'reranker_results',
            'per_query_analysis', and 'recommendation'.
        """
        grid = {**DEFAULT_GRID, **(grid or {})}
        self._init_stores()

        max_dk = max(grid["dense_k"])
        max_bk = max(grid["bm25_k"])

        logger.info(f"Calibration: {len(queries)} queries, "
                    f"grid={_grid_size(grid)} configs, max_dk={max_dk}, max_bk={max_bk}")

        # ------------------------------------------------------------------
        # Step 1: Encode all queries
        # ------------------------------------------------------------------
        print(f"  Encoding {len(queries)} queries…", flush=True)
        all_texts = [{"text": q.get("text", q.get("query", "")), "chunk_id": f"__q{i}__"}
                     for i, q in enumerate(queries)]
        dense_all, sparse_all = encode_chunks(
            all_texts, batch_size=len(queries), device=self.embedding_device
        )
        encoded = [
            (dense_all[i], sparse_all.get(f"__q{i}__", {}))
            for i in range(len(queries))
        ]

        # ------------------------------------------------------------------
        # Step 2: Precompute candidates + sparse scores per query
        # ------------------------------------------------------------------
        print("  Precomputing candidates…", flush=True)
        query_data = []  # one dict per query
        for qi, (qitem, (q_dense, q_sparse)) in enumerate(zip(queries, encoded)):
            qtext = qitem.get("text", qitem.get("query", ""))

            # Time ChromaDB at each k
            dense_at_k: Dict[int, List[str]] = {}
            dense_lat: Dict[int, float] = {}
            for dk in sorted(set(grid["dense_k"])):
                t0 = time.perf_counter()
                raw = self.vs.query(q_dense, dk)
                dense_lat[dk] = (time.perf_counter() - t0) * 1000
                dense_at_k[dk] = raw["ids"]

            # Time FTS5 at each k
            bm25_at_k: Dict[int, List[dict]] = {}
            bm25_lat: Dict[int, float] = {}
            for bk in sorted(set(grid["bm25_k"])):
                t0 = time.perf_counter()
                raw = self.fts.query(qtext, bk)
                bm25_lat[bk] = (time.perf_counter() - t0) * 1000
                bm25_at_k[bk] = raw   # [{id, bm25_score}]

            # Build full candidate pool (union of all max-k results)
            all_ids: List[str] = []
            seen: set = set()
            for cid in dense_at_k[max_dk] + [r["id"] for r in bm25_at_k[max_bk]]:
                if cid not in seen:
                    seen.add(cid)
                    all_ids.append(cid)

            # Compute sparse scores for all candidates
            t0 = time.perf_counter()
            sparse_data = load_sparse_for_chunks(self.ds._conn, all_ids)
            sparse_full = rank_candidates_by_sparse(q_sparse, all_ids, sparse_data)
            sparse_lat = (time.perf_counter() - t0) * 1000

            # Preload chunk metadata
            chunk_meta = {c["chunk_id"]: c for c in self.ds.get_by_ids(all_ids)}

            query_data.append({
                "qitem": qitem,
                "dense_at_k": dense_at_k,
                "bm25_at_k": bm25_at_k,
                "sparse_full": sparse_full,  # full sparse ranking, filtered per combo
                "chunk_meta": chunk_meta,
                "dense_lat": dense_lat,
                "bm25_lat": bm25_lat,
                "sparse_lat_ms": sparse_lat,
                "all_ids_set": seen,
            })

        # ------------------------------------------------------------------
        # Step 3: Grid search (pure in-memory)
        # ------------------------------------------------------------------
        print(f"  Running {_grid_size(grid)} configs × {len(queries)} queries…", flush=True)
        config_metrics: Dict[str, List[dict]] = {}  # config_key → per-query metrics

        for dk, bk, sw, bw, rk in itertools.product(
            grid["dense_k"], grid["bm25_k"],
            grid["sparse_weight"], grid["bm25_weight"],
            grid["rrf_k"],
        ):
            config_key = f"dk{dk}_bk{bk}_sw{sw}_bw{bw}_rk{rk}"
            per_query = []

            for qd in query_data:
                dense_slice = qd["dense_at_k"][dk]
                bm25_slice = qd["bm25_at_k"][bk]

                # Filter sparse to current candidate pool
                current_pool = set(dense_slice) | {r["id"] for r in bm25_slice}
                sparse_slice = [r for r in qd["sparse_full"] if r["id"] in current_pool]

                # Weighted RRF fusion
                fused = weighted_rrf(dense_slice, bm25_slice, sparse_slice,
                                     bm25_w=bw, sparse_w=sw, k=rk)[:self.top_k]

                # Build result chunks from metadata
                chunks = []
                for item in fused:
                    meta = qd["chunk_meta"].get(item["id"])
                    if meta:
                        c = {**meta}
                        c["rrf_score"] = item["rrf_score"]
                        c["dense_score"] = item["dense_score"]
                        c["bm25_score"] = item["bm25_score"]
                        c["sparse_score"] = item["sparse_score"]
                        c["in_dense_topk"] = item["in_dense_topk"]
                        c["in_bm25_topk"] = item["in_bm25_topk"]
                        c["in_sparse_topk"] = item["in_sparse_topk"]
                        chunks.append(c)

                # Estimated latency: fetch + sparse + tiny fusion overhead
                est_lat = (qd["dense_lat"].get(dk, 3.0)
                           + qd["bm25_lat"].get(bk, 1.0)
                           + qd["sparse_lat_ms"] * len(current_pool) / max(len(qd["all_ids_set"]), 1))

                pq = {
                    "query_id": qd["qitem"].get("id", "?"),
                    "recall@5":  recall_at_k(chunks, qd["qitem"], 5),
                    "recall@10": recall_at_k(chunks, qd["qitem"], 10),
                    "recall@30": recall_at_k(chunks, qd["qitem"], 30),
                    "mrr":       mrr(chunks, qd["qitem"]),
                    "ndcg@10":   ndcg_at_k(chunks, qd["qitem"], 10),
                    "candidate_pool_size": len(current_pool),
                    "latency_ms": est_lat,
                    **signal_stats(chunks),
                }
                per_query.append(pq)

            config_metrics[config_key] = per_query

        # ------------------------------------------------------------------
        # Step 4: Aggregate and sort
        # ------------------------------------------------------------------
        grid_rows = []
        for dk, bk, sw, bw, rk in itertools.product(
            grid["dense_k"], grid["bm25_k"],
            grid["sparse_weight"], grid["bm25_weight"],
            grid["rrf_k"],
        ):
            config_key = f"dk{dk}_bk{bk}_sw{sw}_bw{bw}_rk{rk}"
            agg = aggregate_metrics(config_metrics[config_key])
            lats = [pq["latency_ms"] for pq in config_metrics[config_key]]
            lat_stats = latency_percentiles(lats)

            grid_rows.append({
                "config": {"dense_k": dk, "bm25_k": bk, "sparse_weight": sw,
                           "bm25_weight": bw, "rrf_k": rk},
                "ndcg@10":   agg.get("ndcg@10", 0),
                "mrr":       agg.get("mrr", 0),
                "recall@5":  agg.get("recall@5", 0),
                "recall@10": agg.get("recall@10", 0),
                "recall@30": agg.get("recall@30", 0),
                "p50_ms":    lat_stats["p50"],
                "p95_ms":    lat_stats["p95"],
                "pool_mean": agg.get("candidate_pool_size", 0),
                "sparse_cov": agg.get("sparse_coverage", 0),
                "per_query": config_metrics[config_key],
            })

        # Sort: nDCG@10 DESC, MRR DESC, R@5 DESC, latency ASC
        grid_rows.sort(key=lambda r: (-r["ndcg@10"], -r["mrr"], -r["recall@5"], r["p50_ms"]))

        # ------------------------------------------------------------------
        # Step 5: Reranker on top-N
        # ------------------------------------------------------------------
        reranker_rows = []
        if reranker_top_n > 0:
            reranker_rows = self._run_reranker_on_top(
                grid_rows[:reranker_top_n], queries, encoded, reranker_top_n
            )

        # ------------------------------------------------------------------
        # Step 6: Per-query analysis
        # ------------------------------------------------------------------
        analysis = self._per_query_analysis(queries, query_data, grid)

        self._close_stores()

        recommendation = self._recommend(grid_rows)

        return {
            "config": {
                "grid": grid,
                "top_k": self.top_k,
                "n_queries": len(queries),
                "n_configs": len(grid_rows),
            },
            "grid_results": grid_rows,
            "reranker_results": reranker_rows,
            "per_query_analysis": analysis,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Reranker evaluation on top-N base configs
    # ------------------------------------------------------------------

    def _run_reranker_on_top(
        self,
        top_rows: list,
        queries: List[dict],
        encoded: list,
        n: int,
    ) -> list:
        from rag_lab.retrieval.reranker import rerank as _rerank

        print(f"  Running reranker on top-{n} configs…", flush=True)
        reranked = []

        for row in top_rows:
            cfg = row["config"]
            per_query_reranked = []

            for qi, qitem in enumerate(queries):
                qtext = qitem.get("text", qitem.get("query", ""))
                q_dense, q_sparse = encoded[qi]

                from rag_lab.retrieval.hybrid_search import hybrid_search
                t0 = time.perf_counter()
                chunks, stats = hybrid_search(
                    qtext, self.vs, self.ds, self.fts,
                    query_dense=q_dense, query_sparse=q_sparse,
                    top_k=self.top_k, rrf_k=cfg["rrf_k"],
                    _return_stats=True,
                )
                if chunks:
                    chunks = _rerank(qtext, chunks, top_k=len(chunks),
                                     device=self.rerank_device)
                lat = (time.perf_counter() - t0) * 1000

                per_query_reranked.append({
                    "query_id": qitem.get("id", "?"),
                    "recall@5":  recall_at_k(chunks, qitem, 5),
                    "recall@10": recall_at_k(chunks, qitem, 10),
                    "recall@30": recall_at_k(chunks, qitem, 30),
                    "mrr":       mrr(chunks, qitem),
                    "ndcg@10":   ndcg_at_k(chunks, qitem, 10),
                    "latency_ms": lat,
                    **signal_stats(chunks),
                })

            agg = aggregate_metrics(per_query_reranked)
            lats = [pq["latency_ms"] for pq in per_query_reranked]
            lat_stats = latency_percentiles(lats)

            reranked.append({
                "base_config": cfg,
                "ndcg@10":  agg.get("ndcg@10", 0),
                "mrr":      agg.get("mrr", 0),
                "recall@5": agg.get("recall@5", 0),
                "recall@10": agg.get("recall@10", 0),
                "p50_ms":   lat_stats["p50"],
                "p95_ms":   lat_stats["p95"],
                "per_query": per_query_reranked,
            })

        reranked.sort(key=lambda r: (-r["ndcg@10"], -r["mrr"], -r["recall@5"]))
        return reranked

    # ------------------------------------------------------------------
    # Per-query analysis: hybrid vs dense_bm25 at default params
    # ------------------------------------------------------------------

    def _per_query_analysis(
        self, queries: List[dict], query_data: list, grid: dict
    ) -> List[dict]:
        """For each query, compare the two configs and show score-level detail."""
        # Default params: first value of each grid key
        dk_default = 30
        bk_default = 30
        rk_default = 60

        analysis = []
        for qd in query_data:
            qitem = qd["qitem"]
            dense_slice = qd["dense_at_k"].get(dk_default, qd["dense_at_k"][min(qd["dense_at_k"])])
            bm25_slice  = qd["bm25_at_k"].get(bk_default,  qd["bm25_at_k"][min(qd["bm25_at_k"])])
            pool = set(dense_slice) | {r["id"] for r in bm25_slice}
            sparse_slice = [r for r in qd["sparse_full"] if r["id"] in pool]

            # dense_bm25 (sparse_w=0)
            fused_db2 = weighted_rrf(dense_slice, bm25_slice, [], bm25_w=1.0, sparse_w=0.0,
                                     k=rk_default)[:self.top_k]
            chunks_db2 = [
                {**qd["chunk_meta"][r["id"]], **r} for r in fused_db2
                if r["id"] in qd["chunk_meta"]
            ]

            # hybrid (sparse_w=1.0)
            fused_hy = weighted_rrf(dense_slice, bm25_slice, sparse_slice,
                                    bm25_w=1.0, sparse_w=1.0, k=rk_default)[:self.top_k]
            chunks_hy = [
                {**qd["chunk_meta"][r["id"]], **r} for r in fused_hy
                if r["id"] in qd["chunk_meta"]
            ]

            r5_db2 = recall_at_k(chunks_db2, qitem, 5)
            r5_hy  = recall_at_k(chunks_hy,  qitem, 5)

            # Build comparison of top-5 positions
            def _fmt_chunk(c):
                return {
                    "chunk_id": c.get("chunk_id", c.get("id", "?")),
                    "doc_id":   c.get("doc_id", "?"),
                    "rrf":      round(c.get("rrf_score", 0), 5),
                    "bm25":     round(c.get("bm25_score", 0), 3),
                    "sparse":   round(c.get("sparse_score", 0), 4),
                    "in_bm25":  c.get("in_bm25_topk", False),
                    "in_sparse": c.get("in_sparse_topk", False),
                }

            row = {
                "query_id":      qitem.get("id"),
                "query":         qitem.get("text", ""),
                "r5_dense_bm25": r5_db2,
                "r5_hybrid":     r5_hy,
                "r5_diff":       round(r5_hy - r5_db2, 4),
                "mrr_dense_bm25": mrr(chunks_db2, qitem),
                "mrr_hybrid":    mrr(chunks_hy, qitem),
                "top5_dense_bm25": [_fmt_chunk(c) for c in chunks_db2[:5]],
                "top5_hybrid":   [_fmt_chunk(c) for c in chunks_hy[:5]],
            }

            # If hybrid is worse, identify displaced docs and over-represented docs
            if r5_hy < r5_db2:
                from collections import Counter
                docs_in_db2_top5 = {c.get("doc_id") for c in chunks_db2[:5]}
                docs_in_hy_top5  = {c.get("doc_id") for c in chunks_hy[:5]}
                slots_db2 = Counter(c.get("doc_id") for c in chunks_db2[:5])
                slots_hy  = Counter(c.get("doc_id") for c in chunks_hy[:5])

                row["displaced_docs"] = sorted(docs_in_db2_top5 - docs_in_hy_top5)
                # New docs entering top-5 from hybrid
                row["intruder_docs"]  = sorted(docs_in_hy_top5 - docs_in_db2_top5)
                # Docs that take MORE slots in hybrid (crowd out others)
                row["over_represented"] = sorted(
                    d for d in slots_hy
                    if slots_hy[d] > slots_db2.get(d, 0)
                )

                # Find the chunk driving the over-representation (highest sparse_score)
                problem_docs = set(row["intruder_docs"]) | set(row["over_represented"])
                problem_chunks = [c for c in chunks_hy[:5] if c.get("doc_id") in problem_docs]
                if problem_chunks:
                    worst = max(problem_chunks, key=lambda c: c.get("sparse_score", 0))
                    row["top_intruder"] = {
                        "chunk_id":    worst.get("chunk_id", worst.get("id", "?")),
                        "doc_id":      worst.get("doc_id"),
                        "sparse_score": round(worst.get("sparse_score", 0), 4),
                        "bm25_score":  round(worst.get("bm25_score", 0), 3),
                        "slots_db2":   slots_db2.get(worst.get("doc_id"), 0),
                        "slots_hybrid": slots_hy.get(worst.get("doc_id"), 0),
                        "reason":      "over-represented via sparse_score, crowding relevant docs",
                    }

            analysis.append(row)

        return analysis

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def _recommend(grid_rows: list) -> dict:
        """Recommend the config that best balances MRR, R@5, and latency."""
        if not grid_rows:
            return {}

        # Already sorted by nDCG@10, MRR, R@5, latency
        best = grid_rows[0]

        # Find best R@5 (may differ from best nDCG)
        best_r5 = max(grid_rows, key=lambda r: (r["recall@5"], r["ndcg@10"]))

        # Balanced: maximize (0.4*nDCG + 0.4*MRR + 0.2*R@5) with latency < 20ms
        def _score(r):
            lat_ok = r["p50_ms"] < 20
            return (0.4 * r["ndcg@10"] + 0.4 * r["mrr"] + 0.2 * r["recall@5"]) * (1.0 if lat_ok else 0.9)

        balanced = max(grid_rows, key=_score)

        return {
            "best_ndcg":    {"config": best["config"], "ndcg@10": best["ndcg@10"],
                             "mrr": best["mrr"], "recall@5": best["recall@5"],
                             "p50_ms": best["p50_ms"]},
            "best_r5":      {"config": best_r5["config"], "ndcg@10": best_r5["ndcg@10"],
                             "mrr": best_r5["mrr"], "recall@5": best_r5["recall@5"],
                             "p50_ms": best_r5["p50_ms"]},
            "balanced":     {"config": balanced["config"], "ndcg@10": balanced["ndcg@10"],
                             "mrr": balanced["mrr"], "recall@5": balanced["recall@5"],
                             "p50_ms": balanced["p50_ms"]},
        }

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def to_markdown(result: dict, top_n: int = 20) -> str:
        rows = result["grid_results"][:top_n]
        rec  = result.get("recommendation", {})

        lines = [
            "## Calibration Results",
            "",
            f"- Queries: {result['config']['n_queries']}  "
            f"Configs tested: {result['config']['n_configs']}  "
            f"top_k: {result['config']['top_k']}",
            f"- Grid: {result['config']['grid']}",
            "",
            f"### Top-{top_n} configurations (sorted: nDCG@10 ↓, MRR ↓, R@5 ↓, latency ↑)",
            "",
            "| # | dk | bk | sw | bw | rk | nDCG@10 | MRR | R@5 | R@10 | P50ms | Pool |",
            "|---|----|----|----|----|-----|---------|-----|-----|------|-------|------|",
        ]
        for i, r in enumerate(rows, 1):
            c = r["config"]
            lines.append(
                f"| {i:2d} | {c['dense_k']:3d} | {c['bm25_k']:3d} | "
                f"{c['sparse_weight']:.2f} | {c['bm25_weight']:.2f} | {c['rrf_k']:3d} | "
                f"{r['ndcg@10']:.4f}  | {r['mrr']:.4f} | {r['recall@5']:.4f} | "
                f"{r['recall@10']:.4f} | {r['p50_ms']:5.1f} | {r['pool_mean']:.0f} |"
            )

        if rec:
            lines += [
                "",
                "### Recommendation",
                "",
                f"**Best nDCG@10** (overall quality): "
                f"`{rec['best_ndcg']['config']}`  "
                f"nDCG@10={rec['best_ndcg']['ndcg@10']:.4f}  "
                f"MRR={rec['best_ndcg']['mrr']:.4f}  "
                f"R@5={rec['best_ndcg']['recall@5']:.4f}  "
                f"P50={rec['best_ndcg']['p50_ms']:.0f}ms",
                "",
                f"**Best R@5** (coverage): "
                f"`{rec['best_r5']['config']}`  "
                f"nDCG@10={rec['best_r5']['ndcg@10']:.4f}  "
                f"MRR={rec['best_r5']['mrr']:.4f}  "
                f"R@5={rec['best_r5']['recall@5']:.4f}  "
                f"P50={rec['best_r5']['p50_ms']:.0f}ms",
                "",
                f"**Balanced** (40% nDCG + 40% MRR + 20% R@5): "
                f"`{rec['balanced']['config']}`  "
                f"nDCG@10={rec['balanced']['ndcg@10']:.4f}  "
                f"MRR={rec['balanced']['mrr']:.4f}  "
                f"R@5={rec['balanced']['recall@5']:.4f}  "
                f"P50={rec['balanced']['p50_ms']:.0f}ms",
            ]

        # Reranker results
        rr = result.get("reranker_results", [])
        if rr:
            lines += [
                "",
                "### Top configs + cross-encoder reranker",
                "",
                "| # | dk | bk | sw | bw | rk | nDCG@10 | MRR | R@5 | P50ms |",
                "|---|----|----|----|----|-----|---------|-----|-----|-------|",
            ]
            for i, r in enumerate(rr, 1):
                c = r["base_config"]
                lines.append(
                    f"| {i:2d} | {c['dense_k']:3d} | {c['bm25_k']:3d} | "
                    f"{c['sparse_weight']:.2f} | {c['bm25_weight']:.2f} | {c['rrf_k']:3d} | "
                    f"{r['ndcg@10']:.4f}  | {r['mrr']:.4f} | {r['recall@5']:.4f} | "
                    f"{r['p50_ms']:5.1f} |"
                )

        # Per-query analysis
        analysis = result.get("per_query_analysis", [])
        if analysis:
            lines += [
                "",
                "### Per-query analysis: hybrid vs dense_bm25 (default params: dk=30, bk=30, rk=60)",
                "",
                "| query | R@5 db2 | R@5 hybrid | diff | diagnosis |",
                "|-------|---------|-----------|------|-----------|",
            ]
            for a in analysis:
                diff = a["r5_diff"]
                if diff < 0:
                    intruders = a.get("intruder_docs", [])
                    over_reps = a.get("over_represented", [])
                    displaced = a.get("displaced_docs", [])
                    ti = a.get("top_intruder", {})
                    if intruders:
                        diag = f"sparse injects `{', '.join(intruders)}` into top-5, displacing `{', '.join(displaced)}`"
                    elif over_reps:
                        diag = (f"`{', '.join(over_reps)}` over-represented "
                                f"({ti.get('slots_db2','?')}→{ti.get('slots_hybrid','?')} slots), "
                                f"crowding out `{', '.join(displaced)}`")
                    else:
                        diag = f"displacing `{', '.join(displaced)}`"
                elif diff > 0:
                    diag = "sparse helps"
                else:
                    diag = "—"
                lines.append(
                    f"| {a['query_id']} | {a['r5_dense_bm25']:.3f} | {a['r5_hybrid']:.3f} | "
                    f"{diff:+.3f} | {diag} |"
                )

        return "\n".join(lines)

    @staticmethod
    def save(result: dict, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Remove large per_query from grid_results before saving (keeps file small)
        slim = {k: v for k, v in result.items() if k != "grid_results"}
        slim["grid_results"] = [
            {k: v for k, v in r.items() if k != "per_query"}
            for r in result["grid_results"]
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2, ensure_ascii=False)
        logger.info(f"Calibration results saved to {path}")


def _grid_size(grid: dict) -> int:
    import math
    return math.prod(len(v) for v in grid.values())
