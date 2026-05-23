"""HNSW profile benchmarking for vector_store.

Creates temporary ChromaDB collections with different HNSW parameters,
copies the production embeddings into each, and measures raw query latency
and recall vs the production collection.

This tool never modifies the production collection.

Usage:
    python -m rag_lab.maintenance.hnsw_profiles

Output:
    Per-profile table with latency stats and recall vs production.
"""

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from rag_lab.config import VECTOR_STORE_PATH, VECTOR_HNSW_SPACE

logger = logging.getLogger("rag_lab")

# ---------------------------------------------------------------------------
# Profiles to test
# ---------------------------------------------------------------------------

HNSW_PROFILES: Dict[str, dict] = {
    "current": {
        "hnsw:space": "cosine",
        "hnsw:M": 16,
        "hnsw:construction_ef": 100,
        "hnsw:search_ef": 100,
    },
    "fast": {
        "hnsw:space": "cosine",
        "hnsw:M": 8,
        "hnsw:construction_ef": 64,
        "hnsw:search_ef": 50,
    },
    "balanced": {
        "hnsw:space": "cosine",
        "hnsw:M": 16,
        "hnsw:construction_ef": 128,
        "hnsw:search_ef": 100,
    },
    "recall": {
        "hnsw:space": "cosine",
        "hnsw:M": 32,
        "hnsw:construction_ef": 200,
        "hnsw:search_ef": 200,
    },
}

N_QUERY_ITERS = 50   # repetitions per profile for latency measurement
TOP_K = 50           # matches production benchmark top_k
COLLECTION_NAME = "sdmx_rag"


def _load_production_embeddings(
    production_path: Path,
) -> Tuple[List[str], np.ndarray, List[str], List[dict]]:
    """Read all ids, embeddings, documents, metadatas from the production collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(production_path))
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()

    result = col.get(
        include=["embeddings", "documents", "metadatas"],
        limit=total,
    )
    ids = result["ids"]
    embeddings = np.array(result["embeddings"], dtype=np.float32)
    documents = result["documents"]
    metadatas = result["metadatas"]
    return ids, embeddings, documents, metadatas


def _build_temp_collection(
    tmp_dir: Path,
    profile_name: str,
    profile_meta: dict,
    ids: List[str],
    embeddings: np.ndarray,
    documents: List[str],
    metadatas: List[dict],
) -> Tuple[object, float]:
    """Create a temp collection with given params, index all vectors, return (col, build_ms)."""
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_dir))
    col = client.create_collection(
        name=f"profile_{profile_name}",
        metadata=profile_meta,
    )

    t0 = time.perf_counter()
    col.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    build_ms = (time.perf_counter() - t0) * 1000
    return col, build_ms


def _measure_query_latency(
    col: object,
    n_queries: int = N_QUERY_ITERS,
    top_k: int = TOP_K,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (p50, p95, p99) latency in ms over n_queries random queries."""
    rng = np.random.default_rng(seed)
    latencies = []
    for _ in range(n_queries):
        q = rng.standard_normal(1024).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-9
        t0 = time.perf_counter()
        col.query(query_embeddings=[q.tolist()], n_results=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    idx_p50 = int(0.50 * n_queries)
    idx_p95 = int(0.95 * n_queries)
    idx_p99 = int(0.99 * n_queries)
    return latencies[idx_p50], latencies[idx_p95], latencies[min(idx_p99, n_queries - 1)]


def _compute_recall(
    production_col: object,
    temp_col: object,
    n_queries: int = 30,
    top_k: int = TOP_K,
    seed: int = 99,
) -> float:
    """Recall of temp_col vs production_col on random queries (ground truth = production).

    Since both have the same vectors and the same distance metric, recall should be
    ~1.0 for any reasonable HNSW config at ≤1k vectors (brute-force is effectively used).
    """
    rng = np.random.default_rng(seed)
    total_recall = 0.0

    for _ in range(n_queries):
        q = rng.standard_normal(1024).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-9

        prod_res = production_col.query(
            query_embeddings=[q.tolist()], n_results=top_k
        )
        temp_res = temp_col.query(
            query_embeddings=[q.tolist()], n_results=top_k
        )

        prod_ids = set(prod_res["ids"][0])
        temp_ids = set(temp_res["ids"][0])
        overlap = len(prod_ids & temp_ids) / max(len(prod_ids), 1)
        total_recall += overlap

    return total_recall / n_queries


def run_profile_benchmark() -> List[dict]:
    """Run HNSW profile benchmark. Returns list of result dicts."""
    import chromadb

    production_path = VECTOR_STORE_PATH
    print(f"\nLoading production embeddings from {production_path} ...")
    ids, embeddings, documents, metadatas = _load_production_embeddings(production_path)
    n_vectors = len(ids)
    print(f"  {n_vectors} vectors loaded (dim={embeddings.shape[1]})")

    prod_client = chromadb.PersistentClient(path=str(production_path))
    prod_col = prod_client.get_collection(COLLECTION_NAME)

    results = []

    with tempfile.TemporaryDirectory(prefix="hnsw_profile_") as tmp_root:
        for profile_name, profile_meta in HNSW_PROFILES.items():
            print(f"\n[{profile_name}] M={profile_meta['hnsw:M']} "
                  f"ef_c={profile_meta['hnsw:construction_ef']} "
                  f"ef_s={profile_meta['hnsw:search_ef']}")

            tmp_dir = Path(tmp_root) / profile_name

            col, build_ms = _build_temp_collection(
                tmp_dir, profile_name, profile_meta,
                ids, embeddings, documents, metadatas,
            )
            print(f"  build: {build_ms:.0f}ms")

            p50, p95, p99 = _measure_query_latency(col, top_k=TOP_K)
            print(f"  latency: p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms")

            recall = _compute_recall(prod_col, col, top_k=TOP_K)
            print(f"  recall vs production: {recall:.4f}")

            results.append({
                "profile": profile_name,
                "M": profile_meta["hnsw:M"],
                "construction_ef": profile_meta["hnsw:construction_ef"],
                "search_ef": profile_meta["hnsw:search_ef"],
                "n_vectors": n_vectors,
                "build_ms": round(build_ms, 1),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "recall_vs_production": round(recall, 4),
            })

    return results


def print_results_table(results: List[dict]) -> None:
    """Print a markdown table of profile results."""
    print("\n## HNSW Profile Comparison")
    print(f"Corpus: {results[0]['n_vectors']} vectors | "
          f"top_k={TOP_K} | {N_QUERY_ITERS} query iterations\n")
    print(
        f"| Profile    |  M | ef_c | ef_s | build(ms) | p50(ms) | p95(ms) | p99(ms) | recall |"
    )
    print(
        f"|------------|----|----- |------|-----------|---------|---------|---------|--------|"
    )
    for r in results:
        print(
            f"| {r['profile']:<10} | {r['M']:>2} | {r['construction_ef']:>4} | "
            f"{r['search_ef']:>4} | {r['build_ms']:>9.0f} | {r['p50_ms']:>7.2f} | "
            f"{r['p95_ms']:>7.2f} | {r['p99_ms']:>7.2f} | {r['recall_vs_production']:.4f} |"
        )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    results = run_profile_benchmark()
    print_results_table(results)

    # Save results
    output = Path("/tmp/hnsw_profiles_result.json")
    output.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output}")
