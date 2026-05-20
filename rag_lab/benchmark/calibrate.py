"""CLI: python -m rag_lab.benchmark.calibrate

Runs a parameter grid-search over retrieval hyperparameters and outputs
a ranked comparison table plus per-query analysis.

Usage
-----
    # Full grid (324 configs) on default queries
    python -m rag_lab.benchmark.calibrate

    # Custom grid
    python -m rag_lab.benchmark.calibrate \\
        --dense-k 30 50 --bm25-k 30 50 --sparse-weight 0.5 1.0 \\
        --bm25-weight 0.75 1.0 --rrf-k 60 \\
        --output data/calibration.json

    # With reranker evaluation on top-5 configs
    python -m rag_lab.benchmark.calibrate --reranker-top 5
"""

import argparse
import sys
from pathlib import Path

from rag_lab.benchmark.calibration import DEFAULT_GRID, CalibrationRunner
from rag_lab.benchmark.runner import BenchmarkRunner
from rag_lab.logging_config import setup_logging

_DEFAULT_QUERIES = (
    Path(__file__).parent.parent.parent / "data" / "benchmark_queries.yaml"
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG-Lab retrieval calibration — grid-search over hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--queries",
        default=str(_DEFAULT_QUERIES),
        help=f"Path to queries YAML/JSON (default: {_DEFAULT_QUERIES})",
    )
    parser.add_argument(
        "--dense-k", nargs="+", type=int,
        default=DEFAULT_GRID["dense_k"],
        metavar="K",
        help="Dense candidate counts to test (default: 30 50 100)",
    )
    parser.add_argument(
        "--bm25-k", nargs="+", type=int,
        default=DEFAULT_GRID["bm25_k"],
        metavar="K",
        help="BM25 candidate counts to test (default: 30 50 100)",
    )
    parser.add_argument(
        "--sparse-weight", nargs="+", type=float,
        default=DEFAULT_GRID["sparse_weight"],
        metavar="W",
        help="Sparse RRF weights to test (default: 0.25 0.5 0.75 1.0)",
    )
    parser.add_argument(
        "--bm25-weight", nargs="+", type=float,
        default=DEFAULT_GRID["bm25_weight"],
        metavar="W",
        help="BM25 RRF weights to test (default: 0.5 0.75 1.0)",
    )
    parser.add_argument(
        "--rrf-k", nargs="+", type=int,
        default=DEFAULT_GRID["rrf_k"],
        metavar="K",
        help="RRF smoothing constants to test (default: 20 60 100)",
    )
    parser.add_argument(
        "--reranker-top", type=int, default=0, metavar="N",
        help="Run reranker on top-N configs (0 = skip, default: 0)",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top configs to show in table (default: 20)",
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Retrieval top-k for all configs (default: 30)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results as JSON to this path",
    )
    parser.add_argument(
        "--device", default=None,
        help="Embedding device (cuda/cpu)",
    )
    parser.add_argument(
        "--no-markdown", action="store_true",
        help="Suppress markdown output",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        help="Logging level (default: WARNING)",
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: queries file not found: {queries_path}", file=sys.stderr)
        return 1

    queries = BenchmarkRunner.load_queries(queries_path)
    grid = {
        "dense_k":       args.dense_k,
        "bm25_k":        args.bm25_k,
        "sparse_weight": args.sparse_weight,
        "bm25_weight":   args.bm25_weight,
        "rrf_k":         args.rrf_k,
    }

    from rag_lab.benchmark.calibration import _grid_size
    n_configs = _grid_size(grid)
    print(f"Calibration: {len(queries)} queries × {n_configs} configs")

    runner = CalibrationRunner(
        top_k=args.top_k,
        embedding_device=args.device,
    )

    result = runner.run(queries, grid=grid, reranker_top_n=args.reranker_top)

    if args.output:
        CalibrationRunner.save(result, args.output)
        print(f"Results saved to {args.output}")

    if not args.no_markdown:
        print()
        print(CalibrationRunner.to_markdown(result, top_n=args.top_n))

    return 0


if __name__ == "__main__":
    sys.exit(main())
