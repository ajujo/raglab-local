"""CLI entry point: python -m rag_lab.benchmark

Usage
-----
    # Run all 5 variants on the default query file
    python -m rag_lab.benchmark

    # Custom query file and specific variants
    python -m rag_lab.benchmark --queries data/my_queries.yaml --variants dense hybrid full

    # Save JSON output and suppress markdown
    python -m rag_lab.benchmark --output results/bench_$(date +%Y%m%d).json --no-markdown

    # Fast run (CPU-only, skip reranker)
    python -m rag_lab.benchmark --variants dense bm25 dense_bm25 hybrid --device cpu
"""

import argparse
import sys
from pathlib import Path

from rag_lab.benchmark.pipeline_variants import ALL_VARIANT_NAMES, VARIANT_NAMES
from rag_lab.benchmark.runner import BenchmarkRunner
from rag_lab.config import DOC_CAP_N, MMR_LAMBDA, RETRIEVAL_TOP_K, RRF_K
from rag_lab.logging_config import setup_logging

_DEFAULT_QUERIES = Path(__file__).parent.parent.parent / "data" / "benchmark_queries.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG-Lab retrieval benchmark — compare 5 pipeline variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--queries",
        default=str(_DEFAULT_QUERIES),
        help=f"Path to queries YAML/JSON file (default: {_DEFAULT_QUERIES})",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=ALL_VARIANT_NAMES,
        default=None,
        metavar="VARIANT",
        help=(
            f"Variants to run (default: {VARIANT_NAMES}). "
            f"Diversity variants (opt-in): hybrid_cap, hybrid_mmr"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results as JSON to this path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RETRIEVAL_TOP_K,
        help=f"Number of results to retrieve per query (default: {RETRIEVAL_TOP_K} from config)",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help=f"RRF smoothing constant (default: {RRF_K} from config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device (cuda/cpu). Defaults to EMBEDDING_DEVICE from config.",
    )
    parser.add_argument(
        "--rerank-device",
        default=None,
        help="Reranker device (cuda/cpu). Defaults to RERANKER_DEVICE from config.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup pass (first query will be slower)",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Suppress markdown table output",
    )
    parser.add_argument(
        "--doc-cap",
        type=int,
        default=DOC_CAP_N,
        help=f"Max chunks per doc for hybrid_cap variant (default: {DOC_CAP_N} from config)",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=MMR_LAMBDA,
        help=f"MMR lambda for hybrid_mmr variant (default: {MMR_LAMBDA} from config)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (default: WARNING)",
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: queries file not found: {queries_path}", file=sys.stderr)
        return 1

    print(f"Loading queries from {queries_path}")
    queries = BenchmarkRunner.load_queries(queries_path)
    print(f"  {len(queries)} queries loaded")

    variants = args.variants or VARIANT_NAMES
    print(f"  Running variants: {variants}")

    runner = BenchmarkRunner(
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        embedding_device=args.device,
        rerank_device=args.rerank_device,
        doc_cap=args.doc_cap,
        mmr_lambda=args.mmr_lambda,
    )

    print("\nRunning benchmark…")
    result = runner.run(queries, variants=variants, warmup=not args.no_warmup)

    if args.output:
        BenchmarkRunner.save(result, args.output)
        print(f"\nResults saved to {args.output}")

    if not args.no_markdown:
        print()
        print(BenchmarkRunner.to_markdown(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
