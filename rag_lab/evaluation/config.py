"""Configuration for the E2E evaluation module."""

from pathlib import Path

_BASE_DIR = Path(__file__).parent.parent.parent

EVAL_DEFAULT_SUITE = "official"
EVAL_OUTPUT_DIR = _BASE_DIR / "data" / "eval_runs"
EVAL_BENCHMARK_QUERIES_PATH = _BASE_DIR / "data" / "benchmark_queries.yaml"
EVAL_TEMPERATURE = 0.0
EVAL_TOP_K = 50
EVAL_RERANK_TOP_K = 8
EVAL_DISABLE_CACHE = True
