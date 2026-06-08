"""Loads EvalSample objects from benchmark_queries.yaml."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_lab.evaluation.config import EVAL_BENCHMARK_QUERIES_PATH, EVAL_DEFAULT_SUITE
from rag_lab.evaluation.types import EvalSample

logger = logging.getLogger("rag_lab")


class EvaluationError(Exception):
    pass


def load_eval_samples(
    suite: str = EVAL_DEFAULT_SUITE,
    path: Path | None = None,
    validated_only: bool = True,
) -> list[EvalSample]:
    """Load EvalSample objects from benchmark_queries.yaml.

    Args:
        suite: Keep only queries where suite == this value.
        path: Override the default benchmark_queries.yaml path.
        validated_only: Discard entries with validated=False.

    Returns:
        List of EvalSample, filtered and ready to evaluate.

    Raises:
        EvaluationError: If the file cannot be parsed.
        FileNotFoundError: If the path does not exist.
    """
    queries_path = Path(path) if path else EVAL_BENCHMARK_QUERIES_PATH

    try:
        if queries_path.suffix in (".yaml", ".yml"):
            import yaml
            with open(queries_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif queries_path.suffix == ".json":
            with open(queries_path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise EvaluationError(
                f"Unsupported format: {queries_path.suffix} (use .yaml or .json)"
            )
    except (OSError, IOError) as exc:
        raise FileNotFoundError(f"Cannot open {queries_path}: {exc}") from exc
    except Exception as exc:
        raise EvaluationError(f"Failed to parse {queries_path}: {exc}") from exc

    raw_queries: list[dict] = (
        data if isinstance(data, list) else data.get("queries", data)
    )
    if not raw_queries:
        raise EvaluationError(f"No queries found in {queries_path}")

    filtered = [
        q for q in raw_queries
        if q.get("suite", EVAL_DEFAULT_SUITE) == suite
        and (not validated_only or q.get("validated", True) is True)
    ]

    samples = [EvalSample.from_yaml_entry(q) for q in filtered]
    logger.info(
        "Loaded %d eval samples (suite=%s, validated_only=%s)",
        len(samples), suite, validated_only,
    )
    return samples
