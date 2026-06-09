"""RAGAS evaluator — runs in the 'ragas' conda env, NOT in rag-lab.

Reads a JSONL file produced by `rag-lab eval run` and evaluates it with
RAGAS metrics using an external LLM-as-judge (DeepSeek via OpenRouter).

Usage (from ragas env):
    python scripts/ragas_eval.py --input data/eval_runs/v1.21_baseline.jsonl
    python scripts/ragas_eval.py --input /tmp/f3_smoke.jsonl --metrics faithfulness
    python scripts/ragas_eval.py --input FILE --metrics faithfulness,answer_relevancy

Environment variables required (in .env or shell):
    OPENROUTER_API_KEY          API key for OpenRouter
    OPENROUTER_JUDGE_MODEL      Model slug (default: deepseek/deepseek-v4-flash)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force embeddings to CPU — GPU is occupied by the local LLM server
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def load_env() -> None:
    """Load .env from repo root if present."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def get_judge_llm():
    """Build a ragas-compatible LLM wrapper pointing at OpenRouter/DeepSeek."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_JUDGE_MODEL", "deepseek/deepseek-v4-flash")

    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set. Add it to .env or export it.")

    chat_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://github.com/ajujo/raglab-local",
            "X-Title": "RAG-Lab evaluation",
        },
    )
    return LangchainLLMWrapper(chat_llm)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_ragas_dataset(rows: list[dict], answer_field: str = "answer_for_eval"):
    """Convert eval JSONL rows to a RAGAS Dataset.

    Args:
        rows: Parsed JSONL rows from `rag-lab eval run`.
        answer_field: Which field to use as the answer for RAGAS metrics.
            Defaults to ``answer_for_eval`` (inline citations stripped),
            which gives cleaner answer_relevancy scores. Falls back to
            ``answer`` when ``answer_for_eval`` is absent or empty (e.g.
            JSONL files produced before this field was added).
    """
    from datasets import Dataset

    answers = []
    for r in rows:
        if answer_field == "answer_for_eval":
            # Prefer answer_for_eval; fall back to answer for old JSONL
            ans = r.get("answer_for_eval") or r.get("answer", "")
        else:
            ans = r.get(answer_field, r.get("answer", ""))
        answers.append(ans)

    data = {
        "question": [r["question"] for r in rows],
        "answer":   answers,
        "contexts": [r["contexts"] for r in rows],
    }
    return Dataset.from_dict(data)


METRIC_MAP = {
    "faithfulness":      lambda: __import__("ragas.metrics", fromlist=["faithfulness"]).faithfulness,
    "answer_relevancy":  lambda: __import__("ragas.metrics", fromlist=["answer_relevancy"]).answer_relevancy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluator for RAG-Lab JSONL output")
    parser.add_argument("--input", required=True, help="Path to eval JSONL file")
    parser.add_argument(
        "--metrics",
        default="faithfulness",
        help="Comma-separated metrics: faithfulness,answer_relevancy (default: faithfulness)",
    )
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument(
        "--answer-field",
        default="answer_for_eval",
        choices=["answer", "answer_for_eval"],
        help=(
            "Which JSONL field to pass as the answer to RAGAS. "
            "Default: answer_for_eval (inline citations stripped). "
            "Use --answer-field answer to evaluate the raw answer with citations."
        ),
    )
    args = parser.parse_args()

    load_env()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: Input file not found: {input_path}")

    rows = load_jsonl(input_path)
    # Skip rows that errored during eval run
    ok_rows = [r for r in rows if not r.get("error")]
    if not ok_rows:
        sys.exit("ERROR: No valid rows in input (all have errors).")

    print(f"Loaded {len(rows)} rows ({len(ok_rows)} valid, {len(rows)-len(ok_rows)} errors skipped)")

    # Parse requested metrics
    metric_names = [m.strip() for m in args.metrics.split(",")]
    unknown = [m for m in metric_names if m not in METRIC_MAP]
    if unknown:
        sys.exit(f"ERROR: Unknown metrics: {unknown}. Valid: {list(METRIC_MAP)}")

    metrics = [METRIC_MAP[m]() for m in metric_names]
    print(f"Metrics: {metric_names}")

    # Build judge LLM
    judge = get_judge_llm()
    model_name = os.environ.get("OPENROUTER_JUDGE_MODEL", "deepseek/deepseek-v4-flash")
    print(f"Judge: {model_name} via OpenRouter")

    # Build RAGAS dataset
    answer_field = args.answer_field
    n_with_eval = sum(1 for r in ok_rows if r.get("answer_for_eval"))
    if answer_field == "answer_for_eval" and n_with_eval == 0:
        print("WARNING: answer_for_eval not found in any row — falling back to answer.")
        print("         Run `rag-lab eval run` again to generate answer_for_eval fields.")
    else:
        print(f"Answer field: {answer_field} ({n_with_eval}/{len(ok_rows)} rows have answer_for_eval)")
    dataset = build_ragas_dataset(ok_rows, answer_field=answer_field)
    print(f"Dataset: {len(dataset)} rows")

    # Configure metrics with judge LLM + local embeddings (CPU)
    from ragas import evaluate
    from ragas.embeddings import HuggingfaceEmbeddings

    embeddings = HuggingfaceEmbeddings(model="BAAI/bge-small-en-v1.5")

    for metric in metrics:
        metric.llm = judge
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    print("\nRunning RAGAS evaluation...")
    result = evaluate(dataset, metrics=metrics)

    print("\n" + "=" * 50)
    print("RAGAS Results")
    print("=" * 50)
    for name in metric_names:
        score = result[name]
        print(f"  {name:<22} {score:.4f}")
    print("=" * 50)
    print(f"  n queries evaluated: {len(ok_rows)}")
    print(f"  judge model:         {model_name}")
    print("=" * 50)

    if args.output:
        out = {
            "input": str(input_path),
            "n_queries": len(ok_rows),
            "judge_model": model_name,
            "answer_field": answer_field,
            "scores": {name: float(result[name]) for name in metric_names},
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
