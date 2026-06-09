"""Diagnóstico detallado de answer_relevancy — RAG-Lab v1.22.

Lee el JSONL de un eval run, carga metadatos de queries, ejecuta RAGAS con
captura de scores por fila, y genera un informe de diagnóstico completo.

Ejecutar en env ragas:
    python scripts/ragas_diagnose.py \
        --input data/eval_runs/v1.21_baseline.jsonl \
        --output data/eval_runs/v1.21_diagnosis.json

Genera:
  - scores por query (raw + clean sin citas)
  - estadísticas globales y por segmento
  - top-10 peores/mejores
  - clasificación de causas probables
  - comparativa raw vs clean
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

CITATION_PAT = re.compile(
    r'\[\[\d+\][^\]]*\]'          # [[N] Fuente: ... | Líneas: ...]
    r'|\[\d+\]'                    # [N] solo
)
VERIF_PAT = re.compile(r'─{10,}.*?─{10,}', re.DOTALL)


# ---------------------------------------------------------------------------
# env + helpers
# ---------------------------------------------------------------------------

def load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def clean_answer(text: str) -> str:
    """Strip inline citations and verification block from answer."""
    text = CITATION_PAT.sub("", text)
    text = VERIF_PAT.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def get_judge_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_JUDGE_MODEL", "deepseek/deepseek-v4-flash")
    if not api_key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set.")
    chat_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://github.com/ajujo/raglab-local",
            "X-Title": "RAG-Lab diagnosis",
        },
    )
    return LangchainLLMWrapper(chat_llm)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if not r.get("error")]


def load_query_meta(repo_root: Path) -> dict[str, dict]:
    yaml_path = repo_root / "data" / "benchmark_queries.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    queries = data.get("queries", [])
    meta = {}
    for q in queries:
        qid = q["id"]
        doc_rel = q.get("doc_relevance") or {}
        has_relevant = any(g >= 2 for g in doc_rel.values())
        meta[qid] = {
            "category":   q.get("category", "unknown"),
            "language":   q.get("language", "unknown"),
            "in_corpus":  "in_corpus" if has_relevant else ("out_of_corpus" if not doc_rel else "ambiguous"),
            "n_relevant_docs": sum(1 for g in doc_rel.values() if g >= 2),
            "notes":      q.get("notes", ""),
        }
    return meta


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas(rows: list[dict], answer_key: str, judge, embeddings) -> list[float]:
    """Run answer_relevancy on rows[answer_key], return per-row scores."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy

    metric = answer_relevancy
    metric.llm = judge
    metric.embeddings = embeddings

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer":   [r[answer_key] for r in rows],
        "contexts": [r["contexts"] for r in rows],
    })

    result = evaluate(dataset, metrics=[metric])
    df = result.to_pandas()
    return df["answer_relevancy"].tolist()


def run_faithfulness(rows: list[dict], answer_key: str, judge) -> list[float]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness

    metric = faithfulness
    metric.llm = judge

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer":   [r[answer_key] for r in rows],
        "contexts": [r["contexts"] for r in rows],
    })

    result = evaluate(dataset, metrics=[metric])
    df = result.to_pandas()
    return df["faithfulness"].tolist()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def pct(values: list[float], p: int) -> float:
    return float(np.percentile(values, p))


def stats(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "mean":   round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "std":    round(float(a.std()), 4),
        "min":    round(float(a.min()), 4),
        "max":    round(float(a.max()), 4),
        "p10":    round(pct(values, 10), 4),
        "p25":    round(pct(values, 25), 4),
        "p75":    round(pct(values, 75), 4),
        "p90":    round(pct(values, 90), 4),
    }


def segment_mean(rows_aug: list[dict], key: str, metric: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for r in rows_aug:
        val = str(r.get(key, "unknown"))
        groups.setdefault(val, []).append(r[metric])
    return {k: round(float(np.mean(v)), 4) for k, v in sorted(groups.items())}


def classify_cause(r: dict) -> str:
    """Heuristic classification of poor answer_relevancy for a single row."""
    score = r["answer_relevancy_raw"]
    if r.get("in_corpus") == "out_of_corpus":
        return "out_of_corpus"
    if r["answer_length_chars"] > 2500:
        return "respuesta_demasiado_larga"
    if r["answer_length_chars"] < 300:
        return "respuesta_incompleta"
    if r["n_citations_in_text"] > 8:
        return "contaminacion_citas"
    delta = r.get("relevancy_delta", 0)
    if delta > 0.05:
        return "contaminacion_citas_confirmada"
    if score < 0.60:
        return "respuesta_tangencial_o_mala_recuperacion"
    return "respuesta_demasiado_amplia"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-faithfulness", action="store_true",
                        help="Skip faithfulness re-run (faster, cheaper)")
    args = parser.parse_args()

    load_env()

    repo_root = Path(__file__).parent.parent
    rows = load_jsonl(Path(args.input))
    query_meta = load_query_meta(repo_root)

    print(f"Rows: {len(rows)}")

    # Enrich rows with metadata + derived fields
    for r in rows:
        meta = query_meta.get(r["query_id"], {})
        r["category"]   = meta.get("category", "unknown")
        r["language"]   = meta.get("language", "unknown")
        r["in_corpus"]  = meta.get("in_corpus", "unknown")
        r["n_relevant_docs"] = meta.get("n_relevant_docs", 0)

        r["answer_raw"]   = r["answer"]
        r["answer_clean"] = clean_answer(r["answer"])

        r["answer_length_chars"]       = len(r["answer_raw"])
        r["answer_clean_length_chars"] = len(r["answer_clean"])
        r["citation_chars_removed"]    = r["answer_length_chars"] - r["answer_clean_length_chars"]
        r["citation_pct_of_answer"]    = round(r["citation_chars_removed"] / max(r["answer_length_chars"], 1), 3)

        r["n_citations_in_text"] = len(CITATION_PAT.findall(r["answer_raw"]))
        r["has_verification_block"] = bool(VERIF_PAT.search(r["answer_raw"]))
        r["n_citations_reported"] = len(r.get("citations", []))

        # Length bucket
        cl = r["answer_length_chars"]
        r["length_bucket"] = "short" if cl < 600 else ("medium" if cl < 1500 else "long")

    # Judge LLM + embeddings (shared across both runs)
    judge = get_judge_llm()
    from ragas.embeddings import HuggingfaceEmbeddings
    embeddings = HuggingfaceEmbeddings(model="BAAI/bge-small-en-v1.5")

    # --- Run A: raw answers (with inline citations) ---
    print("\n[A] Running answer_relevancy on RAW answers (with citations)...")
    ar_raw = run_ragas(rows, "answer_raw", judge, embeddings)
    for r, score in zip(rows, ar_raw):
        r["answer_relevancy_raw"] = round(score, 4)

    # --- Run B: clean answers (citations stripped) ---
    print("\n[B] Running answer_relevancy on CLEAN answers (citations stripped)...")
    ar_clean = run_ragas(rows, "answer_clean", judge, embeddings)
    for r, score in zip(rows, ar_clean):
        r["answer_relevancy_clean"] = round(score, 4)
        r["relevancy_delta"] = round(score - r["answer_relevancy_raw"], 4)

    # --- Optional: faithfulness on clean ---
    if not args.skip_faithfulness:
        print("\n[C] Running faithfulness on RAW answers...")
        faith_raw = run_faithfulness(rows, "answer_raw", judge)
        for r, score in zip(rows, faith_raw):
            r["faithfulness_raw"] = round(score, 4)
    else:
        for r in rows:
            r["faithfulness_raw"] = None

    # --- Classify causes ---
    for r in rows:
        r["probable_cause"] = classify_cause(r)

    # --- Statistics ---
    ar_raw_vals   = [r["answer_relevancy_raw"]   for r in rows]
    ar_clean_vals = [r["answer_relevancy_clean"] for r in rows]
    deltas        = [r["relevancy_delta"]         for r in rows]

    global_stats = {
        "raw":   stats(ar_raw_vals),
        "clean": stats(ar_clean_vals),
        "delta": stats(deltas),
    }

    by_category  = segment_mean(rows, "category",      "answer_relevancy_raw")
    by_language  = segment_mean(rows, "language",      "answer_relevancy_raw")
    by_in_corpus = segment_mean(rows, "in_corpus",     "answer_relevancy_raw")
    by_length    = segment_mean(rows, "length_bucket", "answer_relevancy_raw")
    by_has_cits  = {
        str(k): round(float(np.mean([r["answer_relevancy_raw"] for r in rows
                                      if (r["n_citations_in_text"] > 0) == k])), 4)
        for k in [True, False]
    }

    # --- Top/bottom 10 ---
    sorted_rows = sorted(rows, key=lambda r: r["answer_relevancy_raw"])
    worst_10 = [
        {
            "query_id":     r["query_id"],
            "question":     r["question"],
            "category":     r["category"],
            "language":     r["language"],
            "in_corpus":    r["in_corpus"],
            "ar_raw":       r["answer_relevancy_raw"],
            "ar_clean":     r["answer_relevancy_clean"],
            "delta":        r["relevancy_delta"],
            "length_chars": r["answer_length_chars"],
            "n_cit_text":   r["n_citations_in_text"],
            "probable_cause": r["probable_cause"],
        }
        for r in sorted_rows[:10]
    ]
    best_10 = [
        {
            "query_id":     r["query_id"],
            "question":     r["question"],
            "category":     r["category"],
            "language":     r["language"],
            "ar_raw":       r["answer_relevancy_raw"],
            "ar_clean":     r["answer_relevancy_clean"],
            "delta":        r["relevancy_delta"],
            "length_chars": r["answer_length_chars"],
        }
        for r in sorted_rows[-10:][::-1]
    ]

    # --- Per-row table (full) ---
    per_row = [
        {
            "query_id":              r["query_id"],
            "question":              r["question"],
            "category":              r["category"],
            "language":              r["language"],
            "in_corpus":             r["in_corpus"],
            "ar_raw":                r["answer_relevancy_raw"],
            "ar_clean":              r["answer_relevancy_clean"],
            "relevancy_delta":       r["relevancy_delta"],
            "faithfulness":          r.get("faithfulness_raw"),
            "length_chars":          r["answer_length_chars"],
            "clean_length_chars":    r["answer_clean_length_chars"],
            "citation_pct":          r["citation_pct_of_answer"],
            "n_citations_in_text":   r["n_citations_in_text"],
            "n_citations_reported":  r["n_citations_reported"],
            "has_verif_block":       r["has_verification_block"],
            "length_bucket":         r["length_bucket"],
            "trust_score":           r.get("trust_score"),
            "trust_level":           r.get("trust_level"),
            "probable_cause":        r["probable_cause"],
        }
        for r in sorted_rows
    ]

    # --- Cause distribution ---
    from collections import Counter
    cause_dist = dict(Counter(r["probable_cause"] for r in rows).most_common())

    # --- Final report ---
    report = {
        "baseline_file":    str(args.input),
        "n_queries":        len(rows),
        "global_stats":     global_stats,
        "segment_breakdown": {
            "by_category":  by_category,
            "by_language":  by_language,
            "by_in_corpus": by_in_corpus,
            "by_length_bucket": by_length,
            "by_has_citations_in_text": by_has_cits,
        },
        "worst_10":         worst_10,
        "best_10":          best_10,
        "cause_distribution": cause_dist,
        "per_row":          per_row,
        "citation_contamination_summary": {
            "mean_citation_pct_of_answer": round(float(np.mean([r["citation_pct_of_answer"] for r in rows])), 3),
            "mean_delta_clean_minus_raw":  round(float(np.mean(deltas)), 4),
            "queries_where_clean_better":  sum(1 for d in deltas if d > 0.01),
            "queries_where_clean_worse":   sum(1 for d in deltas if d < -0.01),
        },
    }

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("DIAGNOSIS REPORT — answer_relevancy")
    print("=" * 60)
    print(f"\n  Global RAW   mean={global_stats['raw']['mean']}  median={global_stats['raw']['median']}  std={global_stats['raw']['std']}")
    print(f"  Global CLEAN mean={global_stats['clean']['mean']}  median={global_stats['clean']['median']}  std={global_stats['clean']['std']}")
    print(f"  Delta (clean-raw) mean={global_stats['delta']['mean']}  max={global_stats['delta']['max']}")
    print(f"\n  By category:")
    for k, v in by_category.items():
        print(f"    {k:<35} {v:.4f}")
    print(f"\n  By language: {by_language}")
    print(f"  By in_corpus: {by_in_corpus}")
    print(f"  By length:   {by_length}")
    print(f"  By has_citations_in_text: {by_has_cits}")
    print(f"\n  Cause distribution: {cause_dist}")
    print(f"\n  Citation contamination:")
    cc = report["citation_contamination_summary"]
    print(f"    Mean citation % of answer: {cc['mean_citation_pct_of_answer']*100:.1f}%")
    print(f"    Queries where clean > raw: {cc['queries_where_clean_better']}")
    print(f"    Queries where clean < raw: {cc['queries_where_clean_worse']}")
    print(f"\n  WORST 10:")
    for w in worst_10:
        print(f"    {w['query_id']}  ar={w['ar_raw']:.3f}→{w['ar_clean']:.3f}  cause={w['probable_cause']}  [{w['category']}]")
    print(f"\n  BEST 10:")
    for b in best_10:
        print(f"    {b['query_id']}  ar={b['ar_raw']:.3f}→{b['ar_clean']:.3f}  [{b['category']}]")
    print("=" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()
