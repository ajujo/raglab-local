"""CLI subcommands for E2E evaluation.

rag-lab eval run   — execute the pipeline on a query suite, write JSONL
rag-lab eval list  — list previous eval runs
rag-lab eval show  — print a summary of a specific run
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag_lab.evaluation.config import EVAL_DEFAULT_SUITE, EVAL_OUTPUT_DIR
from rag_lab.logging_config import setup_logging

eval_app = typer.Typer(name="eval", help="E2E evaluation commands.")
console = Console()


@eval_app.command("run")
def eval_run(
    suite: str = typer.Option(EVAL_DEFAULT_SUITE, "--suite", help="Query suite to evaluate."),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output JSONL path. Defaults to data/eval_runs/<suite>_<timestamp>.jsonl",
    ),
    limit: int = typer.Option(None, "--limit", "-n", help="Evaluate only the first N queries."),
    queries: str = typer.Option(
        None, "--queries",
        help="Comma-separated query IDs to evaluate (e.g. q001,q002).",
    ),
    no_cache: bool = typer.Option(True, "--no-cache/--cache", help="Bypass query cache (default: on)."),
    top_k: int = typer.Option(50, "--top-k", help="Retrieval pool size before reranking."),
    rerank_top_k: int = typer.Option(8, "--rerank-top-k", help="Top-K chunks passed to the LLM."),
    temperature: float = typer.Option(0.0, "--temperature", help="LLM temperature."),
) -> None:
    """Run the full pipeline on a query suite and write results to JSONL."""
    from rag_lab.evaluation.dataset import load_eval_samples, EvaluationError
    from rag_lab.evaluation.e2e_runner import run_eval

    setup_logging("INFO")

    # Load samples
    try:
        samples = load_eval_samples(suite=suite)
    except (FileNotFoundError, EvaluationError) as exc:
        console.print(f"[bold red]Error loading queries:[/bold red] {exc}")
        raise typer.Exit(1)

    # Filter by explicit query IDs
    if queries:
        ids = {q.strip() for q in queries.split(",")}
        samples = [s for s in samples if s.query_id in ids]
        if not samples:
            console.print(f"[bold red]No queries matched: {queries}[/bold red]")
            raise typer.Exit(1)

    # Apply limit
    if limit is not None:
        samples = samples[:limit]

    if not samples:
        console.print("[bold yellow]No samples to evaluate.[/bold yellow]")
        raise typer.Exit(0)

    # Resolve output path
    if output:
        out_path = Path(output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EVAL_OUTPUT_DIR / f"{suite}_{ts}.jsonl"

    console.print(
        f"[bold cyan]Evaluating[/bold cyan] {len(samples)} queries "
        f"(suite={suite}, top_k={top_k}, rerank_top_k={rerank_top_k}, T={temperature})"
    )
    console.print(f"[dim]Output → {out_path}[/dim]")

    run_eval(
        samples=samples,
        output_path=out_path,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        temperature=temperature,
    )

    # Summary
    n_errors = 0
    with open(out_path, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    n_errors = sum(1 for l in lines if l.get("error"))
    avg_score = (
        sum(l["trust_score"] for l in lines if not l.get("error")) / max(1, len(lines) - n_errors)
    )
    avg_latency = sum(l["latency_ms"] for l in lines) // max(1, len(lines))

    console.print(f"\n[bold green]Done.[/bold green] {len(lines)} results written.")
    console.print(f"  Errors         : {n_errors}")
    console.print(f"  Avg trust score: {avg_score:.3f}")
    console.print(f"  Avg latency    : {avg_latency} ms")
    console.print(f"  File           : {out_path}")


@eval_app.command("list")
def eval_list(
    n: int = typer.Option(10, "--limit", "-n", help="Max runs to show."),
) -> None:
    """List previous eval runs in data/eval_runs/."""
    setup_logging("WARNING")
    run_dir = EVAL_OUTPUT_DIR
    if not run_dir.exists():
        console.print("[dim]No eval runs yet.[/dim]")
        return

    files = sorted(run_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        console.print("[dim]No eval runs yet.[/dim]")
        return

    table = Table(title="Eval Runs")
    table.add_column("File", style="cyan")
    table.add_column("Queries", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Avg score", justify="right")
    table.add_column("Modified")

    for f in files[:n]:
        try:
            with open(f, encoding="utf-8") as fh:
                lines = [json.loads(l) for l in fh if l.strip()]
            n_errs = sum(1 for l in lines if l.get("error"))
            scores = [l["trust_score"] for l in lines if not l.get("error")]
            avg_s = f"{sum(scores)/len(scores):.3f}" if scores else "—"
        except Exception:
            lines, n_errs, avg_s = [], "?", "?"

        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        table.add_row(f.name, str(len(lines)), str(n_errs), str(avg_s), mtime)

    console.print(table)


@eval_app.command("show")
def eval_show(
    run_id: str = typer.Argument(..., help="Run file name (or prefix) from data/eval_runs/."),
) -> None:
    """Print a detailed summary of an eval run."""
    setup_logging("WARNING")
    run_dir = EVAL_OUTPUT_DIR

    # Accept full name or prefix
    candidates = list(run_dir.glob(f"{run_id}*")) if run_dir.exists() else []
    if not candidates:
        console.print(f"[bold red]No run found matching '{run_id}'[/bold red]")
        raise typer.Exit(1)
    f = sorted(candidates)[-1]

    with open(f, encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]

    if not lines:
        console.print("[bold yellow]File is empty.[/bold yellow]")
        return

    errors = [l for l in lines if l.get("error")]
    ok = [l for l in lines if not l.get("error")]
    scores = [l["trust_score"] for l in ok]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_latency = sum(l["latency_ms"] for l in lines) // len(lines)

    console.print(f"[bold]Run:[/bold] {f.name}")
    console.print(f"  Queries    : {len(lines)}")
    console.print(f"  Errors     : {len(errors)}")
    console.print(f"  Avg score  : {avg_score:.3f}")
    console.print(f"  Avg latency: {avg_latency} ms")

    by_level: dict[str, int] = {}
    for l in ok:
        by_level[l["trust_level"]] = by_level.get(l["trust_level"], 0) + 1
    if by_level:
        console.print("  Trust levels:")
        for lvl in ("HIGH", "MEDIUM", "LOW"):
            if lvl in by_level:
                console.print(f"    {lvl:<8} {by_level[lvl]}")

    if errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for e in errors:
            console.print(f"  [{e['query_id']}] {e['error']}")
