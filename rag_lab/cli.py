"""CLI interface for the RAG-Lab system.

Provides commands for ingesting documents and querying the system.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from rag_lab.cli_chat import run_chat
from rag_lab.config import (
    DATA_DIR,
    EMBEDDING_DEVICE,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    STORAGE_DIR,
)
from rag_lab.exceptions import RAGLabError, RetrievalError, LLMConnectionError
from rag_lab.embedding.encoder import encode_chunks, load_embedding_model
from rag_lab.storage.vector_store import VectorStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.docstore import DocStore
from rag_lab.retrieval.query_processor import process_query
from rag_lab.retrieval.hybrid_search import hybrid_search
from rag_lab.retrieval.reranker import rerank
from rag_lab.generation.prompt_builder import build_prompt
from rag_lab.generation.llm_client import generate_response
from rag_lab.verification.pipeline import verify_and_score
from rag_lab.verification.scoring import ConfidenceLevel
from rag_lab.feedback.feedback_store import (
    FeedbackEntry,
    save_feedback,
    init_db,
)
from rag_lab.performance.timer import PhaseTimer
from rag_lab.performance.report import generate_report, save_report_json
from rag_lab.logging_config import setup_logging
from rag_lab.cli_docs import docs_app, tags_app
from rag_lab.cli_ingest import ingest_app

app = typer.Typer(
    name="rag-lab",
    help="RAG system for SDMX Technical Notes",
    add_completion=True,
)
app.add_typer(ingest_app, name="ingest")
app.add_typer(docs_app, name="docs")
app.add_typer(tags_app, name="tags")

cache_app = typer.Typer(name="cache", help="Query cache management.")
app.add_typer(cache_app, name="cache")

feedback_app = typer.Typer(name="feedback", help="Chunk-level feedback management.")
app.add_typer(feedback_app, name="feedback")

console = Console()


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask."),
    hyde: bool = typer.Option(False, "--hyde", help="Enable HyDE."),
    rewrite: bool = typer.Option(False, "--rewrite", help="Enable query rewriting."),
    fast: bool = typer.Option(False, "--fast", help="Skip reranking."),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve."),
    no_feedback: bool = typer.Option(False, "--no-feedback", help="Disable feedback prompt."),
    profile: bool = typer.Option(False, "--profile", help="Show performance metrics."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass query cache for this run."),
    cpu_embedding: bool = typer.Option(
        False,
        "--cpu-embedding",
        help="Run embedding model on CPU to free GPU VRAM.",
    ),
    cpu_reranker: bool = typer.Option(
        False,
        "--cpu-reranker",
        help="Run reranker on CPU to free GPU VRAM.",
    ),
) -> None:
    """Query the RAG system with a natural language question."""
    from rag_lab.config import QUERY_CACHE_ENABLED
    from rag_lab.cache.query_cache import make_cache_key, get_corpus_fingerprint, get_cache

    setup_logging("INFO")
    logger = logging.getLogger("rag_lab")
    console = Console()

    console.print(f"[bold cyan]❓ Query:[/bold cyan] {question}")

    # Determine devices
    emb_device = "cpu" if cpu_embedding else EMBEDDING_DEVICE
    rerank_device = "cpu" if cpu_reranker else os.getenv("RERANKER_DEVICE", "cuda")

    # Initialize stores early — needed for corpus fingerprint
    vector_store = VectorStore()
    fts_store = FTSStore()
    doc_store = DocStore()
    vector_store.initialize()
    fts_store.initialize()
    doc_store.initialize()

    # Initialize timer for profiling
    timer = PhaseTimer()

    # ------------------------------------------------------------------
    # Cache lookup — skip embedding + retrieval on hit
    # ------------------------------------------------------------------
    use_cache = QUERY_CACHE_ENABLED and not no_cache
    cache_hit = False
    cache_key = None
    corpus_fp = None
    unique_results = []

    if use_cache:
        corpus_fp = get_corpus_fingerprint(doc_store._conn)
        cache_key = make_cache_key(
            question,
            top_k=top_k * 2,
            corpus_fingerprint=corpus_fp,
        )
        _cache = get_cache()
        cached = _cache.get(cache_key, corpus_fp)
        if cached is not None:
            unique_results = cached
            cache_hit = True
            console.print(
                f"[dim]cache hit · key={cache_key[:12]}… · "
                f"corpus={corpus_fp}[/dim]"
            )

    if not cache_hit:
        # Process query
        if profile:
            timer.start("query_processing")
        queries = process_query(question, use_hyde=hyde, use_rewriting=rewrite)
        if profile:
            timer.stop()

        # Get embeddings for all query variants
        all_query_data = []
        if profile:
            timer.start("embedding")
        for q in queries:
            dense_emb, sparse_dict = encode_chunks(
                [{"text": q["text"]}], batch_size=1, device=emb_device
            )
            query_dense = dense_emb[0]
            if q.get("use_for_sparse", True):
                query_sparse = next(iter(sparse_dict.values()), {}) if sparse_dict else {}
            else:
                query_sparse = {}
            all_query_data.append((query_dense, query_sparse))
        if profile:
            timer.stop()

        # Search with each query variant
        all_results = []
        if profile:
            timer.start("hybrid_search")
        for query_dense, query_sparse in all_query_data:
            results = hybrid_search(
                question,
                vector_store,
                doc_store,
                fts_store,
                query_dense=query_dense,
                query_sparse=query_sparse,
                top_k=top_k * 2,
            )
            all_results.extend(results)
        if profile:
            timer.stop()

        # Deduplicate by chunk_id
        seen: set = set()
        for r in all_results:
            if r.get("chunk_id") not in seen:
                seen.add(r.get("chunk_id"))
                unique_results.append(r)

        # Rerank if not in fast mode
        if not fast and unique_results:
            if profile:
                timer.start("reranking")
            unique_results = rerank(
                question,
                unique_results[:20],
                top_k=min(top_k * 2, len(unique_results)),
                device=rerank_device,
            )
            if profile:
                timer.stop()

        # Store result in cache
        if use_cache and unique_results:
            norm_q = " ".join(question.strip().lower().split())
            get_cache().set(cache_key, corpus_fp, unique_results, query_norm=norm_q)

    # Generate response
    if unique_results:
        system_prompt, user_prompt = build_prompt(question, unique_results[:RERANK_TOP_K])
        try:
            if profile:
                timer.start("llm_generation")
            response = generate_response(system_prompt, user_prompt)
            if profile:
                timer.stop()
            if not response:
                console.print("\n[bold yellow]⚠️ El LLM no devolvió respuesta.[/bold yellow]")
            else:
                # Use rerank_score if available, fallback to retrieval score
                retrieval_scores = [r.get("rerank_score", r.get("score", 0.5)) for r in unique_results[:RERANK_TOP_K]]

                # Run verification pipeline
                from rag_lab.config import ENABLE_CONSISTENCY_CHECK
                try:
                    if profile:
                        timer.start("verification")
                    verification = verify_and_score(
                        response,
                        unique_results[:RERANK_TOP_K],
                        retrieval_scores,
                        enable_consistency_check=ENABLE_CONSISTENCY_CHECK,
                    )
                    if profile:
                        timer.stop()

                    # Print response with verification block
                    console.print(f"\n[bold green]🤖 Response:[/bold green]\n{verification.response}")

                    # Print warnings if any
                    warnings = verification.get_warnings()
                    for warning in warnings:
                        console.print(f"[bold yellow]⚠️ {warning}[/bold yellow]")

                    # Print verification block
                    console.print(f"\n{verification.format_verification_block()}")

                    # Print chunk feedback hints
                    _print_feedback_hints(question, unique_results[:RERANK_TOP_K], cache_hit, cache_key)

                    # Profile report
                    if profile:
                        console.print(f"\n{generate_report(timer.get_all_durations())}")
                        save_report_json(timer.get_all_durations())

                    # Feedback prompt
                    if not no_feedback:
                        _collect_feedback(
                            question=question,
                            rewritten_query=None,
                            hyde_used=hyde,
                            chunks=unique_results[:RERANK_TOP_K],
                            final_score=verification.score_result.final_score,
                            score_level=verification.score_result.confidence_level.value,
                        )
                except Exception as e:
                    if profile and timer._current_phase is not None:
                        timer.stop()
                    console.print(f"\n[bold green]🤖 Response:[/bold green]\n{response}")
                    console.print(f"\n[bold yellow]⚠️ Error en la capa de verificación: {e}[/bold yellow]")

        except LLMConnectionError as e:
            console.print(f"[bold red]LLM Error:[/bold red] {e}")
        except RAGLabError as e:
            console.print(f"[bold red]RAG-Lab Error:[/bold red] {e}")
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
    else:
        console.print("[bold yellow]⚠️ No results found.[/bold yellow]")


@app.command()
def chat(
    cpu_embedding: bool = typer.Option(
        False,
        "--cpu-embedding",
        help="Run embedding model on CPU to free GPU VRAM.",
    ),
    cpu_reranker: bool = typer.Option(
        False,
        "--cpu-reranker",
        help="Run reranker model on CPU to free GPU VRAM.",
    ),
    profile: bool = typer.Option(
        False,
        "--profile",
        help="Show performance metrics per query.",
    ),
) -> None:
    """Start an interactive chat session with document filtering."""
    setup_logging("INFO")
    run_chat(cpu_embedding=cpu_embedding, cpu_reranker=cpu_reranker, profile=profile)


# =============================================================================
# Cache management sub-commands
# =============================================================================

@cache_app.command("stats")
def cache_stats() -> None:
    """Show query cache statistics."""
    from rag_lab.cache.query_cache import get_cache
    setup_logging("WARNING")
    s = get_cache().stats()
    console.print("[bold]Query Cache Statistics[/bold]")
    console.print(f"  enabled:        {s['enabled']}")
    console.print(f"  total entries:  {s['total_entries']}")
    console.print(f"  total hits:     {s['total_hits']}")
    console.print(f"  db size:        {s['db_size_bytes'] / 1024:.1f} KB")
    if s["oldest_entry_age_s"]:
        console.print(f"  oldest entry:   {s['oldest_entry_age_s'] // 3600}h ago")
    if s["latest_access_age_s"] and s["total_entries"]:
        console.print(f"  last accessed:  {s['latest_access_age_s'] // 60}m ago")
    ttl = s["ttl_seconds"]
    console.print(f"  TTL:            {'none' if ttl == 0 else f'{ttl // 86400}d'}")


@cache_app.command("clear")
def cache_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Remove all query cache entries."""
    from rag_lab.cache.query_cache import get_cache
    setup_logging("WARNING")
    if not yes:
        typer.confirm("Clear all cache entries?", abort=True)
    n = get_cache().clear()
    console.print(f"[bold green]Cache cleared — {n} entries removed.[/bold green]")


@cache_app.command("vacuum")
def cache_vacuum() -> None:
    """Remove expired entries and compact the cache database."""
    from rag_lab.cache.query_cache import get_cache
    setup_logging("WARNING")
    get_cache().vacuum()
    console.print("[bold green]Cache vacuumed.[/bold green]")


@cache_app.command("inspect")
def cache_inspect(
    cache_key: str = typer.Argument(..., help="Full or partial cache key to inspect."),
) -> None:
    """Show metadata for a specific cache entry."""
    from rag_lab.cache.query_cache import get_cache
    setup_logging("WARNING")
    info = get_cache().inspect(cache_key)
    if info is None:
        console.print(f"[bold yellow]No entry found for key: {cache_key}[/bold yellow]")
        raise typer.Exit(1)
    console.print(f"  cache_key:   {info['cache_key'][:32]}…")
    console.print(f"  corpus_fp:   {info['corpus_fp']}")
    console.print(f"  query:       {info['query_norm']}")
    console.print(f"  age:         {info['age_s'] // 60}m")
    console.print(f"  hits:        {info['hit_count']}")


# =============================================================================
# Feedback sub-commands
# =============================================================================

@feedback_app.command("add")
def feedback_add(
    query: str = typer.Option(..., "--query", "-q", help="Query text."),
    chunk_id: str = typer.Option(..., "--chunk-id", "-c", help="Chunk ID."),
    feedback: str = typer.Option(..., "--feedback", "-f",
        help=f"Feedback type: {', '.join(sorted(__import__('rag_lab.feedback.store', fromlist=['VALID_FEEDBACK']).VALID_FEEDBACK))}"),
    doc_id: str = typer.Option("", "--doc-id", help="Document ID (optional if chunk-id is unique)."),
    rank: int = typer.Option(None, "--rank", help="Rank position of chunk in results."),
    rating: int = typer.Option(None, "--rating", help="Numeric rating 1-5 (optional)."),
    reason: str = typer.Option(None, "--reason", help="Short reason (optional)."),
    note: str = typer.Option(None, "--note", help="Free-text note (optional)."),
) -> None:
    """Record chunk-level feedback for a retrieval result."""
    from rag_lab.feedback.store import FeedbackStore, VALID_FEEDBACK
    setup_logging("WARNING")
    if feedback not in VALID_FEEDBACK:
        console.print(f"[bold red]Invalid feedback {feedback!r}. Valid: {sorted(VALID_FEEDBACK)}[/bold red]")
        raise typer.Exit(1)
    store = FeedbackStore()
    store.initialize()
    row_id = store.add(
        query, chunk_id=chunk_id, doc_id=doc_id, rank=rank,
        feedback=feedback, rating=rating, reason=reason, user_note=note,
    )
    store.close()
    console.print(f"[bold green]Feedback recorded (id={row_id}).[/bold green]")


@feedback_app.command("list")
def feedback_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show."),
    query_hash: str = typer.Option(None, "--query-hash", help="Filter by query hash."),
    chunk_id: str = typer.Option(None, "--chunk-id", help="Filter by chunk ID."),
    feedback_type: str = typer.Option(None, "--feedback", help="Filter by feedback type."),
) -> None:
    """List recent feedback events."""
    from rag_lab.feedback.store import FeedbackStore
    setup_logging("WARNING")
    store = FeedbackStore()
    store.initialize()
    rows = store.list(
        query_hash=query_hash,
        chunk_id=chunk_id,
        feedback=feedback_type,
        limit=limit,
    )
    store.close()
    if not rows:
        console.print("[dim]No feedback events found.[/dim]")
        return
    console.print(f"[bold]Feedback events[/bold] (showing {len(rows)})")
    for r in rows:
        chunk_short = r["chunk_id"][:16] + "…" if len(r["chunk_id"]) > 16 else r["chunk_id"]
        console.print(
            f"  [{r['id']}] {r['created_at'][:16]}  rank={r['rank'] or '-':>3}  "
            f"{r['feedback']:<12}  {chunk_short:<18}  {r['doc_id']}"
        )


@feedback_app.command("stats")
def feedback_stats() -> None:
    """Show feedback aggregate statistics."""
    from rag_lab.feedback.store import FeedbackStore
    setup_logging("WARNING")
    store = FeedbackStore()
    store.initialize()
    s = store.stats()
    store.close()
    console.print("[bold]Feedback Statistics[/bold]")
    console.print(f"  total events:    {s['total_events']}")
    console.print(f"  unique queries:  {s['unique_queries']}")
    console.print(f"  unique chunks:   {s['unique_chunks']}")
    if s["by_feedback_type"]:
        console.print("  by type:")
        for ftype, count in sorted(s["by_feedback_type"].items()):
            console.print(f"    {ftype:<16} {count}")


@feedback_app.command("export")
def feedback_export(
    output: str = typer.Option(None, "--output", "-o", help="Output file path (default: stdout)."),
    fmt: str = typer.Option("jsonl", "--format", help="Export format (only 'jsonl' supported)."),
) -> None:
    """Export all feedback events to JSONL."""
    from rag_lab.feedback.store import FeedbackStore
    setup_logging("WARNING")
    if fmt != "jsonl":
        console.print(f"[bold red]Unsupported format {fmt!r}. Only 'jsonl' is supported.[/bold red]")
        raise typer.Exit(1)
    store = FeedbackStore()
    store.initialize()
    out_path = Path(output) if output else None
    jsonl = store.export_jsonl(path=out_path)
    store.close()
    if output:
        console.print(f"[bold green]Exported to {output}[/bold green]")
    else:
        console.print(jsonl)


@feedback_app.command("clear")
def feedback_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete all feedback events."""
    from rag_lab.feedback.store import FeedbackStore
    setup_logging("WARNING")
    if not yes:
        typer.confirm("Delete all feedback events?", abort=True)
    store = FeedbackStore()
    store.initialize()
    n = store.clear()
    store.close()
    console.print(f"[bold green]Cleared — {n} events deleted.[/bold green]")


# =============================================================================
# Operational commands — thin pass-through wrappers for argparse-based tools
# =============================================================================

_OPS_CTX = {"allow_extra_args": True, "ignore_unknown_options": True}


@app.command("doctor", context_settings=_OPS_CTX, add_help_option=False)
def cmd_doctor(ctx: typer.Context) -> None:
    """System health checks (config, stores, reconcile, test query)."""
    from rag_lab.doctor import main as _main
    raise typer.Exit(_main(ctx.args))


@app.command("benchmark", context_settings=_OPS_CTX, add_help_option=False)
def cmd_benchmark(ctx: typer.Context) -> None:
    """Retrieval benchmark — compare pipeline variants.

    Examples:
      rag-lab benchmark --suite official --variants full --no-cache
      rag-lab benchmark run --suite official --variants full --no-cache
    """
    args = list(ctx.args)
    # Accept 'run' as a compatibility sub-command alias — strip before forwarding.
    if args and args[0] == "run":
        args = args[1:]
    from rag_lab.benchmark.__main__ import main as _main
    raise typer.Exit(_main(args))


@app.command("reconcile", context_settings=_OPS_CTX, add_help_option=False)
def cmd_reconcile(ctx: typer.Context) -> None:
    """Cross-store consistency check (DocStore vs ChromaDB vs FTS5)."""
    from rag_lab.maintenance.reconcile import main as _main
    raise typer.Exit(_main(ctx.args))


@app.command("diagnose", context_settings=_OPS_CTX, add_help_option=False)
def cmd_diagnose(ctx: typer.Context) -> None:
    """Full system diagnostic with optional test query."""
    from rag_lab.maintenance.diagnose import main as _main
    raise typer.Exit(_main(ctx.args))


if __name__ == "__main__":
    app()


def _print_feedback_hints(
    question: str,
    chunks: list[dict],
    cache_hit: bool,
    cache_key: str | None,
) -> None:
    """Print a compact table of top chunks with feedback command hints."""
    if not chunks:
        return
    console.print("\n[dim]── Retrieved chunks (for feedback) ──[/dim]")
    for i, c in enumerate(chunks, 1):
        cid = c.get("chunk_id", "")
        cid_short = cid[:20] + "…" if len(cid) > 20 else cid
        score = c.get("rerank_score", c.get("rrf_score", 0.0))
        doc = c.get("doc_id", "")
        console.print(
            f"  [dim]{i:>2}. chunk={cid_short}  doc={doc}  score={score:.3f}[/dim]"
        )
    # Show one example command for the top chunk
    top = chunks[0]
    top_cid = top.get("chunk_id", "")
    q_escaped = question.replace('"', '\\"')
    console.print(
        f'\n[dim]  To give feedback: rag-lab feedback add --query "{q_escaped}" '
        f'--chunk-id "{top_cid}" --feedback relevant[/dim]'
    )


def _collect_feedback(
    question: str,
    rewritten_query: str | None,
    hyde_used: bool,
    chunks: list[dict],
    final_score: float,
    score_level: str,
) -> None:
    """Prompt the user for feedback and save to SQLite."""
    init_db()

    # Build chunk metadata (no full text)
    chunk_metas = []
    for c in chunks:
        chunk_metas.append({
            "doc_id": c.get("doc_id", ""),
            "heading_path": c.get("heading_path", ""),
            "line_start": c.get("line_start", 0),
            "line_end": c.get("line_end", 0),
            "retrieval_score": c.get("rerank_score", c.get("score", 0.5)),
        })

    console.print("\n¿Esta respuesta fue útil? [s/n] (Enter para omitir): ", end="")
    try:
        answer = sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if answer in ("s", "sí", "si", "yes", "y"):
        entry = FeedbackEntry(
            question=question,
            rewritten_query=rewritten_query,
            hyde_used=hyde_used,
            chunks_retrieved=json.dumps(chunk_metas, ensure_ascii=False),
            final_score=final_score,
            score_level=score_level,
            useful=True,
            timestamp=datetime.now().isoformat(),
        )
        save_feedback(entry)
        console.print("[bold green]✅ Feedback guardado: Útil[/bold green]")
    elif answer in ("n", "no"):
        entry = FeedbackEntry(
            question=question,
            rewritten_query=rewritten_query,
            hyde_used=hyde_used,
            chunks_retrieved=json.dumps(chunk_metas, ensure_ascii=False),
            final_score=final_score,
            score_level=score_level,
            useful=False,
            timestamp=datetime.now().isoformat(),
        )
        save_feedback(entry)
        console.print("[bold yellow]⚠️ Feedback guardado: No útil[/bold yellow]")
    else:
        # Enter or unrecognized — skip
        pass