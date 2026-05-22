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
    setup_logging("INFO")
    logger = logging.getLogger("rag_lab")
    console = Console()

    console.print(f"[bold cyan]❓ Query:[/bold cyan] {question}")

    # Determine devices
    emb_device = "cpu" if cpu_embedding else EMBEDDING_DEVICE
    rerank_device = "cpu" if cpu_reranker else os.getenv("RERANKER_DEVICE", "cuda")

    # Initialize timer for profiling
    timer = PhaseTimer()

    # Process query
    if profile:
        timer.start("query_processing")
    queries = process_query(question, use_hyde=hyde, use_rewriting=rewrite)
    if profile:
        timer.stop()

    # Get embeddings for all query variants
    # encode_chunks returns (dense_embeddings: np.ndarray, sparse_embeddings: Dict)
    all_query_data = []
    if profile:
        timer.start("embedding")
    for q in queries:
        dense_emb, sparse_dict = encode_chunks([{"text": q["text"]}], batch_size=1, device=emb_device)
        query_dense = dense_emb[0]
        # HyDE queries should not use generated text for sparse scoring
        if q.get("use_for_sparse", True):
            query_sparse = next(iter(sparse_dict.values()), {}) if sparse_dict else {}
        else:
            query_sparse = {}  # suppress sparse signal for this query variant
        all_query_data.append((query_dense, query_sparse))
    if profile:
        timer.stop()

    # Perform hybrid search
    vector_store = VectorStore()
    fts_store = FTSStore()
    doc_store = DocStore()
    vector_store.initialize()
    fts_store.initialize()
    doc_store.initialize()

    # Search with each query variant
    all_results = []
    if profile:
        timer.start("hybrid_search")
    for query_dense, query_sparse in all_query_data:
        results = hybrid_search(
            question,      # always original question for BM25
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
    seen = set()
    unique_results = []
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


if __name__ == "__main__":
    app()


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