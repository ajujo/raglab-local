"""CLI sub-app for document ingestion and ingest-run management.

Commands::

    rag-lab ingest [--doc PATH] [--force] [--resume] [--retry-failed]
    rag-lab ingest runs   [--doc DOC_ID] [--status STATUS] [--limit N]
    rag-lab ingest show   RUN_ID
    rag-lab ingest rollback RUN_ID [--force]
    rag-lab ingest retry    RUN_ID [--force]
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

ingest_app = typer.Typer(
    name="ingest",
    help="Ingest documents and manage ingest-run history.",
    invoke_without_command=True,
)


# ---------------------------------------------------------------------------
# Main ingest callback (bare `rag-lab ingest [options]`)
# ---------------------------------------------------------------------------

@ingest_app.callback(invoke_without_command=True)
def ingest_main(
    ctx: typer.Context,
    doc: Optional[str] = typer.Option(
        None, "--doc", help="Path to a single source document."
    ),
    force: bool = typer.Option(
        False, "--force", help="Force re-ingestion even if already ingested."
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Roll back and retry stale IN_PROGRESS runs (crashed ingests).",
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Roll back and retry all FAILED runs."
    ),
    cpu_embedding: bool = typer.Option(
        False, "--cpu-embedding", help="Run embedding on CPU."
    ),
) -> None:
    """Ingest one or more documents: clean, chunk, embed, and store."""
    if ctx.invoked_subcommand is not None:
        return

    from rag_lab.logging_config import setup_logging
    setup_logging("INFO")

    from rag_lab.config import EMBEDDING_DEVICE, SOURCES
    from rag_lab.storage.docstore import DocStore
    from rag_lab.storage.vector_store import VectorStore

    device = "cpu" if cpu_embedding else EMBEDDING_DEVICE

    doc_store = DocStore()
    doc_store.initialize()
    vector_store = VectorStore()
    vector_store.initialize()

    # --- Handle --resume / --retry-failed ---
    if resume or retry_failed:
        _handle_resume_retry(
            doc_store=doc_store,
            vector_store=vector_store,
            device=device,
            resume=resume,
            retry_failed=retry_failed,
        )
        doc_store.close()
        return

    # --- Normal ingest ---
    paths = [Path(doc)] if doc else list(SOURCES)

    total = 0
    for source_path in paths:
        n = _ingest_one(
            source_path=source_path,
            doc_store=doc_store,
            vector_store=vector_store,
            force=force,
            device=device,
        )
        total += n

    doc_store.close()
    console.print(f"[bold green]Total: {total} chunks ingested[/bold green]")


# ---------------------------------------------------------------------------
# `rag-lab ingest runs`
# ---------------------------------------------------------------------------

@ingest_app.command("runs")
def ingest_runs(
    doc_id: Optional[str] = typer.Option(None, "--doc", help="Filter by doc_id."),
    status: Optional[str] = typer.Option(
        None, "--status",
        help="Filter by status (IN_PROGRESS, COMMITTED, FAILED, ROLLED_BACK).",
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum rows to show."),
) -> None:
    """List recent ingest runs."""
    from rag_lab.storage.docstore import DocStore
    from rag_lab.ingest.transaction import IngestRunStore

    ds = DocStore()
    ds.initialize()
    store = IngestRunStore(ds._conn)
    runs = store.list_runs(doc_id=doc_id, status=status, limit=limit)
    ds.close()

    if not runs:
        console.print("[dim]No ingest runs found.[/dim]")
        return

    t = Table(title=f"Ingest runs ({len(runs)})")
    for col in ("run_id", "doc_id", "status", "started_at", "finished_at",
                "chunks_expected", "chunks_written_chroma", "chunks_written_docstore"):
        t.add_column(col, no_wrap=True)

    _status_color = {
        "COMMITTED": "green",
        "FAILED": "red",
        "ROLLED_BACK": "yellow",
        "IN_PROGRESS": "cyan",
        "PENDING": "dim",
    }

    for r in runs:
        color = _status_color.get(r["status"], "white")
        t.add_row(
            r["run_id"],
            r["doc_id"],
            f"[{color}]{r['status']}[/{color}]",
            r["started_at"] or "",
            r["finished_at"] or "",
            str(r["chunks_expected"]),
            str(r["chunks_written_chroma"]),
            str(r["chunks_written_docstore"]),
        )

    console.print(t)


# ---------------------------------------------------------------------------
# `rag-lab ingest show RUN_ID`
# ---------------------------------------------------------------------------

@ingest_app.command("show")
def ingest_show(run_id: str = typer.Argument(..., help="Run ID to display.")) -> None:
    """Show full details for a specific ingest run."""
    from rag_lab.storage.docstore import DocStore
    from rag_lab.ingest.transaction import IngestRunStore

    ds = DocStore()
    ds.initialize()
    run = IngestRunStore(ds._conn).get(run_id)
    ds.close()

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    for k, v in run.items():
        console.print(f"  [bold]{k:<30}[/bold] {v}")


# ---------------------------------------------------------------------------
# `rag-lab ingest rollback RUN_ID`
# ---------------------------------------------------------------------------

@ingest_app.command("rollback")
def ingest_rollback(
    run_id: str = typer.Argument(..., help="Run ID to roll back."),
    force: bool = typer.Option(
        False, "--force",
        help="Roll back even if run is COMMITTED or already ROLLED_BACK.",
    ),
) -> None:
    """Roll back a FAILED or stale IN_PROGRESS run (compensation delete)."""
    from rag_lab.storage.docstore import DocStore
    from rag_lab.ingest.transaction import IngestRunStore, IngestTransaction

    ds = DocStore()
    ds.initialize()
    store = IngestRunStore(ds._conn)
    run = store.get(run_id)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    if run["status"] not in ("FAILED", "IN_PROGRESS") and not force:
        console.print(
            f"[yellow]Run {run_id} has status {run['status']!r}. "
            "Use --force to roll back anyway.[/yellow]"
        )
        raise typer.Exit(1)

    txn = IngestTransaction(run["doc_id"], run["source_path"], ds)
    txn.run_id = run_id
    txn.rollback()
    ds.close()

    console.print(
        f"[green]Rolled back run {run_id} "
        f"(doc_id={run['doc_id']!r})[/green]"
    )


# ---------------------------------------------------------------------------
# `rag-lab ingest retry RUN_ID`
# ---------------------------------------------------------------------------

@ingest_app.command("retry")
def ingest_retry(
    run_id: str = typer.Argument(..., help="Run ID of a FAILED run to retry."),
    force: bool = typer.Option(
        False, "--force",
        help="Retry even if run is not in FAILED status.",
    ),
    cpu_embedding: bool = typer.Option(False, "--cpu-embedding"),
) -> None:
    """Roll back a FAILED run and re-ingest the document from scratch."""
    from rag_lab.config import EMBEDDING_DEVICE
    from rag_lab.storage.docstore import DocStore
    from rag_lab.storage.vector_store import VectorStore
    from rag_lab.ingest.transaction import IngestRunStore, IngestTransaction
    from rag_lab.logging_config import setup_logging

    setup_logging("INFO")
    device = "cpu" if cpu_embedding else EMBEDDING_DEVICE

    ds = DocStore()
    ds.initialize()
    run = IngestRunStore(ds._conn).get(run_id)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        raise typer.Exit(1)

    if run["status"] not in ("FAILED", "IN_PROGRESS") and not force:
        console.print(
            f"[yellow]Run {run_id} has status {run['status']!r}. "
            "Use --force to retry anyway.[/yellow]"
        )
        raise typer.Exit(1)

    if run["status"] in ("FAILED", "IN_PROGRESS"):
        txn = IngestTransaction(run["doc_id"], run["source_path"], ds)
        txn.run_id = run_id
        txn.rollback()

    source_path = Path(run["source_path"]) if run["source_path"] else None
    if not source_path or not source_path.exists():
        console.print(
            f"[red]Source file not found: {run['source_path']!r}. "
            "Cannot retry.[/red]"
        )
        raise typer.Exit(1)

    vs = VectorStore()
    vs.initialize()
    _ingest_one(
        source_path=source_path,
        doc_store=ds,
        vector_store=vs,
        force=True,
        device=device,
    )
    ds.close()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ingest_one(
    source_path: Path,
    doc_store,
    vector_store,
    force: bool,
    device: str,
) -> int:
    """Ingest one document, wrapped in an IngestTransaction.

    Returns:
        Number of chunks ingested (0 if skipped).
    """
    import numpy as np
    from rag_lab.config import (
        CHUNK_MAX_TOKENS,
        CHUNK_OVERLAP,
        EMBEDDING_BATCH_SIZE,
        EMBEDDING_DIM,
        EMBEDDING_MODEL,
        EMBEDDING_MODEL_VERSION,
        SPARSE_FORMAT_VERSION,
    )
    from rag_lab.ingest.cleaner import clean_document
    from rag_lab.ingest.manifest import create_manifest
    from rag_lab.chunking.splitter import chunk_document
    from rag_lab.embedding.encoder import encode_chunks
    from rag_lab.ingest.transaction import IngestTransaction
    from rag_lab.storage.metadata_store import MetadataStore

    logger = logging.getLogger("rag_lab")

    if not source_path.exists():
        logger.warning(f"Source file not found: {source_path} — skipping")
        return 0

    console.print(f"[bold cyan]Ingesting: {source_path.name}[/bold cyan]")

    cleaned_path = clean_document(source_path)
    create_manifest(source_path, cleaned_path, force=force)

    cleaned_text = cleaned_path.read_text(encoding="utf-8")
    chunks = chunk_document(
        cleaned_text,
        doc_id=source_path.stem,
        max_tokens=CHUNK_MAX_TOKENS,
        overlap=CHUNK_OVERLAP,
    )
    logger.info(f"Created {len(chunks)} chunks from {source_path.name}")

    chunk_dicts = [c.to_dict() for c in chunks]

    dense_embeddings, sparse_embeddings = encode_chunks(
        chunk_dicts,
        batch_size=EMBEDDING_BATCH_SIZE,
        device=device,
    )

    for chunk_d in chunk_dicts:
        sparse = sparse_embeddings.get(chunk_d["chunk_id"], {})
        if sparse:
            tokens_arr = np.array(list(sparse.keys()), dtype=np.int32)
            weights_arr = np.array(list(sparse.values()), dtype=np.float32)
            chunk_d["sparse_tokens"] = tokens_arr.tobytes()
            chunk_d["sparse_weights"] = weights_arr.tobytes()
        else:
            chunk_d["sparse_tokens"] = None
            chunk_d["sparse_weights"] = None
        chunk_d["embedding_model_name"] = EMBEDDING_MODEL
        chunk_d["embedding_model_version"] = EMBEDDING_MODEL_VERSION
        chunk_d["embedding_dim"] = EMBEDDING_DIM
        chunk_d["sparse_format_version"] = SPARSE_FORMAT_VERSION

    doc_id = source_path.stem
    n_sparse = sum(1 for c in chunk_dicts if c.get("sparse_tokens") is not None)

    with IngestTransaction(doc_id, source_path, doc_store) as txn:
        txn.update(chunks_expected=len(chunk_dicts))

        vector_store.add(
            ids=[c.get("chunk_id", "") for c in chunk_dicts],
            embeddings=dense_embeddings,
            documents=[c.get("text", "") for c in chunk_dicts],
            metadatas=[
                {"heading_path": c.get("heading_path", ""), "doc_id": c.get("doc_id", "")}
                for c in chunk_dicts
            ],
        )
        txn.update(chunks_written_chroma=len(chunk_dicts))

        doc_store.add(chunk_dicts)
        txn.update(
            chunks_written_docstore=len(chunk_dicts),
            chunks_written_fts5=len(chunk_dicts),
            chunks_written_sparse=n_sparse,
        )

        MetadataStore(conn=doc_store._conn).upsert_document(
            doc_id,
            path=str(source_path),
        )
        txn.update(metadata_written=1)

    console.print(
        f"[bold green]Ingested {len(chunk_dicts)} chunks from {source_path.name}[/bold green]"
    )
    return len(chunk_dicts)


def _handle_resume_retry(
    *,
    doc_store,
    vector_store,
    device: str,
    resume: bool,
    retry_failed: bool,
) -> None:
    """Roll back stale/failed runs and re-ingest their documents."""
    from rag_lab.ingest.transaction import IngestRunStore, IngestTransaction

    store = IngestRunStore(doc_store._conn)
    targets = []

    if resume:
        stale = store.get_stale_in_progress()
        if stale:
            console.print(
                f"[yellow]Found {len(stale)} stale IN_PROGRESS run(s) — rolling back...[/yellow]"
            )
        targets.extend(stale)

    if retry_failed:
        failed = store.get_failed()
        if failed:
            console.print(
                f"[yellow]Found {len(failed)} FAILED run(s) — rolling back...[/yellow]"
            )
        targets.extend(failed)

    if not targets:
        console.print("[green]Nothing to resume or retry.[/green]")
        return

    for run in targets:
        txn = IngestTransaction(run["doc_id"], run["source_path"], doc_store)
        txn.run_id = run["run_id"]
        txn.rollback()

        source_path = Path(run["source_path"]) if run["source_path"] else None
        if not source_path or not source_path.exists():
            console.print(
                f"[red]Source file missing for run {run['run_id']}: "
                f"{run['source_path']!r} — skipped[/red]"
            )
            continue

        console.print(
            f"[cyan]Re-ingesting {run['doc_id']!r} "
            f"(previous run={run['run_id']})[/cyan]"
        )
        _ingest_one(
            source_path=source_path,
            doc_store=doc_store,
            vector_store=vector_store,
            force=True,
            device=device,
        )
