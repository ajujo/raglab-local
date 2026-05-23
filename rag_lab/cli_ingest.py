"""CLI sub-app for document ingestion and ingest-run management.

Commands::

    rag-lab ingest [--doc PATH|DIR] [--force] [--resume] [--retry-failed]
                   [--workers N] [--strict] [--cpu-embedding]
    rag-lab ingest runs     [--doc DOC_ID] [--status STATUS] [--limit N]
    rag-lab ingest batches  [--limit N]
    rag-lab ingest show     RUN_ID
    rag-lab ingest rollback RUN_ID [--force]
    rag-lab ingest retry    RUN_ID [--force]

Pipeline (v1.16+)
-----------------
  1. Parallel preparation (--workers N):
     validate + clean + chunk each document independently (CPU-bound, no DB writes).
  2. Status update + SKIPPED detection (main thread, single writer):
     Check content_hash against previous COMMITTED rows; mark SKIPPED if unchanged.
  3. Sequential embedding:
     Batch-embed all validated docs (GPU/CPU model is not thread-safe).
  4. Sequential write (single-writer guarantee):
     IngestTransaction per document: ChromaDB → SQLite chunks → FTS5 → metadata.

Single-writer guarantee
-----------------------
SQLite and ChromaDB receive writes only from the main thread.  Worker threads
compute but never touch any store.  This prevents WAL contention and ensures
reconcile is always clean after a run.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
# Preparation result (returned by worker threads, no DB state)
# ---------------------------------------------------------------------------

@dataclass
class _PrepResult:
    """Outcome of the parallel preparation phase. Never touches the DB."""
    source_path: Path
    doc_id: str
    content_hash: str
    chunk_dicts: List[dict] = field(default_factory=list)
    validation_summary: str = ""
    ok: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Thread-safe preparation function (runs in worker threads)
# ---------------------------------------------------------------------------

def _prepare_no_db(source_path: Path, strict: bool) -> _PrepResult:
    """Validate + clean + chunk a single document.  No DB access.

    Safe to call from multiple threads simultaneously — each document is
    independent and all I/O targets different files.
    """
    from rag_lab.ingest.manifest import compute_md5
    from rag_lab.ingest.cleaner import clean_document
    from rag_lab.ingest.markdown_contract import validate_markdown
    from rag_lab.chunking.splitter import chunk_document
    from rag_lab.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP

    doc_id = source_path.stem
    try:
        content_hash = compute_md5(source_path)
        report = validate_markdown(source_path)
        validation_summary = report.summary()

        blocked = report.has_errors or (strict and report.has_warnings)
        if blocked:
            return _PrepResult(
                source_path=source_path, doc_id=doc_id, content_hash=content_hash,
                validation_summary=validation_summary,
                ok=False, error=f"validation failed: {validation_summary}",
            )

        cleaned_path = clean_document(source_path)
        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        chunks = chunk_document(
            cleaned_text, doc_id=doc_id,
            max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP,
        )
        return _PrepResult(
            source_path=source_path, doc_id=doc_id, content_hash=content_hash,
            chunk_dicts=[c.to_dict() for c in chunks],
            validation_summary=validation_summary,
            ok=True,
        )
    except Exception as exc:
        return _PrepResult(
            source_path=source_path, doc_id=doc_id, content_hash="",
            ok=False, error=str(exc),
        )


# ---------------------------------------------------------------------------
# Core batch-ingest pipeline
# ---------------------------------------------------------------------------

def _run_batch_ingest(
    paths: List[Path],
    doc_store,
    vector_store,
    force: bool,
    device: str,
    strict: bool,
    workers: int = 1,
    source_desc: str = "",
    resume_batch_id: Optional[str] = None,
) -> dict:
    """Parallel-prepare → sequential-embed → sequential-write pipeline.

    Workers parallelize validation + chunking only.  All DB and store writes
    are performed on the calling thread (single-writer guarantee).

    Returns a summary dict: batch_id, committed, skipped, failed, total_chunks, elapsed_s.
    """
    import numpy as np
    from rag_lab.config import (
        EMBEDDING_BATCH_SIZE, EMBEDDING_DIM,
        EMBEDDING_MODEL, EMBEDDING_MODEL_VERSION, SPARSE_FORMAT_VERSION,
    )
    from rag_lab.embedding.encoder import encode_chunks
    from rag_lab.ingest.manifest import create_manifest
    from rag_lab.ingest.transaction import (
        IngestBatchStore, IngestDocumentStore, IngestTransaction, _now,
    )
    from rag_lab.storage.metadata_store import MetadataStore

    logger = logging.getLogger("rag_lab")
    t_start = time.time()

    batch_store = IngestBatchStore(doc_store._conn)
    idc_store = IngestDocumentStore(doc_store._conn)

    # --- Create or resume batch ---
    if resume_batch_id:
        batch_id = resume_batch_id
    else:
        batch_id = batch_store.create(source_path=source_desc or None)
        batch_store.update(batch_id, total_docs=len(paths))

    # --- Create PENDING rows (or find resumable ones) ---
    pending_ids: dict = {}  # source_path -> ingest_documents.id

    if resume_batch_id:
        existing = {row["path"]: row for row in idc_store.list_by_batch(batch_id)}
        for sp in paths:
            key = str(sp)
            if key in existing:
                row = existing[key]
                if row["status"] in ("COMMITTED", "SKIPPED"):
                    continue  # already done; do not re-process
                # Increment retry_count and reset to PENDING for re-attempt
                idc_store.set_status(
                    row["id"], "PENDING",
                    retry_count=row["retry_count"] + 1,
                    error_message=None,
                )
                pending_ids[sp] = row["id"]
            else:
                pending_ids[sp] = idc_store.create(batch_id, sp.stem, str(sp))
    else:
        for sp in paths:
            pending_ids[sp] = idc_store.create(batch_id, sp.stem, str(sp))

    paths_to_process = [sp for sp in paths if sp in pending_ids]
    if not paths_to_process:
        console.print("[green]Nothing to process — all documents already completed.[/green]")
        batch_store.finalize(batch_id)
        return {"batch_id": batch_id, "committed": 0, "skipped": 0, "failed": 0,
                "total_chunks": 0, "elapsed_s": 0.0}

    console.print(
        f"[bold cyan]Batch {batch_id}: {len(paths_to_process)} document(s)"
        f"{' (workers=' + str(workers) + ')' if workers > 1 else ''}[/bold cyan]"
    )

    # --- Phase 1: Parallel preparation (worker threads, NO DB access) ---
    prep_results: dict = {}  # source_path -> _PrepResult

    if workers > 1:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ingest-prep"
        ) as pool:
            futures = {
                pool.submit(_prepare_no_db, sp, strict): sp
                for sp in paths_to_process
            }
            for fut in as_completed(futures):
                sp = futures[fut]
                try:
                    prep_results[sp] = fut.result()
                except Exception as exc:
                    prep_results[sp] = _PrepResult(
                        source_path=sp, doc_id=sp.stem, content_hash="",
                        ok=False, error=str(exc),
                    )
    else:
        for sp in paths_to_process:
            prep_results[sp] = _prepare_no_db(sp, strict)

    # --- Phase 2: DB status update + SKIPPED detection (main thread, single writer) ---
    validated_preps: List = []  # [(entry_id, _PrepResult)]
    skipped_count = 0
    failed_count = 0

    for sp in paths_to_process:
        entry_id = pending_ids[sp]
        result = prep_results[sp]

        if not result.ok:
            idc_store.set_status(
                entry_id, "FAILED",
                error_message=result.error,
                validation_summary=result.validation_summary,
                finished_at=_now(),
            )
            failed_count += 1
            console.print(f"  [red]FAILED[/red]   {sp.name}: {result.error}")
            continue

        # SKIPPED detection via ingest_documents (v1.16+) or manifest fallback
        if not force:
            if idc_store.find_committed(result.doc_id, result.content_hash):
                idc_store.set_status(
                    entry_id, "SKIPPED",
                    content_hash=result.content_hash,
                    validation_summary=result.validation_summary,
                    finished_at=_now(),
                )
                skipped_count += 1
                console.print(f"  [dim]SKIPPED[/dim]  {sp.name} (content hash unchanged)")
                continue
            # Fallback: check manifest file (backward compat for pre-v1.16 ingests)
            if _manifest_has_hash(sp, result.content_hash):
                idc_store.set_status(
                    entry_id, "SKIPPED",
                    content_hash=result.content_hash,
                    validation_summary=result.validation_summary,
                    finished_at=_now(),
                )
                skipped_count += 1
                console.print(
                    f"  [dim]SKIPPED[/dim]  {sp.name} (manifest: content hash unchanged)"
                )
                continue

        idc_store.set_status(
            entry_id, "VALIDATED",
            content_hash=result.content_hash,
            validation_summary=result.validation_summary,
        )
        validated_preps.append((entry_id, result))

    # --- Phase 3: Sequential embedding (GPU/CPU model, not thread-safe) ---
    embedded_preps: List = []  # [(entry_id, result, dense_embs)]

    for entry_id, result in validated_preps:
        idc_store.set_status(entry_id, "EMBEDDING")
        try:
            dense_embs, sparse_embs = encode_chunks(
                result.chunk_dicts, batch_size=EMBEDDING_BATCH_SIZE, device=device,
            )
            # Attach embedding metadata to chunk dicts
            for chunk_d in result.chunk_dicts:
                sparse = sparse_embs.get(chunk_d["chunk_id"], {})
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

            embedded_preps.append((entry_id, result, dense_embs))

        except Exception as exc:
            idc_store.set_status(
                entry_id, "FAILED",
                error_message=str(exc)[:500],
                finished_at=_now(),
            )
            failed_count += 1
            console.print(f"  [red]EMBED FAILED[/red] {result.source_path.name}: {exc}")

    # --- Phase 4: Sequential write (single-writer guarantee) ---
    committed_count = 0
    total_chunks_written = 0

    for entry_id, result, dense_embs in embedded_preps:
        idc_store.set_status(entry_id, "WRITING")
        doc_id = result.doc_id
        n_sparse = sum(1 for c in result.chunk_dicts if c.get("sparse_tokens") is not None)

        try:
            with IngestTransaction(doc_id, result.source_path, doc_store) as txn:
                # Link ingest_documents → ingest_runs for full traceability
                idc_store.update(entry_id, run_id=txn.run_id)

                txn.update(chunks_expected=len(result.chunk_dicts))
                vector_store.add(
                    ids=[c["chunk_id"] for c in result.chunk_dicts],
                    embeddings=dense_embs,
                    documents=[c["text"] for c in result.chunk_dicts],
                    metadatas=[
                        {"heading_path": c.get("heading_path", ""), "doc_id": doc_id}
                        for c in result.chunk_dicts
                    ],
                )
                txn.update(chunks_written_chroma=len(result.chunk_dicts))
                doc_store.add(result.chunk_dicts)
                txn.update(
                    chunks_written_docstore=len(result.chunk_dicts),
                    chunks_written_fts5=len(result.chunk_dicts),
                    chunks_written_sparse=n_sparse,
                )
                MetadataStore(conn=doc_store._conn).upsert_document(
                    doc_id, path=str(result.source_path),
                )
                txn.update(metadata_written=1)

            # Update manifest for backward compatibility
            try:
                create_manifest(result.source_path, force=True)
            except Exception:
                pass  # manifest is advisory; corpus is already committed

            idc_store.set_status(
                entry_id, "COMMITTED",
                chunks_count=len(result.chunk_dicts),
                finished_at=_now(),
            )
            committed_count += 1
            total_chunks_written += len(result.chunk_dicts)
            console.print(
                f"  [green]COMMITTED[/green] {result.source_path.name}: "
                f"{len(result.chunk_dicts)} chunks"
            )

        except Exception as exc:
            # IngestTransaction.__exit__ already called rollback()
            idc_store.set_status(
                entry_id, "ROLLED_BACK",
                error_message=str(exc)[:500],
                finished_at=_now(),
            )
            failed_count += 1
            console.print(
                f"  [red]FAILED+ROLLED_BACK[/red] {result.source_path.name}: {exc}"
            )

    # --- Finalize batch ---
    batch_store.finalize(batch_id)
    elapsed = time.time() - t_start

    return {
        "batch_id": batch_id,
        "committed": committed_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "total_chunks": total_chunks_written,
        "elapsed_s": elapsed,
    }


def _print_batch_summary(summary: dict) -> None:
    """Print a human-readable ingest summary."""
    e = summary["elapsed_s"]
    elapsed_str = f"{e:.1f}s" if e < 60 else f"{int(e)//60}m{int(e)%60:02d}s"
    console.print(f"\n[bold]Ingest batch {summary['batch_id']} completed:[/bold]")
    console.print(f"  committed   {summary['committed']}")
    console.print(f"  skipped     {summary['skipped']}")
    console.print(f"  failed      {summary['failed']}")
    console.print(f"  chunks      {summary['total_chunks']}")
    console.print(f"  elapsed     {elapsed_str}")
    if summary["failed"] > 0:
        console.print(
            f"\n[yellow]To retry failed documents:[/yellow]\n"
            f"  rag-lab ingest --retry-failed"
        )


def _manifest_has_hash(source_path: Path, content_hash: str) -> bool:
    """Return True if source_path's doc_id is in the manifest with the given hash."""
    import json
    from rag_lab.config import DATA_DIR
    manifest_path = DATA_DIR / "ingested.jsonl"
    if not manifest_path.exists() or not content_hash:
        return False
    doc_id = source_path.stem
    try:
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("doc_id") == doc_id and entry.get("hash") == content_hash:
                        return True
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return False


def _collect_paths(doc_arg: Optional[str]) -> List[Path]:
    """Resolve --doc argument to a list of Markdown paths.

    - If None: use SOURCES from config.
    - If a directory: glob all *.md files (sorted for determinism).
    - If a file: single-element list.
    """
    from rag_lab.config import SOURCES
    if doc_arg is None:
        return list(SOURCES)
    p = Path(doc_arg)
    if p.is_dir():
        paths = sorted(p.glob("*.md"))
        if not paths:
            console.print(f"[yellow]No .md files found in {p}[/yellow]")
        return paths
    return [p]


# ---------------------------------------------------------------------------
# Main ingest callback (bare `rag-lab ingest [options]`)
# ---------------------------------------------------------------------------

@ingest_app.callback(invoke_without_command=True)
def ingest_main(
    ctx: typer.Context,
    doc: Optional[str] = typer.Option(
        None, "--doc", help="Path to a document file or directory of .md files."
    ),
    force: bool = typer.Option(
        False, "--force", help="Force re-ingestion even if content hash unchanged."
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Resume the most recent incomplete batch (PENDING/FAILED docs).",
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Re-ingest all FAILED/ROLLED_BACK documents."
    ),
    workers: int = typer.Option(
        None, "--workers", "-w",
        help="Worker threads for parallel validation+chunking (default: INGEST_MAX_WORKERS).",
    ),
    cpu_embedding: bool = typer.Option(
        False, "--cpu-embedding", help="Run embedding on CPU."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Treat validation warnings as errors."
    ),
) -> None:
    """Ingest one or more Markdown documents: validate, chunk, embed, and store."""
    if ctx.invoked_subcommand is not None:
        return

    from rag_lab.logging_config import setup_logging
    setup_logging("INFO")

    from rag_lab.config import EMBEDDING_DEVICE, INGEST_MAX_WORKERS
    from rag_lab.storage.docstore import DocStore
    from rag_lab.storage.vector_store import VectorStore
    from rag_lab.ingest.transaction import IngestBatchStore, IngestDocumentStore, _now

    n_workers = workers if workers is not None else INGEST_MAX_WORKERS
    device = "cpu" if cpu_embedding else EMBEDDING_DEVICE

    doc_store = DocStore()
    doc_store.initialize()
    vector_store = VectorStore()
    vector_store.initialize()

    # --- Handle --resume ---
    if resume:
        batch_store = IngestBatchStore(doc_store._conn)
        idc_store = IngestDocumentStore(doc_store._conn)
        batch_id = batch_store.get_latest_incomplete()

        if batch_id is None:
            # Backward compat: fall back to stale IN_PROGRESS ingest_runs
            console.print(
                "[yellow]No incomplete batch found. "
                "Checking for stale IN_PROGRESS runs...[/yellow]"
            )
            _handle_legacy_resume_retry(
                doc_store=doc_store, vector_store=vector_store,
                device=device, strict=strict, workers=n_workers,
                resume=True, retry_failed=False,
            )
        else:
            batch = batch_store.get(batch_id)
            resumable = idc_store.list_resumable(batch_id)
            if not resumable:
                console.print(f"[green]Batch {batch_id} has no pending documents.[/green]")
            else:
                console.print(
                    f"[yellow]Resuming batch {batch_id}: "
                    f"{len(resumable)} document(s) pending[/yellow]"
                )
                paths = [Path(r["path"]) for r in resumable]
                summary = _run_batch_ingest(
                    paths=paths, doc_store=doc_store, vector_store=vector_store,
                    force=True, device=device, strict=strict, workers=n_workers,
                    resume_batch_id=batch_id,
                )
                _print_batch_summary(summary)

        doc_store.close()
        return

    # --- Handle --retry-failed ---
    if retry_failed:
        idc_store = IngestDocumentStore(doc_store._conn)
        failed_docs = idc_store.list_failed()

        if not failed_docs:
            # Backward compat: fall back to FAILED ingest_runs
            console.print(
                "[yellow]No failed documents in ingest_documents. "
                "Checking legacy ingest_runs...[/yellow]"
            )
            _handle_legacy_resume_retry(
                doc_store=doc_store, vector_store=vector_store,
                device=device, strict=strict, workers=n_workers,
                resume=False, retry_failed=True,
            )
        else:
            console.print(
                f"[yellow]Re-ingesting {len(failed_docs)} failed document(s)...[/yellow]"
            )
            paths = [Path(r["path"]) for r in failed_docs]
            # New batch for the retry run
            summary = _run_batch_ingest(
                paths=paths, doc_store=doc_store, vector_store=vector_store,
                force=True, device=device, strict=strict, workers=n_workers,
            )
            _print_batch_summary(summary)

        doc_store.close()
        return

    # --- Normal ingest ---
    paths = _collect_paths(doc)
    if not paths:
        doc_store.close()
        return

    summary = _run_batch_ingest(
        paths=paths,
        doc_store=doc_store,
        vector_store=vector_store,
        force=force,
        device=device,
        strict=strict,
        workers=n_workers,
        source_desc=doc or "SOURCES",
    )
    _print_batch_summary(summary)
    doc_store.close()


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
    """List recent ingest runs (low-level per-document transaction records)."""
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

    _STATUS_COLOR = {
        "COMMITTED": "green", "FAILED": "red", "ROLLED_BACK": "yellow",
        "IN_PROGRESS": "cyan", "PENDING": "dim",
    }

    for r in runs:
        color = _STATUS_COLOR.get(r["status"], "white")
        t.add_row(
            r["run_id"], r["doc_id"],
            f"[{color}]{r['status']}[/{color}]",
            r["started_at"] or "", r["finished_at"] or "",
            str(r["chunks_expected"]), str(r["chunks_written_chroma"]),
            str(r["chunks_written_docstore"]),
        )

    console.print(t)


# ---------------------------------------------------------------------------
# `rag-lab ingest batches`
# ---------------------------------------------------------------------------

@ingest_app.command("batches")
def ingest_batches_cmd(
    limit: int = typer.Option(20, "--limit", help="Maximum rows to show."),
) -> None:
    """List recent ingest batches."""
    from rag_lab.storage.docstore import DocStore
    from rag_lab.ingest.transaction import IngestBatchStore

    ds = DocStore()
    ds.initialize()
    batches = IngestBatchStore(ds._conn).list_batches(limit=limit)
    ds.close()

    if not batches:
        console.print("[dim]No ingest batches found.[/dim]")
        return

    t = Table(title=f"Ingest batches ({len(batches)})")
    for col in ("batch_id", "status", "started_at", "finished_at",
                "committed_docs", "skipped_docs", "failed_docs", "total_chunks"):
        t.add_column(col, no_wrap=True)

    _STATUS_COLOR = {
        "COMPLETED": "green", "PARTIAL": "yellow",
        "FAILED": "red", "IN_PROGRESS": "cyan",
    }

    for b in batches:
        color = _STATUS_COLOR.get(b["status"], "white")
        t.add_row(
            b["batch_id"],
            f"[{color}]{b['status']}[/{color}]",
            (b["started_at"] or "")[:16],
            (b["finished_at"] or "")[:16],
            str(b["committed_docs"]), str(b["skipped_docs"]),
            str(b["failed_docs"]), str(b["total_chunks"]),
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
        f"[green]Rolled back run {run_id} (doc_id={run['doc_id']!r})[/green]"
    )


# ---------------------------------------------------------------------------
# `rag-lab ingest retry RUN_ID`
# ---------------------------------------------------------------------------

@ingest_app.command("retry")
def ingest_retry(
    run_id: str = typer.Argument(..., help="Run ID of a FAILED run to retry."),
    force: bool = typer.Option(False, "--force",
        help="Retry even if run is not in FAILED status."),
    cpu_embedding: bool = typer.Option(False, "--cpu-embedding"),
    strict: bool = typer.Option(False, "--strict",
        help="Treat validation warnings as errors."),
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
            f"[red]Source file not found: {run['source_path']!r}. Cannot retry.[/red]"
        )
        raise typer.Exit(1)

    vs = VectorStore()
    vs.initialize()
    _ingest_one(
        source_path=source_path, doc_store=ds, vector_store=vs,
        force=True, device=device, strict=strict,
    )
    ds.close()


# ---------------------------------------------------------------------------
# Private helpers (legacy + backward-compat)
# ---------------------------------------------------------------------------

def _ingest_one(
    source_path: Path,
    doc_store,
    vector_store,
    force: bool,
    device: str,
    strict: bool = False,
) -> int:
    """Ingest one document, wrapped in an IngestTransaction.

    Legacy function used by `ingest retry` and `_handle_legacy_resume_retry`.
    New code should use `_run_batch_ingest` instead.

    Returns:
        Number of chunks ingested (0 if skipped or validation failed).
    """
    import numpy as np
    from rag_lab.config import (
        CHUNK_MAX_TOKENS, CHUNK_OVERLAP, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM,
        EMBEDDING_MODEL, EMBEDDING_MODEL_VERSION, SPARSE_FORMAT_VERSION,
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

    # Validation gate — runs before any store writes
    from rag_lab.ingest.markdown_contract import validate_markdown
    from rag_lab.ingest.validation import ValidationSeverity

    report = validate_markdown(source_path)
    if report.issues:
        _SCOLOR = {
            ValidationSeverity.ERROR: "red",
            ValidationSeverity.WARN: "yellow",
            ValidationSeverity.INFO: "dim",
        }
        for issue in report.issues:
            color = _SCOLOR[issue.severity]
            console.print(
                f"  [{color}]{issue.severity.value:<5}[/{color}] "
                f"[{issue.code}]: {issue.message}"
            )

    blocked = report.has_errors or (strict and report.has_warnings)
    if blocked:
        reason = "validation errors" if report.has_errors else "validation warnings (--strict mode)"
        console.print(f"[red]Skipping {source_path.name}: {reason}[/red]")
        return 0

    console.print(f"[bold cyan]Ingesting: {source_path.name}[/bold cyan]")

    cleaned_path = clean_document(source_path)
    create_manifest(source_path, cleaned_path, force=force)

    cleaned_text = cleaned_path.read_text(encoding="utf-8")
    chunks = chunk_document(
        cleaned_text, doc_id=source_path.stem,
        max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP,
    )
    logger.info(f"Created {len(chunks)} chunks from {source_path.name}")

    chunk_dicts = [c.to_dict() for c in chunks]

    dense_embeddings, sparse_embeddings = encode_chunks(
        chunk_dicts, batch_size=EMBEDDING_BATCH_SIZE, device=device,
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

        MetadataStore(conn=doc_store._conn).upsert_document(doc_id, path=str(source_path))
        txn.update(metadata_written=1)

    console.print(
        f"[bold green]Ingested {len(chunk_dicts)} chunks from {source_path.name}[/bold green]"
    )
    return len(chunk_dicts)


def _handle_legacy_resume_retry(
    *,
    doc_store,
    vector_store,
    device: str,
    strict: bool,
    workers: int,
    resume: bool,
    retry_failed: bool,
) -> None:
    """Backward-compat: handle stale ingest_runs (pre-v1.16 ingests)."""
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
            f"[cyan]Re-ingesting {run['doc_id']!r} (previous run={run['run_id']})[/cyan]"
        )
        _ingest_one(
            source_path=source_path, doc_store=doc_store, vector_store=vector_store,
            force=True, device=device, strict=strict,
        )
