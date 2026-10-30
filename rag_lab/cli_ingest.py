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
    # Frontmatter fields parsed during preparation
    fm_title: Optional[str] = None
    fm_domain: Optional[str] = None
    fm_source_type: Optional[str] = None
    fm_language: Optional[str] = None
    fm_version: Optional[str] = None
    fm_tags: List[str] = field(default_factory=list)
    fm_derived_tags: List[str] = field(default_factory=list)


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
    from rag_lab.ingest.frontmatter import parse_frontmatter, extract_h1_title
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

        # Parse frontmatter for document classification metadata
        raw_text = source_path.read_text(encoding="utf-8")
        fm = parse_frontmatter(raw_text)
        # Resolve title: frontmatter > H1 heading
        title = fm.title or extract_h1_title(raw_text)

        # Stable document identity (RAG-05): prefer the declared frontmatter
        # doc_id; fall back to the file stem for legacy documents without
        # frontmatter (compatible with the existing corpus).
        resolved_doc_id = doc_id
        if fm.doc_id and fm.doc_id.strip():
            resolved_doc_id = fm.doc_id.strip()

        cleaned_path = clean_document(source_path)
        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        chunks = chunk_document(
            cleaned_text, doc_id=resolved_doc_id,
            max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP,
        )
        return _PrepResult(
            source_path=source_path, doc_id=resolved_doc_id, content_hash=content_hash,
            chunk_dicts=[c.to_dict() for c in chunks],
            validation_summary=validation_summary,
            ok=True,
            fm_title=title,
            fm_domain=fm.domain,
            fm_source_type=fm.source_type,
            fm_language=fm.language,
            fm_version=fm.version,
            fm_tags=fm.tags,
            fm_derived_tags=fm.derived_tags,
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
    from rag_lab.ingest import versioning
    from rag_lab.ingest import ownership

    logger = logging.getLogger("rag_lab")
    t_start = time.time()

    batch_store = IngestBatchStore(doc_store._conn)
    idc_store = IngestDocumentStore(doc_store._conn)

    # Finish any version operation interrupted by a previous crash (RAG-01/02).
    try:
        versioning.recover_pending_ops(doc_store._conn, vector_store)
    except Exception as exc:
        console.print(
            f"[red]Aborting ingest: pending version operations could not be "
            f"recovered:[/red] {exc}"
        )
        return {
            "batch_id": None, "committed": 0, "skipped": 0,
            "failed": len(paths), "total_chunks": 0, "elapsed_s": 0.0,
        }

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

    # --- Phase 2: identity + ownership + SKIPPED detection (single writer) ---
    validated_preps: List = []  # [(entry_id, _PrepResult, Decision)]
    skipped_count = 0
    failed_count = 0
    _seen_doc_ids: dict = {}

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

        # R2-05: persist the resolved identity (frontmatter doc_id) BEFORE the
        # skip checks, so SKIPPED rows reference the real document.
        idc_store.update(
            entry_id,
            doc_id=result.doc_id,
            content_hash=result.content_hash or None,
        )

        # P0C: deterministic ownership decision over the persisted confirmed
        # version (replaces the loose-hash guard and the historical SKIPPED
        # checks).  NO-OP of a different source writes no documental state.
        conflict = None
        prior_path = _seen_doc_ids.get(result.doc_id)
        if prior_path is not None and _norm_path(prior_path) != _norm_path(result.source_path):
            conflict = (
                f"duplicate doc_id {result.doc_id!r}: already claimed by "
                f"{prior_path!r} in this batch"
            )
        if conflict is not None:
            idc_store.set_status(
                entry_id, "FAILED",
                error_message=conflict,
                finished_at=_now(),
            )
            failed_count += 1
            console.print(f"  [red]FAILED[/red]   {sp.name}: {conflict}")
            continue
        _seen_doc_ids[result.doc_id] = str(result.source_path)

        decision = ownership.evaluate(
            doc_store._conn, vector_store,
            doc_id=result.doc_id,
            origin_path=result.source_path,
            content_hash=result.content_hash,
            chunk_dicts=result.chunk_dicts,
            classification=ownership.classification_from_fields(
                result.fm_title, result.fm_domain, result.fm_source_type,
                result.fm_language, result.fm_version,
                result.fm_tags + result.fm_derived_tags,
            ),
            force=force,
        )
        if decision.is_denied():
            idc_store.set_status(
                entry_id, "FAILED",
                error_message=f"{decision.kind.value}: {decision.reason}",
                finished_at=_now(),
            )
            failed_count += 1
            console.print(f"  [red]DENIED[/red]   {sp.name}: {decision.reason}")
            continue
        if decision.is_noop():
            label = ("SKIPPED" if decision.kind is ownership.DecisionKind.NOOP_OWNER
                     else "NOOP-EQUIV")
            idc_store.set_status(
                entry_id, "SKIPPED",
                content_hash=result.content_hash,
                validation_summary=f"{decision.kind.value}: {decision.reason}",
                finished_at=_now(),
            )
            skipped_count += 1
            console.print(f"  [dim]{label}[/dim]  {sp.name} ({decision.reason})")
            continue

        idc_store.set_status(
            entry_id, "VALIDATED",
            content_hash=result.content_hash,
            validation_summary=result.validation_summary,
        )
        validated_preps.append((entry_id, result, decision))

    # --- Phase 3: Sequential embedding (GPU/CPU model, not thread-safe) ---
    embedded_preps: List = []  # [(entry_id, result, dense_embs, Decision)]

    for entry_id, result, decision in validated_preps:
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

            embedded_preps.append((entry_id, result, dense_embs, decision))

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

    for entry_id, result, dense_embs, decision in embedded_preps:
        doc_id = result.doc_id

        # TOCTOU guard: revalidate the authorization immediately before the
        # write.  The decision must be unchanged (same kind/run/generation)
        # and still grant CREATE/REPLACE.  Single-writer process; multiprocess
        # concurrency is documented as out of scope.
        revalidate = ownership.evaluate(
            doc_store._conn, vector_store,
            doc_id=result.doc_id,
            origin_path=result.source_path,
            content_hash=result.content_hash,
            chunk_dicts=result.chunk_dicts,
            classification=ownership.classification_from_fields(
                result.fm_title, result.fm_domain, result.fm_source_type,
                result.fm_language, result.fm_version,
                result.fm_tags + result.fm_derived_tags,
            ),
            force=force,
        )
        if (revalidate.is_denied() or revalidate.is_noop()
                or revalidate.kind is not decision.kind
                or revalidate.active_run_id != decision.active_run_id
                or revalidate.generation != decision.generation):
            idc_store.set_status(
                entry_id, "FAILED",
                error_message=("authorization changed before publish: "
                               f"{revalidate.kind.value}: {revalidate.reason}"),
                finished_at=_now(),
            )
            failed_count += 1
            console.print(
                f"  [red]REVALIDATION FAILED[/red] {result.source_path.name}: "
                f"{revalidate.reason}"
            )
            continue

        idc_store.set_status(entry_id, "WRITING")
        n_sparse = sum(1 for c in result.chunk_dicts if c.get("sparse_tokens") is not None)

        try:
            chunk_ids = [c["chunk_id"] for c in result.chunk_dicts]
            with IngestTransaction(
                doc_id, result.source_path, doc_store,
                vector_store=vector_store, chunk_ids=chunk_ids,
                content_hash=result.content_hash,
            ) as txn:
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
                ms = MetadataStore(conn=doc_store._conn)
                ms.upsert_document(
                    doc_id,
                    path=str(result.source_path),
                    content_hash=result.content_hash,
                    title=result.fm_title,
                    domain=result.fm_domain,
                    source_type=result.fm_source_type,
                    language=result.fm_language,
                    version=result.fm_version,
                )
                # Import explicit tags + derived tags from frontmatter
                for tag in result.fm_tags + result.fm_derived_tags:
                    ms.assign_tag(doc_id, tag)
                txn.update(metadata_written=1)

            # Update manifest for backward compatibility
            try:
                create_manifest(result.source_path, force=True, doc_id=doc_id)
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


def _norm_path(p) -> str:
    """Best-effort normalised path string for ownership comparison."""
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


def _table_exists(conn, name: str) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None
    except Exception:
        return False


def _active_run_id(conn, doc_id: str) -> Optional[str]:
    if not _table_exists(conn, "document_versions"):
        return None
    row = conn.execute(
        "SELECT active_run_id FROM document_versions WHERE doc_id=?", (doc_id,)
    ).fetchone()
    return row[0] if row else None


def _manifest_owner(doc_id: str):
    """Return (path, hash) from the advisory manifest for doc_id, or None."""
    import json
    from rag_lab.config import DATA_DIR
    p = DATA_DIR / "ingested.jsonl"
    if not p.exists():
        return None
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("doc_id") == doc_id:
                return (entry.get("path"), entry.get("hash"))
    except OSError:
        return None
    return None


def _manifest_hash_for_path(doc_id: str, owner_path) -> Optional[str]:
    """Historical hash from the manifest, only if its path matches the owner."""
    man = _manifest_owner(doc_id)
    if not man or not owner_path:
        return None
    man_path, man_hash = man
    if man_hash and man_path and _norm_path(man_path) == _norm_path(owner_path):
        return man_hash
    return None


def _stored_content_md5(conn, doc_id: str) -> Optional[str]:
    """MD5 of the persisted confirmed content (ordered chunk texts)."""
    import hashlib
    try:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE doc_id=? ORDER BY rowid", (doc_id,)
        ).fetchall()
    except Exception:
        return None
    if not rows:
        return None
    h = hashlib.md5()
    for (t,) in rows:
        h.update((t or "").encode("utf-8"))
    return h.hexdigest()


def _incoming_chunk_set(conn, doc_id: str, source_path):
    """Chunk the incoming source the same way ingest would, for legacy proof."""
    try:
        from rag_lab.ingest.cleaner import clean_document
        from rag_lab.chunking.splitter import chunk_document
        from rag_lab.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP
        cleaned = clean_document(Path(source_path))
        text = cleaned.read_text(encoding="utf-8")
        chunks = chunk_document(text, doc_id=doc_id,
                                max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP)
        return {c.chunk_id: c.text for c in chunks}
    except Exception:
        return None


def _legacy_proof(conn, doc_id: str, owner_path, source_path=None) -> Optional[str]:
    """Prove a legacy transfer against the *persisted* confirmed content.

    The advisory manifest hash is trusted only when corroborated by the
    persisted content: either as a whole blob (single-chunk legacy stores) or by
    the incoming source reproducing the stored chunk set exactly.  Editing both
    the manifest and the current source cannot pass this test (R2-F2).
    """
    man_hash = _manifest_hash_for_path(doc_id, owner_path)
    if not man_hash:
        return None
    if _stored_content_md5(conn, doc_id) == man_hash:
        return man_hash
    incoming = _incoming_chunk_set(conn, doc_id, source_path) if source_path else None
    if incoming:
        try:
            rows = conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE doc_id=? ORDER BY rowid", (doc_id,)
            ).fetchall()
        except Exception:
            rows = []
        stored = {r[0]: (r[1] or "") for r in rows}
        incoming = {k: (v or "") for k, v in incoming.items()}
        if stored and set(stored) == set(incoming) and stored == incoming:
            return man_hash
    return None


def _confirmed_owner(conn, doc_id: str, source_path=None):
    """Return (path, content_hash, source, coherent) of the confirmed owner.

    All confirmed hashes bound to the same version must agree; otherwise the
    evidence is incoherent and ownership fails closed (R2-F1).  The advisory
    manifest never overrides a confirmed hash; it can only authorize a legacy
    transfer when its hash also matches the persisted content (R2-F2).  Path
    coherence is enforced against the active OR legacy COMMITTED run (R2-F3).
    """
    doc_path = doc_hash = None
    if _table_exists(conn, "documents"):
        row = conn.execute(
            "SELECT path, content_hash FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if row:
            doc_path, doc_hash = row

    active_run = _active_run_id(conn, doc_id)
    active_hash = ""
    if active_run and _table_exists(conn, "document_versions"):
        r = conn.execute(
            "SELECT content_hash FROM document_versions WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if r and r[0]:
            active_hash = r[0]

    run_path = run_hash = None
    if active_run and _table_exists(conn, "ingest_runs"):
        row = conn.execute(
            "SELECT source_path FROM ingest_runs WHERE run_id=?", (active_run,)
        ).fetchone()
        if row:
            run_path = row[0]
        if _table_exists(conn, "ingest_documents"):
            row = conn.execute(
                "SELECT content_hash FROM ingest_documents "
                "WHERE run_id=? AND doc_id=? AND status='COMMITTED' LIMIT 1",
                (active_run, doc_id),
            ).fetchone()
            if row:
                run_hash = row[0]

    legacy_path = legacy_hash = None
    if active_run is None and _table_exists(conn, "ingest_runs"):
        row = conn.execute(
            "SELECT run_id, source_path FROM ingest_runs WHERE doc_id=? AND status='COMMITTED' "
            "AND source_path IS NOT NULL ORDER BY rowid DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
        if row:
            legacy_path = row[1]
            if _table_exists(conn, "ingest_documents"):
                r2 = conn.execute(
                    "SELECT content_hash FROM ingest_documents "
                    "WHERE run_id=? AND doc_id=? AND status='COMMITTED' LIMIT 1",
                    (row[0], doc_id),
                ).fetchone()
                if r2:
                    legacy_hash = r2[0]

    # Path coherence: documents must agree with the active OR legacy run path.
    if doc_path and run_path and _norm_path(doc_path) != _norm_path(run_path):
        return (None, None, "incoherent", False)
    if active_run is None and doc_path and legacy_path and _norm_path(doc_path) != _norm_path(legacy_path):
        return (None, None, "incoherent", False)

    # Collect every confirmed hash bound to the confirmed owner; all must agree.
    hashes = []
    if active_run:
        if active_hash:
            hashes.append(active_hash)
        if run_hash:
            hashes.append(run_hash)
        if doc_hash and (not doc_path or not run_path or _norm_path(doc_path) == _norm_path(run_path)):
            hashes.append(doc_hash)
        owner_path = run_path or doc_path
        source = "active_run"
    elif legacy_path:
        if legacy_hash:
            hashes.append(legacy_hash)
        if doc_hash and doc_path and _norm_path(doc_path) == _norm_path(legacy_path):
            hashes.append(doc_hash)
        owner_path = legacy_path
        source = "legacy_run"
    elif doc_path:
        if doc_hash:
            hashes.append(doc_hash)
        owner_path = doc_path
        source = "documents"
    else:
        return None

    hashes = [h for h in hashes if h]
    if len(set(hashes)) > 1:
        return (None, None, "incoherent", False)
    if hashes:
        return (owner_path, hashes[0], source, True)
    # No confirmed hash: the advisory manifest may authorize only if it also
    # matches the persisted content.
    if owner_path:
        return (owner_path, _legacy_proof(conn, doc_id, owner_path, source_path),
                source, True)
    return None


def _has_document_data(conn, doc_id: str) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM chunks WHERE doc_id=? LIMIT 1", (doc_id,)
        ).fetchone() is not None
    except Exception:
        # Fail closed: if we cannot check, assume data may exist.
        return True


def _identity_owner_conflict(conn, doc_id: str, source_path, content_hash: str = None) -> Optional[str]:
    """Return a conflict message if ``doc_id`` is owned by a different source.

    Authorization is only by the same source path or by a matching *historical*
    content hash bound to the confirmed version.  Different path with no
    trustworthy historical hash, contradictory evidence, or data with no
    provable owner all fail closed.
    """
    try:
        owner = _confirmed_owner(conn, doc_id, source_path)
    except Exception as exc:
        return f"cannot verify ownership for doc_id {doc_id!r} (fail-closed): {exc}"

    cur = _norm_path(source_path)
    if owner is None:
        if _has_document_data(conn, doc_id):
            return (
                f"doc_id {doc_id!r} already has data but no provable confirmed "
                "owner; refusing to overwrite (fail-closed)"
            )
        return None

    owner_path, owner_hash, src, coherent = owner
    if not coherent:
        return (
            f"doc_id {doc_id!r} has contradictory ownership evidence ({src}); "
            "refusing to overwrite (fail-closed)"
        )
    if owner_path and _norm_path(owner_path) == cur:
        return None
    if content_hash and owner_hash and content_hash == owner_hash:
        return None
    return (
        f"doc_id {doc_id!r} is already owned by {owner_path!r} ({src}); refusing "
        "to overwrite a different source. If this is the same document moved or "
        "re-exported, reconcile it explicitly (a historical content hash is required)."
    )


def _manifest_has_hash(source_path: Path, content_hash: str, doc_id: str = None) -> bool:
    """Return True if source_path's doc_id is in the manifest with the given hash."""
    import json
    from rag_lab.config import DATA_DIR
    manifest_path = DATA_DIR / "ingested.jsonl"
    if not manifest_path.exists() or not content_hash:
        return False
    doc_id = doc_id or source_path.stem
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

    # P0B-03: recover interrupted version operations BEFORE selecting pending
    # documents, so WRITING/RESTORING rows are reconciled (and become resumable)
    # instead of being invisible to list_resumable().
    from rag_lab.ingest import versioning
    try:
        versioning.recover_pending_ops(doc_store._conn, vector_store)
    except Exception as exc:
        console.print(
            f"[red]Aborting ingest: pending version operations could not be "
            f"recovered:[/red] {exc}"
        )
        doc_store.close()
        raise typer.Exit(1)

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
    from rag_lab.storage.vector_store import VectorStore
    from rag_lab.ingest.transaction import IngestRunStore
    from rag_lab.ingest import versioning

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

    vs = VectorStore()
    vs.initialize()
    result = versioning.admin_rollback(run_id, ds._conn, vs)
    ds.close()

    if result["status"] == "error":
        console.print(f"[red]{result['reason']}[/red]")
        raise typer.Exit(1)
    if result["status"] == "refused":
        # RAG-02: never destroy a newer committed version.
        console.print(f"[yellow]{result['reason']}[/yellow]")
        raise typer.Exit(1)

    restored = result.get("restored_previous")
    console.print(
        f"[green]Rolled back run {run_id} (doc_id={run['doc_id']!r}); "
        f"restored_previous_version={restored}[/green]"
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
    from rag_lab.ingest.transaction import IngestRunStore
    from rag_lab.ingest import versioning
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

    vs = VectorStore()
    vs.initialize()

    if run["status"] in ("FAILED", "IN_PROGRESS"):
        result = versioning.admin_rollback(run_id, ds._conn, vs)
        if result["status"] == "refused":
            # A newer version exists; do not destroy it, just retry the source.
            console.print(f"[yellow]{result['reason']}[/yellow]")

    source_path = Path(run["source_path"]) if run["source_path"] else None
    if not source_path or not source_path.exists():
        console.print(
            f"[red]Source file not found: {run['source_path']!r}. Cannot retry.[/red]"
        )
        raise typer.Exit(1)

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
    from rag_lab.ingest.manifest import create_manifest, compute_md5
    from rag_lab.chunking.splitter import chunk_document
    from rag_lab.embedding.encoder import encode_chunks
    from rag_lab.ingest.transaction import IngestTransaction
    from rag_lab.storage.metadata_store import MetadataStore
    from rag_lab.ingest import ownership

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

    # Parse frontmatter for document classification metadata
    from rag_lab.ingest.frontmatter import parse_frontmatter, extract_h1_title
    raw_text = source_path.read_text(encoding="utf-8")
    _fm = parse_frontmatter(raw_text)
    _fm_title = _fm.title or extract_h1_title(raw_text)

    cleaned_path = clean_document(source_path)

    cleaned_text = cleaned_path.read_text(encoding="utf-8")
    _resolved_doc_id = _fm.doc_id.strip() if _fm.doc_id and _fm.doc_id.strip() else source_path.stem
    chunks = chunk_document(
        cleaned_text, doc_id=_resolved_doc_id,
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

    doc_id = _resolved_doc_id
    n_sparse = sum(1 for c in chunk_dicts if c.get("sparse_tokens") is not None)
    chunk_ids = [c.get("chunk_id", "") for c in chunk_dicts]
    _content_hash = compute_md5(source_path)

    # P0C: deterministic ownership decision over the confirmed persisted
    # version.  NO-OP (same owner or exact equivalence from another path)
    # performs no documental write and does not bump the generation.
    _decision = ownership.evaluate(
        doc_store._conn, vector_store,
        doc_id=doc_id,
        origin_path=source_path,
        content_hash=_content_hash,
        chunk_dicts=chunk_dicts,
        classification=ownership.classification_from_fields(
            _fm_title, _fm.domain, _fm.source_type, _fm.language, _fm.version,
            _fm.tags + _fm.derived_tags,
        ),
        force=force,
    )
    if _decision.is_denied():
        console.print(
            f"[red]Denied {source_path.name}: {_decision.reason}[/red]"
        )
        return 0
    if _decision.is_noop():
        label = ("NO-OP (owner)" if _decision.kind is ownership.DecisionKind.NOOP_OWNER
                 else "NO-OP (equivalent other source)")
        console.print(f"[dim]{label}: {source_path.name}[/dim]")
        return 0

    _revalidate = ownership.evaluate(
        doc_store._conn, vector_store,
        doc_id=doc_id,
        origin_path=source_path,
        content_hash=_content_hash,
        chunk_dicts=chunk_dicts,
        classification=ownership.classification_from_fields(
            _fm_title, _fm.domain, _fm.source_type, _fm.language, _fm.version,
            _fm.tags + _fm.derived_tags,
        ),
        force=force,
    )
    if (_revalidate.is_denied() or _revalidate.is_noop()
            or _revalidate.kind is not _decision.kind
            or _revalidate.active_run_id != _decision.active_run_id
            or _revalidate.generation != _decision.generation):
        console.print(
            f"[red]Authorization changed before publish: "
            f"{_revalidate.reason}[/red]"
        )
        return 0

    with IngestTransaction(doc_id, source_path, doc_store,
                           vector_store=vector_store, chunk_ids=chunk_ids,
                           content_hash=_content_hash) as txn:
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

        _ms = MetadataStore(conn=doc_store._conn)
        _ms.upsert_document(
            doc_id,
            path=str(source_path),
            content_hash=compute_md5(source_path),
            title=_fm_title,
            domain=_fm.domain,
            source_type=_fm.source_type,
            language=_fm.language,
            version=_fm.version,
        )
        for tag in _fm.tags + _fm.derived_tags:
            _ms.assign_tag(doc_id, tag)
        txn.update(metadata_written=1)

    # Manifest is advisory; update only after a successful commit (RAG-01).
    try:
        create_manifest(source_path, doc_id=doc_id, force=True)
    except Exception:
        pass

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
        from rag_lab.ingest import versioning
        from rag_lab.storage.vector_store import VectorStore as _VS
        _vs = vector_store or _VS()
        try:
            _vs.initialize()
        except Exception:
            pass
        versioning.admin_rollback(run["run_id"], doc_store._conn, _vs)

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
