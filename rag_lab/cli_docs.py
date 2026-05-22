"""Typer sub-apps for docs and tags management commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rag_lab.config import DOCDSTORE_SQLITE_PATH

__all__ = ["docs_app", "tags_app"]

docs_app = typer.Typer(name="docs", help="Manage documents across all stores.")
tags_app = typer.Typer(name="tags", help="Manage document tags.")

console = Console()


# ---------------------------------------------------------------------------
# docs commands
# ---------------------------------------------------------------------------


@docs_app.command("list")
def docs_list(
    tag: Optional[str] = typer.Option(None, "--tag", help="Filter by tag name."),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source_id."),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status (default: all statuses)."),
):
    """List documents with optional filters."""
    from rag_lab.storage.metadata_store import MetadataStore

    store = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
    store.initialize()
    try:
        docs = store.list_documents(
            tag=tag,
            source_id=source,
            status=status,
        )
    finally:
        store.close()

    if not docs:
        console.print("[yellow]No documents found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("doc_id", no_wrap=True)
    table.add_column("title / path")
    table.add_column("status", justify="center")
    table.add_column("chunks", justify="right")
    table.add_column("tags")

    for doc in docs:
        label = doc["title"] or doc["path"] or "(no title)"
        tags_str = ", ".join(doc["tags"]) if doc["tags"] else ""
        table.add_row(
            doc["doc_id"],
            label,
            doc["status"],
            str(doc["chunk_count"]),
            tags_str,
        )

    console.print(table)


@docs_app.command("show")
def docs_show(doc_id: str = typer.Argument(..., help="Document ID to inspect.")):
    """Show all metadata for a document."""
    from rag_lab.storage.metadata_store import MetadataStore
    from rag_lab.storage.docstore import DocStore

    store = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
    store.initialize()
    try:
        doc = store.get_document(doc_id)
    finally:
        store.close()

    if doc is None:
        console.print(f"[red]Document not found: {doc_id}[/red]")
        raise typer.Exit(1)

    ds = DocStore(db_path=DOCDSTORE_SQLITE_PATH)
    ds.initialize()
    try:
        chunk_count = ds.count_chunks(doc_id)
    finally:
        ds.close()

    console.print(f"\n[bold cyan]Document:[/bold cyan] {doc_id}\n")
    for key in (
        "title", "path", "content_hash", "source_id",
        "status", "created_at", "updated_at", "ingested_at",
        "embedding_model_version", "embedding_dim", "sparse_format_version",
    ):
        console.print(f"  {key:<28} {doc.get(key)}")
    console.print(f"  {'chunks':<28} {chunk_count}")
    tags_str = ", ".join(doc["tags"]) if doc["tags"] else "(none)"
    console.print(f"  {'tags':<28} {tags_str}")
    console.print()


@docs_app.command("tag")
def docs_tag(
    doc_id: str = typer.Argument(...),
    tag_name: str = typer.Argument(...),
):
    """Add a tag to a document."""
    from rag_lab.storage.metadata_store import MetadataStore

    store = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
    store.initialize()
    try:
        if store.get_document(doc_id) is None:
            console.print(f"[red]Document not found: {doc_id}[/red]")
            raise typer.Exit(1)
        store.assign_tag(doc_id, tag_name)
    finally:
        store.close()

    console.print(f"Tag '{tag_name}' added to {doc_id}.")


@docs_app.command("untag")
def docs_untag(
    doc_id: str = typer.Argument(...),
    tag_name: str = typer.Argument(...),
):
    """Remove a tag from a document."""
    from rag_lab.storage.metadata_store import MetadataStore

    store = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
    store.initialize()
    try:
        store.unassign_tag(doc_id, tag_name)
    finally:
        store.close()

    console.print(f"Tag '{tag_name}' removed from {doc_id} (no-op if not assigned).")


@docs_app.command("delete")
def docs_delete(
    doc_id: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", "-f", help="Delete without confirmation."),
):
    """Delete a document from all stores (SQLite + ChromaDB)."""
    from rag_lab.storage.docstore import DocStore
    from rag_lab.storage.vector_store import VectorStore

    ds = DocStore(db_path=DOCDSTORE_SQLITE_PATH)
    ds.initialize()
    try:
        chunk_count = ds.count_chunks(doc_id)
    finally:
        ds.close()

    if chunk_count == 0:
        # Still check whether the document row itself exists before aborting.
        from rag_lab.storage.metadata_store import MetadataStore
        ms = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
        ms.initialize()
        try:
            exists = ms.get_document(doc_id) is not None
        finally:
            ms.close()
        if not exists:
            console.print(f"[red]Document not found: {doc_id}[/red]")
            raise typer.Exit(1)

    if not force:
        console.print(
            f"Will delete doc_id=[bold]{doc_id}[/bold]: "
            f"{chunk_count} chunks from SQLite, vectors from ChromaDB."
        )
        confirmed = typer.confirm("Proceed?")
        if not confirmed:
            console.print("Aborted.")
            raise typer.Exit(0)

    ds2 = DocStore(db_path=DOCDSTORE_SQLITE_PATH)
    ds2.initialize()
    try:
        deleted_chunks = ds2.delete_by_doc_id(doc_id)
    finally:
        ds2.close()

    vs = VectorStore()
    vs.initialize()
    vs._collection.delete(where={"doc_id": {"$eq": doc_id}})

    console.print(
        f"Deleted {doc_id}: {deleted_chunks} chunks from SQLite, vectors removed from ChromaDB."
    )


@docs_app.command("set-source")
def docs_set_source(
    doc_id: str = typer.Argument(...),
    source_id: str = typer.Argument(...),
):
    """Set the source_id for a document."""
    from rag_lab.storage.metadata_store import MetadataStore
    import sqlite3

    store = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
    store.initialize()
    try:
        if store.get_document(doc_id) is None:
            console.print(f"[red]Document not found: {doc_id}[/red]")
            raise typer.Exit(1)
        store._conn.execute(
            "UPDATE documents SET source_id = ?, updated_at = datetime('now') WHERE doc_id = ?",
            (source_id, doc_id),
        )
        store._conn.commit()
    finally:
        store.close()

    console.print(f"source_id for {doc_id} set to '{source_id}'.")


# ---------------------------------------------------------------------------
# docs validate / inspect / preview-chunks
# ---------------------------------------------------------------------------


@docs_app.command("validate")
def docs_validate(
    path: str = typer.Argument(..., help="Path to the Markdown file to validate."),
    strict: bool = typer.Option(
        False, "--strict", help="Treat warnings as errors (exit 1 on any warning)."
    ),
) -> None:
    """Validate a Markdown document against the canonical contract."""
    from pathlib import Path as P
    from rag_lab.ingest.markdown_contract import validate_markdown
    from rag_lab.ingest.validation import ValidationSeverity

    doc_path = P(path)
    if not doc_path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    report = validate_markdown(doc_path)

    if not report.issues:
        console.print(f"[green]✓[/green] {doc_path.name} — OK")
        return

    _SEVERITY_COLOR = {
        ValidationSeverity.ERROR: "red",
        ValidationSeverity.WARN: "yellow",
        ValidationSeverity.INFO: "dim",
    }
    for issue in report.issues:
        color = _SEVERITY_COLOR[issue.severity]
        loc = f" (line {issue.line_number})" if issue.line_number else ""
        console.print(
            f"[{color}]{issue.severity.value:<5}[/{color}]  "
            f"[{issue.code}]{loc}: {issue.message}"
        )

    if report.has_errors:
        console.print(f"\n[red]INVALID[/red] — {report.summary()}")
        raise typer.Exit(1)
    elif strict and report.has_warnings:
        console.print(f"\n[yellow]INVALID[/yellow] (--strict) — {report.summary()}")
        raise typer.Exit(1)
    else:
        console.print(f"\n[yellow]VALID with warnings[/yellow] — {report.summary()}")


@docs_app.command("inspect")
def docs_inspect(
    path: str = typer.Argument(..., help="Path to the Markdown file."),
) -> None:
    """Show structural summary of a Markdown document."""
    import re
    from pathlib import Path as P
    from rag_lab.ingest.markdown_contract import (
        MarkdownValidationConfig,
        validate_markdown,
    )
    from rag_lab.ingest.validation import ValidationSeverity, count_tokens_approx
    from rag_lab.config import CHUNK_MAX_TOKENS

    doc_path = P(path)
    if not doc_path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    try:
        text = doc_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        console.print(f"[red]File is not valid UTF-8[/red]")
        raise typer.Exit(1)

    lines = text.splitlines()
    total_tokens = count_tokens_approx(text)
    estimated_chunks = max(1, total_tokens // CHUNK_MAX_TOKENS)

    heading_counts: dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    title: Optional[str] = None
    for line in lines:
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            heading_counts[level] += 1
            if level == 1 and title is None:
                title = m.group(2).strip()

    total_headings = sum(heading_counts.values())

    in_table = False
    table_count = 0
    for line in lines:
        stripped = line.strip()
        is_table = stripped.startswith("|") and "|" in stripped
        if is_table and not in_table:
            in_table = True
            table_count += 1
        elif not is_table:
            in_table = False

    report = validate_markdown(doc_path, MarkdownValidationConfig())
    file_size_kb = doc_path.stat().st_size / 1024

    console.print(f"\n[bold cyan]Inspect:[/bold cyan] {doc_path.name}\n")
    console.print(f"  {'doc_id':<24} {doc_path.stem}")
    console.print(f"  {'title':<24} {title or '(none)'}")
    console.print(f"  {'file_size':<24} {file_size_kb:.1f} KB")
    console.print(f"  {'total_lines':<24} {len(lines):,}")
    console.print(f"  {'total_tokens (~)':<24} {total_tokens:,}")
    console.print(f"  {'estimated_chunks (~)':<24} {estimated_chunks}")
    console.print(f"  {'total_headings':<24} {total_headings}")
    for level in range(1, 7):
        if heading_counts[level] > 0:
            console.print(f"  {'  H' + str(level):<24} {heading_counts[level]}")
    console.print(f"  {'tables':<24} {table_count}")

    n_errors = len(report.errors)
    n_warns = len(report.warnings)
    if not report.issues:
        val_status = "[green]OK[/green]"
    elif n_errors:
        val_status = f"[red]{n_errors} error(s)[/red]"
        if n_warns:
            val_status += f", [yellow]{n_warns} warning(s)[/yellow]"
    else:
        val_status = f"[yellow]{n_warns} warning(s)[/yellow]"
    console.print(f"  {'validation':<24} {val_status}")

    if report.issues:
        console.print()
        _COLOR = {
            ValidationSeverity.ERROR: "red",
            ValidationSeverity.WARN: "yellow",
            ValidationSeverity.INFO: "dim",
        }
        for issue in report.issues:
            color = _COLOR[issue.severity]
            loc = f"  line {issue.line_number}" if issue.line_number else ""
            console.print(
                f"    [{color}]{issue.severity.value:<5}[/{color}] "
                f"[{issue.code}]{loc}: {issue.message}"
            )
    console.print()


@docs_app.command("preview-chunks")
def docs_preview_chunks(
    path: str = typer.Argument(..., help="Path to the Markdown file."),
    limit: int = typer.Option(0, "--limit", help="Max chunks to display (0 = all)."),
) -> None:
    """Preview how a document will be chunked without writing to stores."""
    import tempfile
    from pathlib import Path as P
    from rag_lab.ingest.cleaner import clean_document
    from rag_lab.chunking.splitter import chunk_document

    doc_path = P(path)
    if not doc_path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cleaned_path = clean_document(doc_path, output_dir=P(tmpdir))
            text = cleaned_path.read_text(encoding="utf-8")
    except Exception as exc:
        console.print(f"[red]Failed to read/clean document: {exc}[/red]")
        raise typer.Exit(1)

    try:
        chunks = chunk_document(text, doc_id=doc_path.stem)
    except Exception as exc:
        console.print(f"[red]Chunking failed: {exc}[/red]")
        raise typer.Exit(1)

    total = len(chunks)
    shown = chunks[:limit] if limit > 0 else chunks

    console.print(
        f"\n[bold cyan]Chunk preview:[/bold cyan] {doc_path.name}  "
        f"[dim]({total} chunks total)[/dim]\n"
    )
    console.rule()

    for i, chunk in enumerate(shown, start=1):
        console.print(
            f"[bold]\\[{i}/{total}][/bold]  [cyan]{chunk.heading_path}[/cyan]"
        )
        console.print(
            f"        tipo=[yellow]{chunk.tipo}[/yellow]  "
            f"tokens=[green]{chunk.n_tokens}[/green]  "
            f"lines={chunk.line_start}-{chunk.line_end}"
        )
        preview = chunk.text[:120].replace("\n", " ")
        if len(chunk.text) > 120:
            preview += "…"
        console.print(f"        [dim]{preview}[/dim]")
        console.print()

    if limit > 0 and total > limit:
        console.print(
            f"[dim]… {total - limit} more chunks. Use --limit 0 to see all.[/dim]"
        )


# ---------------------------------------------------------------------------
# tags commands
# ---------------------------------------------------------------------------


@tags_app.command("list")
def tags_list():
    """List all tags with document counts."""
    from rag_lab.storage.metadata_store import MetadataStore

    store = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
    store.initialize()
    try:
        tags = store.list_tags()
    finally:
        store.close()

    if not tags:
        console.print("[yellow]No tags found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("tag_id", justify="right")
    table.add_column("name")
    table.add_column("doc_count", justify="right")

    for tag in tags:
        table.add_row(str(tag["tag_id"]), tag["name"], str(tag["doc_count"]))

    console.print(table)


@tags_app.command("rename")
def tags_rename(
    old_name: str = typer.Argument(...),
    new_name: str = typer.Argument(...),
):
    """Rename a tag."""
    from rag_lab.storage.metadata_store import MetadataStore

    store = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
    store.initialize()
    try:
        renamed = store.rename_tag(old_name, new_name)
    finally:
        store.close()

    if not renamed:
        console.print(f"[red]Tag not found: {old_name}[/red]")
        raise typer.Exit(1)

    console.print(f"Tag '{old_name}' renamed to '{new_name}'.")


@tags_app.command("delete")
def tags_delete(
    tag_name: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", "-f", help="Delete without confirmation."),
):
    """Remove a tag and all its document associations."""
    from rag_lab.storage.metadata_store import MetadataStore

    store = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
    store.initialize()
    try:
        row = store._conn.execute(
            """
            SELECT t.tag_id, COUNT(dt.doc_id) AS doc_count
            FROM tags t
            LEFT JOIN document_tags dt ON t.tag_id = dt.tag_id
            WHERE t.name = ?
            GROUP BY t.tag_id
            """,
            (tag_name,),
        ).fetchone()

        if row is None:
            console.print(f"[red]Tag not found: {tag_name}[/red]")
            raise typer.Exit(1)

        doc_count = row[1]

        if not force:
            console.print(
                f"Will delete tag '[bold]{tag_name}[/bold]' "
                f"and remove it from {doc_count} document(s)."
            )
            confirmed = typer.confirm("Proceed?")
            if not confirmed:
                console.print("Aborted.")
                raise typer.Exit(0)

        store.delete_tag(tag_name)
    finally:
        store.close()

    console.print(f"Tag '{tag_name}' deleted.")
