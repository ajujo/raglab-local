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
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Filter by dataset_id."),
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
            dataset_id=dataset,
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
        "title", "path", "content_hash", "source_id", "dataset_id",
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


@docs_app.command("set-dataset")
def docs_set_dataset(
    doc_id: str = typer.Argument(...),
    dataset_id: str = typer.Argument(...),
):
    """Set the dataset_id for a document."""
    from rag_lab.storage.metadata_store import MetadataStore

    store = MetadataStore(db_path=DOCDSTORE_SQLITE_PATH)
    store.initialize()
    try:
        if store.get_document(doc_id) is None:
            console.print(f"[red]Document not found: {doc_id}[/red]")
            raise typer.Exit(1)
        store._conn.execute(
            "UPDATE documents SET dataset_id = ?, updated_at = datetime('now') WHERE doc_id = ?",
            (dataset_id, doc_id),
        )
        store._conn.commit()
    finally:
        store.close()

    console.print(f"dataset_id for {doc_id} set to '{dataset_id}'.")


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
