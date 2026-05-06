"""CLI for the document manager.

Provides commands to manage ingested documents, including listing,
adding, deleting, tagging, and searching.
"""

import json
import typer
from pathlib import Path
from rich.console import Console

from rag_lab.config import DATA_DIR
from rag_lab.doc_manager.doc_store import DocManager

app = typer.Typer(
    name="doc-manager",
    help="Manage ingested documents in the RAG system.",
)

console = Console()


@app.command()
def list():
    """List all ingested documents with their metadata."""
    manager = DocManager()
    documents = manager.list_documents()

    if not documents:
        console.print("[bold yellow]No documents found.[/bold yellow]")
        return

    console.print(f"\n[bold cyan]📚 Ingested Documents ({len(documents)})[/bold cyan]\n")
    for doc in documents:
        tags_str = f" [{doc['tags']}]" if doc['tags'] else ""
        console.print(
            f"  [bold]{doc['doc_id']}[/bold] | {doc['chunk_count']} chunks | "
            f"{doc['size']} bytes | {doc['hash']} {tags_str}"
        )


@app.command()
def add(file_path: str):
    """Add a new document to the RAG system."""
    path = Path(file_path)
    manager = DocManager()

    if not path.exists():
        console.print(f"[bold red]File not found: {file_path}[/bold red]")
        raise typer.Exit(1)

    added = manager.add_document(path)
    if added:
        console.print(f"[bold green]✅ Document added: {path.name}[/bold green]")
    else:
        console.print(f"[bold yellow]⚠️ Document already exists (duplicate hash)[/bold yellow]")


@app.command()
def delete(doc_id: str):
    """Delete a document and all its chunks from the system."""
    manager = DocManager()
    deleted = manager.delete_document(doc_id)

    if deleted:
        console.print(f"[bold green]✅ Document deleted: {doc_id}[/bold green]")
    else:
        console.print(f"[bold red]Document not found: {doc_id}[/bold red]")


@app.command()
def tag(doc_id: str, tag_name: str):
    """Assign a tag to a document."""
    manager = DocManager()
    success = manager.assign_tag(doc_id, tag_name)

    if success:
        console.print(f"[bold green]✅ Tag '{tag_name}' assigned to {doc_id}[/bold green]")
    else:
        console.print(f"[bold red]Document not found: {doc_id}[/bold red]")


@app.command()
def untag(doc_id: str, tag_name: str):
    """Remove a tag from a document."""
    manager = DocManager()
    success = manager.remove_tag(doc_id, tag_name)

    if success:
        console.print(f"[bold green]✅ Tag '{tag_name}' removed from {doc_id}[/bold green]")
    else:
        console.print(f"[bold yellow]Tag not found or document doesn't exist[/bold yellow]")


@app.command()
def search(query: str):
    """Search documents by name, path, or tags."""
    manager = DocManager()
    results = manager.search_documents(query)

    if not results:
        console.print(f"[bold yellow]No documents match: {query}[/bold yellow]")
        return

    console.print(f"\n[bold cyan]🔍 Search Results ({len(results)})[/bold cyan]\n")
    for doc in results:
        tags_str = f" [{doc['tags']}]" if doc['tags'] else ""
        console.print(
            f"  [bold]{doc['doc_id']}[/bold] | {doc['chunk_count']} chunks | "
            f"{doc['size']} bytes {tags_str}"
        )


@app.command()
def collections():
    """List all tags/collections in the system."""
    manager = DocManager()
    tags = manager.list_all_tags()

    if not tags:
        console.print("[bold yellow]No tags found.[/bold yellow]")
        return

    console.print(f"\n[bold cyan]🏷️ Tags/Collections ({len(tags)})[/bold cyan]\n")
    for tag in tags:
        console.print(f"  • {tag}")


@app.command()
def info(doc_id: str):
    """Show detailed information about a document."""
    manager = DocManager()
    doc = manager.get_document(doc_id)

    if not doc:
        console.print(f"[bold red]Document not found: {doc_id}[/bold red]")
        return

    console.print(f"\n[bold cyan]📄 Document Info: {doc_id}[/bold cyan]\n")
    console.print(f"  Path      : {doc['path']}")
    console.print(f"  Hash      : {doc['hash']}")
    console.print(f"  Size      : {doc['size']} bytes")
    console.print(f"  Chunks    : {doc['chunk_count']}")
    console.print(f"  Ingested  : {doc['ingested_at']}")
    console.print(f"  Tags      : {', '.join(doc['tags']) if doc['tags'] else '(none)'}")


@app.command()
def migrate():
    """Migrate existing documents from ingested.jsonl to the doc manager."""
    manager = DocManager()
    manifest_path = Path(DATA_DIR) / "ingested.jsonl"

    if not manifest_path.exists():
        console.print(f"[bold red]Manifest not found: {manifest_path}[/bold red]")
        raise typer.Exit(1)

    migrated = 0
    skipped = 0
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        doc_id = entry["doc_id"]
        file_path = Path(entry["path"])

        if file_path.exists():
            added = manager.add_document(file_path)
            if added:
                migrated += 1

            # Always update chunk count from docstore
            from rag_lab.storage.docstore import DocStore
            try:
                ds = DocStore()
                ds.initialize()
                chunk_count = ds.count_chunks(doc_id)
                manager.update_chunk_count(doc_id, chunk_count)
                ds.close()
            except Exception as e:
                console.print(f"[bold red]Failed to get chunk count for {doc_id}: {e}[/bold red]")
            skipped += 1
        else:
            console.print(f"[bold yellow]⚠️ File not found: {file_path}[/bold yellow]")
            skipped += 1

    console.print(f"\n[bold green]✅ Migrated: {migrated}[/bold green]")
    console.print(f"[bold yellow]Updated/Skipped: {skipped}[/bold yellow]\n")


if __name__ == "__main__":
    app()
