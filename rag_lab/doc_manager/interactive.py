"""Interactive CLI for the document manager.

Provides an interactive menu-based interface for managing documents.
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def interactive_mode():
    """Run the interactive menu loop."""
    from rag_lab.config import DATA_DIR
    from rag_lab.doc_manager.doc_store import DocManager
    from rag_lab.storage.docstore import DocStore

    console = Console()
    manager = DocManager()

    def show_banner():
        """Display the application banner."""
        banner = Text()
        banner.append("📚 ", style="bold cyan")
        banner.append("RAG Document Manager", style="bold cyan")
        console.print(Panel(banner, expand=False, border_style="cyan"))

    def show_menu():
        """Display the main menu."""
        console.print("\n[bold]Seleccione una opción:[/bold]")
        console.print("  [1] 📋 Listar documentos")
        console.print("  [2] ➕ Añadir documento")
        console.print("  [3] 🗑️  Borrar documento")
        console.print("  [4] 🏷️  Etiquetar documento")
        console.print("  [5] 🔍 Buscar documentos")
        console.print("  [6] 📊 Migrar documentos existentes")
        console.print("  [7] ℹ️  Info de documento")
        console.print("  [8] 📁 Ver colecciones/etiquetas")
        console.print("  [9] 💣 Borrar todos los documentos")
        console.print("  [0] Salir\n")

    def select_document(prompt_msg: str = "Seleccione un documento (número):") -> dict | None:
        """Display documents and let user select one by number."""
        documents = manager.list_documents()
        if not documents:
            console.print("[bold red]No hay documentos registrados.[/bold red]")
            return None

        console.print("\n[bold cyan]Documentos disponibles:[/bold cyan]")
        for i, doc in enumerate(documents, 1):
            tags_str = f" [{doc['tags']}]" if doc['tags'] else ""
            console.print(
                f"  [{i}] [bold]{doc['doc_id']}[/bold] | {doc['chunk_count']} chunks | "
                f"{doc['size']} bytes {tags_str}"
            )

        valid = [str(i) for i in range(1, len(documents) + 1)]
        while True:
            choice = input(f"{prompt_msg} ({', '.join(valid)} o 'volver'): ")
            if choice.lower() in ("volver", "v", "n"):
                return None
            if choice in valid:
                return documents[int(choice) - 1]
            console.print(f"[bold red]Opción inválida. Elige: {', '.join(valid)} o 'volver'[/bold red]")

    def action_list():
        """List all documents."""
        documents = manager.list_documents()
        if not documents:
            console.print("[bold yellow]No hay documentos registrados.[/bold yellow]")
            return

        table = Table(title="📚 Documentos Ingestados", show_header=True)
        table.add_column("#", style="cyan")
        table.add_column("Doc ID", style="bold")
        table.add_column("Chunks", style="green")
        table.add_column("Tamaño", style="green")
        table.add_column("Etiquetas", style="magenta")

        for i, doc in enumerate(documents, 1):
            table.add_row(
                str(i),
                doc["doc_id"],
                str(doc["chunk_count"]),
                f"{doc['size']}B",
                doc["tags"] or "(ninguna)",
            )
        console.print(table)

    def action_add():
        """Add a new document."""
        path = input("Ruta al archivo a añadir: ")
        file_path = Path(path)

        if not file_path.exists():
            console.print(f"[bold red]Archivo no encontrado: {path}[/bold red]")
            return

        added = manager.add_document(file_path)
        if added:
            try:
                ds = DocStore()
                ds.initialize()
                chunk_count = ds.count_chunks(file_path.stem)
                manager.update_chunk_count(file_path.stem, chunk_count)
                ds.close()
            except Exception as e:
                console.print(f"[bold yellow]No se pudo actualizar chunk count: {e}[/bold yellow]")
            console.print(f"[bold green]✅ Documento añadido: {file_path.name}[/bold green]")
        else:
            console.print(f"[bold yellow]⚠️ Documento ya existe (hash duplicado)[/bold yellow]")

    def action_delete():
        """Delete a document."""
        doc = select_document("¿Qué documento quieres borrar?")
        if doc is None:
            return

        while True:
            confirm = input(f"¿Seguro que quieres borrar '{doc['doc_id']}'? (s/n/volver): ")
            if confirm.lower() == "s":
                deleted = manager.delete_document(doc["doc_id"])
                if deleted:
                    console.print(f"[bold green]✅ Documento borrado: {doc['doc_id']}[/bold green]")
                else:
                    console.print(f"[bold red]Error al borrar el documento.[/bold red]")
                break
            elif confirm.lower() in ("volver", "v", "n"):
                console.print("[bold yellow]Se canceló la eliminación.[/bold yellow]")
                return
            else:
                console.print("[bold red]Respuesta inválida. Escribe 's' para sí, 'n' para no, o 'volver' para cancelar.[/bold red]")

    def action_delete_all():
        """Delete all documents with double confirmation."""
        documents = manager.list_documents()
        if not documents:
            console.print("[bold yellow]No hay documentos para borrar.[/bold yellow]")
            return

        console.print(f"[bold red]⚠️ Atención: vas a borrar {len(documents)} documento(s):[/bold red]")
        for doc in documents:
            console.print(f"  • {doc['doc_id']}")

        confirm = input("\nEscribe 'BORRAR' para confirmar: ")
        if confirm == "BORRAR":
            for doc in documents:
                manager.delete_document(doc["doc_id"])
            console.print(f"[bold green]✅ Se borraron {len(documents)} documento(s).[/bold green]")
        else:
            console.print("[bold yellow]Se canceló la eliminación masiva.[/bold yellow]")

    def action_tag():
        """Tag a document."""
        doc = select_document("¿Qué documento quieres etiquetar?")
        if not doc:
            return

        tag_name = input("Nombre de la etiqueta: ")
        success = manager.assign_tag(doc["doc_id"], tag_name)
        if success:
            console.print(f"[bold green]✅ Etiqueta '{tag_name}' asignada a {doc['doc_id']}[/bold green]")
        else:
            console.print(f"[bold red]Error al asignar etiqueta.[/bold red]")

    def action_search():
        """Search documents."""
        query = input("Buscar (nombre, ruta o etiqueta): ")
        results = manager.search_documents(query)

        if not results:
            console.print(f"[bold yellow]No se encontraron documentos para: {query}[/bold yellow]")
            return

        console.print(f"\n[bold cyan]🔍 Resultados ({len(results)})[/bold cyan]")
        for i, doc in enumerate(results, 1):
            tags_str = f" [{doc['tags']}]" if doc['tags'] else ""
            console.print(
                f"  [{i}] [bold]{doc['doc_id']}[/bold] | {doc['chunk_count']} chunks | "
                f"{doc['size']} bytes {tags_str}"
            )

    def action_migrate():
        """Migrate existing documents."""
        manifest_path = Path(DATA_DIR) / "ingested.jsonl"
        if not manifest_path.exists():
            console.print(f"[bold red]Manifiesto no encontrado: {manifest_path}[/bold red]")
            return

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

                try:
                    ds = DocStore()
                    ds.initialize()
                    chunk_count = ds.count_chunks(doc_id)
                    manager.update_chunk_count(doc_id, chunk_count)
                    ds.close()
                except Exception as e:
                    console.print(f"[bold red]Error al contar chunks para {doc_id}: {e}[/bold red]")
                skipped += 1
            else:
                console.print(f"[bold yellow]⚠️ Archivo no encontrado: {file_path}[/bold yellow]")
                skipped += 1

        console.print(f"\n[bold green]✅ Migrados: {migrated}[/bold green]")
        console.print(f"[bold yellow]Actualizados/Omitidos: {skipped}[/bold yellow]")

    def action_info():
        """Show document info."""
        doc = select_document("¿De qué documento quieres ver la info?")
        if not doc:
            return

        console.print(f"\n[bold cyan]📄 Info: {doc['doc_id']}[/bold cyan]")
        console.print(f"  Ruta      : {doc['path']}")
        console.print(f"  Hash      : {doc['hash']}")
        console.print(f"  Tamaño    : {doc['size']} bytes")
        console.print(f"  Chunks    : {doc['chunk_count']}")
        console.print(f"  Ingestado : {doc['ingested_at']}")
        tags = manager.get_tags(doc['doc_id'])
        console.print(f"  Etiquetas : {', '.join(tags) if tags else '(ninguna)'}")

    def action_collections():
        """List all tags/collections."""
        tags = manager.list_all_tags()
        if not tags:
            console.print("[bold yellow]No hay etiquetas creadas.[/bold yellow]")
            return

        console.print(f"\n[bold cyan]🏷️ Etiquetas ({len(tags)})[/bold cyan]")
        for tag in tags:
            console.print(f"  • {tag}")

    # Main loop
    show_banner()

    valid_choices = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    while True:
        show_menu()
        choice = input("Elige una opción (0-9): ")
        if choice not in valid_choices:
            console.print(f"[bold red]Opción inválida: {choice}[/bold red]")
            continue

        if choice == "0":
            console.print("[bold green]👋 ¡Hasta luego![/bold green]")
            sys.exit(0)
        elif choice == "1":
            action_list()
        elif choice == "2":
            action_add()
        elif choice == "3":
            action_delete()
        elif choice == "4":
            action_tag()
        elif choice == "5":
            action_search()
        elif choice == "6":
            action_migrate()
        elif choice == "7":
            action_info()
        elif choice == "8":
            action_collections()
        elif choice == "9":
            action_delete_all()

        console.print("\n" + "=" * 60)


if __name__ == "__main__":
    interactive_mode()
