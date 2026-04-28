"""CLI interface for the RAG-Lab system.

Provides commands for ingesting documents and querying the system.
"""

import json
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console

from rag_lab.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP,
    DATA_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    SOURCES,
    STORAGE_DIR,
)
from rag_lab.exceptions import RAGLabError, RetrievalError, LLMConnectionError
from rag_lab.ingest.cleaner import clean_document
from rag_lab.ingest.manifest import create_manifest
from rag_lab.chunking.splitter import chunk_document
from rag_lab.embedding.encoder import encode_chunks, load_embedding_model
from rag_lab.storage.vector_store import VectorStore
from rag_lab.storage.sparse_store import SparseStore
from rag_lab.storage.docstore import DocStore
from rag_lab.retrieval.query_processor import process_query
from rag_lab.retrieval.hybrid_search import hybrid_search
from rag_lab.retrieval.reranker import rerank
from rag_lab.generation.prompt_builder import build_prompt
from rag_lab.generation.llm_client import generate_response
from rag_lab.generation.verifier import verify_citations
from rag_lab.logging_config import setup_logging

app = typer.Typer(
    name="rag-lab",
    help="RAG system for SDMX Technical Notes",
    add_completion=True,
)

console = Console()


@app.command()
def ingest(
    doc: str = typer.Option(
        None,
        "--doc",
        help="Path to a single source document. If not specified, ingests all SOURCES.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-ingestion even if already ingested.",
    ),
    cpu_embedding: bool = typer.Option(
        False,
        "--cpu-embedding",
        help="Run embedding model on CPU to free GPU VRAM.",
    ),
) -> None:
    """Ingest one or more documents: clean, chunk, embed, and store."""
    setup_logging("INFO")
    logger = logging.getLogger("rag_lab")

    # Determine embedding device
    device = "cpu" if cpu_embedding else EMBEDDING_DEVICE

    # Decide which documents to process
    if doc is not None:
        # Single document mode
        paths_to_process = [Path(doc)]
    else:
        # Multi-document mode: process all SOURCES
        paths_to_process = list(SOURCES)

    total_chunks = 0
    for source_path in paths_to_process:
        console.print(f"[bold cyan]📥 Ingesting: {source_path.name}[/bold cyan]")

        if not source_path.exists():
            logger.warning(f"Source file not found: {source_path} — skipping")
            continue

        # Phase 1: Clean document
        cleaned_path = clean_document(source_path)
        create_manifest(source_path, cleaned_path, force=force)

        # Phase 2: Chunking
        logger.info("Starting chunking...")
        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        chunks = chunk_document(
            cleaned_text,
            doc_id=source_path.stem,
            max_tokens=CHUNK_MAX_TOKENS,
            overlap=CHUNK_OVERLAP,
        )
        logger.info(f"Created {len(chunks)} chunks from {source_path.name}")
        total_chunks += len(chunks)

        # Save chunks to JSONL (append mode for multi-doc)
        chunks_path = DATA_DIR / "chunks.jsonl"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        # Phase 3: Embedding
        logger.info(f"Generating embeddings on {device}...")
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        dense_embeddings, sparse_embeddings = encode_chunks(
            chunk_dicts,
            batch_size=EMBEDDING_BATCH_SIZE,
            device=device,
        )

        # Phase 4: Storage
        logger.info("Storing embeddings...")
        vector_store = VectorStore()
        vector_store.initialize()
        vector_store.add(
            ids=[c.get("chunk_id", "") for c in chunk_dicts],
            embeddings=dense_embeddings,
            documents=[c.get("text", "") for c in chunk_dicts],
            metadatas=[{"heading_path": c.get("heading_path", ""), "doc_id": c.get("doc_id", "")} for c in chunk_dicts],
        )

        sparse_store = SparseStore()
        sparse_store.add(
            ids=[c.get("chunk_id", "") for c in chunk_dicts],
            sparse_vectors=list(sparse_embeddings.values()),
        )
        sparse_store.save()

        doc_store = DocStore()
        doc_store.add(chunk_dicts)

        console.print(f"[bold green]✅ Ingested {len(chunks)} chunks from {source_path.name}[/bold green]")

    console.print(f"[bold green]🎉 Total: {total_chunks} chunks ingested[/bold green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask."),
    hyde: bool = typer.Option(False, "--hyde", help="Enable HyDE."),
    fast: bool = typer.Option(False, "--fast", help="Skip reranking."),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve."),
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

    # Process query
    queries = process_query(question, use_hyde=hyde)

    # Get embeddings for all query variants
    # encode_chunks returns (dense_embeddings: np.ndarray, sparse_embeddings: Dict)
    all_query_data = []
    for q in queries:
        dense_emb, sparse_dict = encode_chunks([{"text": q["text"]}], batch_size=1, device=emb_device)
        # dense_emb is shape (1, 1024), take first element
        query_dense = dense_emb[0]
        # sparse_dict maps chunk_id -> {token_idx: weight}, get first (and only) entry
        query_sparse = next(iter(sparse_dict.values()), {}) if sparse_dict else {}
        all_query_data.append((query_dense, query_sparse))

    # Perform hybrid search
    vector_store = VectorStore()
    sparse_store = SparseStore()
    doc_store = DocStore()
    vector_store.initialize()
    sparse_store.load()

    # Search with each query variant
    all_results = []
    for query_dense, query_sparse in all_query_data:
        results = hybrid_search(
            question,
            vector_store,
            sparse_store,
            doc_store,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=top_k * 2,
        )
        all_results.extend(results)

    # Deduplicate by chunk_id
    seen = set()
    unique_results = []
    for r in all_results:
        if r.get("chunk_id") not in seen:
            seen.add(r.get("chunk_id"))
            unique_results.append(r)

    # Rerank if not in fast mode
    if not fast and unique_results:
        unique_results = rerank(
            question,
            unique_results[:20],
            top_k=min(top_k * 2, len(unique_results)),
            device=rerank_device,
        )

    # Generate response
    if unique_results:
        system_prompt, user_prompt = build_prompt(question, unique_results[:RERANK_TOP_K])
        try:
            response = generate_response(system_prompt, user_prompt)
            if not response:
                console.print("\n[bold yellow]⚠️ El LLM no devolvió respuesta.[/bold yellow]")
            else:
                verified = verify_citations(response, unique_results)
                console.print(f"\n[bold green]🤖 Response:[/bold green]\n{verified}")
        except LLMConnectionError as e:
            console.print(f"[bold red]LLM Error:[/bold red] {e}")
        except RAGLabError as e:
            console.print(f"[bold red]RAG-Lab Error:[/bold red] {e}")
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
    else:
        console.print("[bold yellow]⚠️ No results found.[/bold yellow]")


if __name__ == "__main__":
    app()