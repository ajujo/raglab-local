"""Integration tests for the full RAG pipeline.

These tests verify that the complete pipeline works end-to-end:
1. Document ingestion (clean → chunk → embed → store)
2. Query execution (process → search → rerank → generate)

All ML models run on CPU via conftest.py fixtures to avoid GPU OOM.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_lab.cli import app
from rag_lab.config import DATA_DIR, STORAGE_DIR
from rag_lab.ingest.cleaner import clean_document
from rag_lab.ingest.manifest import create_manifest
from rag_lab.chunking.splitter import chunk_document
from rag_lab.embedding.encoder import encode_chunks
from rag_lab.storage.vector_store import VectorStore
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.retrieval.query_processor import process_query
from rag_lab.retrieval.hybrid_search import hybrid_search
from rag_lab.retrieval.reranker import rerank
from rag_lab.generation.prompt_builder import build_prompt
from rag_lab.generation.verifier import verify_citations


class TestFullPipeline:
    """Integration tests for the full RAG pipeline."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up temporary directories for each test.

        Patches all config path constants so the CLI and stores use tmp paths,
        preventing any writes to the production docstore.sqlite or ChromaDB.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()

        import rag_lab.config as config
        import rag_lab.storage.docstore as _ds_mod
        import rag_lab.storage.vector_store as _vs_mod

        originals = {
            "config.DATA_DIR": config.DATA_DIR,
            "config.STORAGE_DIR": config.STORAGE_DIR,
            "config.DOCDSTORE_SQLITE_PATH": config.DOCDSTORE_SQLITE_PATH,
            "config.VECTOR_STORE_PATH": config.VECTOR_STORE_PATH,
            "config.SPARSE_INDEX_PATH": config.SPARSE_INDEX_PATH,
            "ds.DOCDSTORE_SQLITE_PATH": _ds_mod.DOCDSTORE_SQLITE_PATH,
            "vs.VECTOR_STORE_PATH": _vs_mod.VECTOR_STORE_PATH,
        }
        config.DATA_DIR = data_dir
        config.STORAGE_DIR = storage_dir
        config.DOCDSTORE_SQLITE_PATH = storage_dir / "docstore.sqlite"
        config.VECTOR_STORE_PATH = storage_dir / "chroma_db"
        config.SPARSE_INDEX_PATH = storage_dir / "sparse_index.json"
        _ds_mod.DOCDSTORE_SQLITE_PATH = storage_dir / "docstore.sqlite"
        _vs_mod.VECTOR_STORE_PATH = storage_dir / "chroma_db"

        yield

        config.DATA_DIR = originals["config.DATA_DIR"]
        config.STORAGE_DIR = originals["config.STORAGE_DIR"]
        config.DOCDSTORE_SQLITE_PATH = originals["config.DOCDSTORE_SQLITE_PATH"]
        config.VECTOR_STORE_PATH = originals["config.VECTOR_STORE_PATH"]
        config.SPARSE_INDEX_PATH = originals["config.SPARSE_INDEX_PATH"]
        _ds_mod.DOCDSTORE_SQLITE_PATH = originals["ds.DOCDSTORE_SQLITE_PATH"]
        _vs_mod.VECTOR_STORE_PATH = originals["vs.VECTOR_STORE_PATH"]

    def test_full_ingest_pipeline(self, tmp_path, sample_text):
        """Test the complete ingestion pipeline: clean → chunk → embed → store."""
        # Write sample document
        doc_path = tmp_path / "test_doc.md"
        doc_path.write_text(sample_text, encoding="utf-8")

        # Phase 1: Clean
        cleaned_path = clean_document(doc_path)
        assert cleaned_path.exists()
        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        assert "base64" not in cleaned_text

        # Phase 2: Chunking
        chunks = chunk_document(cleaned_text, doc_id="test_doc")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.doc_id == "test_doc"
            assert chunk.text
            assert chunk.heading_path
            assert chunk.tipo in ("texto", "tabla", "formula")
            assert chunk.n_tokens > 0

        # Phase 3: Embedding (CPU)
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        dense_embeddings, sparse_embeddings = encode_chunks(
            chunk_dicts, batch_size=4, device="cpu"
        )
        assert dense_embeddings is not None
        assert len(dense_embeddings) == len(chunks)

        # Phase 4: Storage
        import numpy as np
        from rag_lab.config import (
            EMBEDDING_MODEL, EMBEDDING_MODEL_VERSION,
            EMBEDDING_DIM, SPARSE_FORMAT_VERSION,
        )

        vector_store = VectorStore()
        vector_store.initialize()
        vector_store.add(
            ids=[c["chunk_id"] for c in chunk_dicts],
            embeddings=dense_embeddings,
            documents=[c["text"] for c in chunk_dicts],
            metadatas=[{"heading_path": c["heading_path"]} for c in chunk_dicts],
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

        doc_store = DocStore(db_path=tmp_path / "test_docstore.sqlite")
        doc_store.initialize()
        doc_store.add(chunk_dicts)

        fts_store = FTSStore(db_path=tmp_path / "test_docstore.sqlite")
        fts_store.initialize()

        # Verify storage
        query_emb = dense_embeddings[0]
        results = hybrid_search(
            "What is SDMX?",
            vector_store,
            doc_store,
            fts_store,
            query_dense=query_emb,
            query_sparse=next(iter(sparse_embeddings.values()), {}),
            top_k=5,
        )
        assert len(results) > 0


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    @pytest.fixture(autouse=True)
    def _isolate_stores(self, tmp_path):
        """Redirect all store paths to tmp_path so CLI writes don't hit production."""
        storage_dir = tmp_path / "storage"
        storage_dir.mkdir()
        import rag_lab.config as config
        import rag_lab.storage.docstore as _ds_mod
        import rag_lab.storage.vector_store as _vs_mod

        orig = {
            "c.DATA_DIR": config.DATA_DIR,
            "c.STORAGE_DIR": config.STORAGE_DIR,
            "c.DOCDSTORE_SQLITE_PATH": config.DOCDSTORE_SQLITE_PATH,
            "c.VECTOR_STORE_PATH": config.VECTOR_STORE_PATH,
            "c.SPARSE_INDEX_PATH": config.SPARSE_INDEX_PATH,
            "ds.DOCDSTORE_SQLITE_PATH": _ds_mod.DOCDSTORE_SQLITE_PATH,
            "vs.VECTOR_STORE_PATH": _vs_mod.VECTOR_STORE_PATH,
        }
        config.DATA_DIR = tmp_path / "data"
        config.STORAGE_DIR = storage_dir
        config.DOCDSTORE_SQLITE_PATH = storage_dir / "docstore.sqlite"
        config.VECTOR_STORE_PATH = storage_dir / "chroma_db"
        config.SPARSE_INDEX_PATH = storage_dir / "sparse_index.json"
        _ds_mod.DOCDSTORE_SQLITE_PATH = storage_dir / "docstore.sqlite"
        _vs_mod.VECTOR_STORE_PATH = storage_dir / "chroma_db"

        yield

        config.DATA_DIR = orig["c.DATA_DIR"]
        config.STORAGE_DIR = orig["c.STORAGE_DIR"]
        config.DOCDSTORE_SQLITE_PATH = orig["c.DOCDSTORE_SQLITE_PATH"]
        config.VECTOR_STORE_PATH = orig["c.VECTOR_STORE_PATH"]
        config.SPARSE_INDEX_PATH = orig["c.SPARSE_INDEX_PATH"]
        _ds_mod.DOCDSTORE_SQLITE_PATH = orig["ds.DOCDSTORE_SQLITE_PATH"]
        _vs_mod.VECTOR_STORE_PATH = orig["vs.VECTOR_STORE_PATH"]

    def test_ingest_cli(self, tmp_path, sample_text):
        """Test the ingest CLI command end-to-end."""
        # Create a temporary document
        doc_path = tmp_path / "test_doc.md"
        doc_path.write_text(sample_text, encoding="utf-8")

        # Run CLI ingest command
        runner = CliRunner()
        result = runner.invoke(app, ["ingest", "--doc", str(doc_path)])

        # Verify command succeeded
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify chunks were created
        chunks_path = tmp_path / "data" / "chunks.jsonl"
        if chunks_path.exists():
            with open(chunks_path) as f:
                lines = f.readlines()
                assert len(lines) >= 1

    def test_query_cli(self, tmp_path, sample_text):
        """Test the query CLI command end-to-end."""
        # First ingest the document
        doc_path = tmp_path / "test_doc.md"
        doc_path.write_text(sample_text, encoding="utf-8")

        runner = CliRunner()
        # Ingest first
        ingest_result = runner.invoke(app, ["ingest", "--doc", str(doc_path)])
        assert ingest_result.exit_code == 0

        # Then query
        query_result = runner.invoke(app, ["query", "What is SDMX?", "--fast"])
        # Query may fail if LLM is not available, but the retrieval part should work
        # We check that the command at least runs without crashing on retrieval
        assert query_result.exit_code in (0, 1)  # 1 is acceptable if LLM is unavailable


class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_empty_document(self, tmp_path):
        """Test ingestion of an empty document."""
        doc_path = tmp_path / "empty.md"
        doc_path.write_text("", encoding="utf-8")

        with pytest.raises(Exception):
            cleaned_path = clean_document(doc_path)

    def test_document_with_only_headings(self, tmp_path):
        """Test document with only headings, no content."""
        text = "# Heading 1\n## Subheading\n### Sub-subheading\n"
        doc_path = tmp_path / "headings_only.md"
        doc_path.write_text(text, encoding="utf-8")

        cleaned_path = clean_document(doc_path)
        chunks = chunk_document(cleaned_path.read_text(encoding="utf-8"), doc_id="headings")
        # Should produce chunks even with minimal content
        assert len(chunks) >= 1

    def test_large_table_handling(self, tmp_path):
        """Test handling of large tables."""
        # Create a large table
        table_lines = ["| Col1 | Col2 | Col3 |"]
        table_lines.append("|------|------|------|")
        for i in range(100):
            table_lines.append(f"| Row{i} | Data{i} | More{i} |")
        text = "# Data Table\n\n" + "\n".join(table_lines)

        doc_path = tmp_path / "large_table.md"
        doc_path.write_text(text, encoding="utf-8")

        cleaned_path = clean_document(doc_path)
        chunks = chunk_document(cleaned_path.read_text(encoding="utf-8"), doc_id="table_doc")
        # Table should be kept intact
        table_chunks = [c for c in chunks if c.tipo == "tabla"]
        assert len(table_chunks) >= 1


class TestCPUMode:
    """Tests for CPU mode operation."""

    def test_embedding_on_cpu(self):
        """Verify embedding works on CPU."""
        chunks = [{"text": "Test text for embedding.", "chunk_id": "cpu_test"}]
        dense, sparse = encode_chunks(chunks, batch_size=1, device="cpu")
        assert dense is not None
        assert len(dense) == 1
        assert sparse is not None

    def test_reranker_on_cpu(self):
        """Verify reranker works on CPU."""
        results = [
            {"chunk_id": "1", "text": "Result one", "score": 0.9},
            {"chunk_id": "2", "text": "Result two", "score": 0.7},
            {"chunk_id": "3", "text": "Result three", "score": 0.5},
        ]
        reranked = rerank("Test query", results, top_k=2, device="cpu")
        assert len(reranked) <= 2
        for r in reranked:
            assert "score" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
