"""Regression tests for Fase 0 bug fixes.

Covers:
- Bug 1: SparseStore auto-loads existing data on __init__ (no overwrite on reinit)
- Bug 2: _delete_chunks_from_chroma uses where filter, not broken query_embedding
- Bug 3: cli_chat._run_query encodes each query variant exactly once (no triple-encoding)
- Bug 4: ingest command does not write chunks.jsonl
- Bug 5: chunk_id is unique across chunks with identical first 100 chars
"""

import inspect
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Bug 1 & 2: SparseStore — auto-load and no overwrite
# ---------------------------------------------------------------------------

class TestSparseStoreAutoLoad:
    def test_auto_loads_existing_file_on_init(self, tmp_path):
        """SparseStore.__init__ must load existing JSON without explicit load() call."""
        from rag_lab.storage.sparse_store import SparseStore

        index_file = tmp_path / "sparse.json"
        index_file.write_text(json.dumps({
            "chunk_aaa": {"sparse": {"token1": 0.9}},
            "chunk_bbb": {"sparse": {"token2": 0.7}},
        }))

        store = SparseStore(storage_path=index_file)
        assert store.count() == 2, (
            "SparseStore must auto-load existing data in __init__; "
            "otherwise each ingest overwrites the full index"
        )

    def test_no_overwrite_when_reinited_on_same_file(self, tmp_path):
        """A second SparseStore instance pointing at the same file must not lose data."""
        from rag_lab.storage.sparse_store import SparseStore

        index_file = tmp_path / "sparse.json"

        # First ingest: create store, add data, save
        store_a = SparseStore(storage_path=index_file)
        store_a.add(["doc1_c1"], [{"token_a": 0.8}])
        store_a.save()

        # Second ingest: new instance (simulates a second doc ingest)
        store_b = SparseStore(storage_path=index_file)
        store_b.add(["doc2_c1"], [{"token_b": 0.6}])
        store_b.save()

        # Third read: must see BOTH docs
        store_verify = SparseStore(storage_path=index_file)
        assert store_verify.count() == 2, (
            "Second ingest must accumulate, not overwrite the previous index"
        )

    def test_init_with_missing_file_starts_empty(self, tmp_path):
        """SparseStore must start empty when file doesn't exist (no crash)."""
        from rag_lab.storage.sparse_store import SparseStore

        store = SparseStore(storage_path=tmp_path / "nonexistent.json")
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Bug 2: _delete_chunks_from_chroma uses where= filter
# ---------------------------------------------------------------------------

class TestDeleteChunksFromChroma:
    def test_uses_where_filter_not_query_embedding(self, tmp_path):
        """_delete_chunks_from_chroma must call collection.delete(where=...) directly."""
        from rag_lab.doc_manager.doc_store import DocManager

        manager = DocManager(db_path=tmp_path / "test.db")

        mock_collection = MagicMock()
        mock_vector_store = MagicMock()
        mock_vector_store._collection = mock_collection

        with patch("rag_lab.doc_manager.doc_store.VectorStore", return_value=mock_vector_store):
            manager._delete_chunks_from_chroma("some_doc")

        mock_collection.delete.assert_called_once_with(where={"doc_id": "some_doc"})

    def test_does_not_call_query_before_delete(self, tmp_path):
        """The old buggy code called vector_store.query() first — must not happen."""
        from rag_lab.doc_manager.doc_store import DocManager

        manager = DocManager(db_path=tmp_path / "test.db")

        mock_collection = MagicMock()
        mock_vector_store = MagicMock()
        mock_vector_store._collection = mock_collection

        with patch("rag_lab.doc_manager.doc_store.VectorStore", return_value=mock_vector_store):
            manager._delete_chunks_from_chroma("some_doc")

        mock_vector_store.query.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 3: cli_chat — no triple-encoding
# ---------------------------------------------------------------------------

class TestCliChatEncoding:
    def test_each_query_variant_encoded_once(self):
        """_run_query must call encode_chunks with exactly 1 item per query variant."""
        # Import inline to avoid heavy module-level imports in test session
        from rag_lab.cli_chat import ChatSession

        captured_calls = []

        def fake_encode(items, batch_size, device):
            captured_calls.append(len(items))
            import numpy as np
            n = len(items)
            dense = np.zeros((n, 4), dtype="float32")
            sparse = {f"fake_{i}": {} for i in range(n)}
            return dense, sparse

        fake_session = MagicMock(spec=ChatSession)
        fake_session.hyde_enabled = False
        fake_session.rewrite_enabled = False
        fake_session.embedding_device = "cpu"
        fake_session.top_k = 5
        fake_session.rerank_top_k = 3
        fake_session.mode = "fast"
        fake_session.active_docs = None

        fake_session.vector_store = MagicMock()
        fake_session.fts_store = MagicMock()
        fake_session.doc_store = MagicMock()
        fake_session.temperature = 0.1
        fake_session._get_doc_ids = lambda: None
        fake_session._filter_results = lambda r: r

        fake_chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": "hello"}]

        with patch("rag_lab.cli_chat.process_query", return_value=[{"text": "test query", "type": "original"}]), \
             patch("rag_lab.cli_chat.encode_chunks", side_effect=fake_encode), \
             patch("rag_lab.cli_chat.hybrid_search", return_value=fake_chunks), \
             patch("rag_lab.cli_chat.build_prompt", return_value=("sys", "usr")), \
             patch("rag_lab.cli_chat.generate_response", return_value="answer"), \
             patch("rag_lab.cli_chat.verify_and_score", return_value=MagicMock(
                 response="answer",
                 get_warnings=lambda: [],
                 format_verification_block=lambda: "",
                 score_result=MagicMock(final_score=0.8, confidence_level=MagicMock(value="HIGH")),
             )):
            ChatSession._run_query(fake_session, "test query")

        assert len(captured_calls) == 1, "encode_chunks must be called once per query variant"
        assert captured_calls[0] == 1, (
            f"encode_chunks must receive exactly 1 item per call, got {captured_calls[0]}. "
            "Triple-encoding bug: old code passed [original, hyde, original]."
        )


# ---------------------------------------------------------------------------
# Bug 4: ingest does not write chunks.jsonl
# ---------------------------------------------------------------------------

class TestIngestNoChunksJsonl:
    def test_ingest_source_does_not_write_chunks_jsonl(self):
        """The ingest command must not append to chunks.jsonl.

        chunks.jsonl grows unboundedly and duplicates what DocStore already stores.
        """
        import rag_lab.cli as cli_module
        source = inspect.getsource(cli_module.ingest)
        assert "chunks.jsonl" not in source, (
            "ingest() must not write chunks.jsonl — DocStore is the source of truth. "
            "Remove the open(..., 'a') block that appends to chunks.jsonl."
        )


# ---------------------------------------------------------------------------
# Bug 5: chunk_id uniqueness — same 100-char prefix, different position
# ---------------------------------------------------------------------------

class TestChunkIdUniqueness:
    def test_different_positions_yield_different_ids(self):
        """Two chunks from different line positions must have different chunk_ids."""
        from rag_lab.chunking.splitter import _make_chunk_id

        text = "A" * 300  # long repeated content — first 200 chars are identical
        id1 = _make_chunk_id("doc_x", 100, text)
        id2 = _make_chunk_id("doc_x", 200, text)
        assert id1 != id2, (
            "chunk_id must include line_start so that two chunks with "
            "the same text prefix but different positions don't collide"
        )

    def test_different_docs_yield_different_ids(self):
        """Same text at same position in different docs must have different IDs."""
        from rag_lab.chunking.splitter import _make_chunk_id

        text = "identical content"
        id1 = _make_chunk_id("doc_alpha", 10, text)
        id2 = _make_chunk_id("doc_beta", 10, text)
        assert id1 != id2

    def test_same_doc_position_text_yields_same_id(self):
        """chunk_id must be stable across re-ingests (deterministic)."""
        from rag_lab.chunking.splitter import _make_chunk_id

        text = "stable content"
        assert _make_chunk_id("doc_x", 5, text) == _make_chunk_id("doc_x", 5, text)

    def test_chunk_document_ids_are_unique(self, sample_text):
        """chunk_document() must produce chunks with unique chunk_ids."""
        from rag_lab.chunking.splitter import chunk_document

        chunks = chunk_document(sample_text, doc_id="test_doc")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), (
            f"chunk_ids must be unique; found duplicates among: {ids}"
        )
