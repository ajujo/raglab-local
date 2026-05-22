"""Tests for build_reranker_text and reranker heading-context integration (v1.10)."""

from unittest.mock import MagicMock, patch

import pytest

from rag_lab.retrieval.reranker import build_reranker_text, rerank, reset_reranker_cache


# ---------------------------------------------------------------------------
# build_reranker_text
# ---------------------------------------------------------------------------

class TestBuildRerankerText:
    def _chunk(self, text="content", doc_id="SDMX_Notes", heading_path="## Section"):
        return {"text": text, "doc_id": doc_id, "heading_path": heading_path}

    def test_full_context_format(self):
        chunk = self._chunk()
        result = build_reranker_text(chunk, use_heading_context=True)
        assert result.startswith("Document: SDMX_Notes\nSection: ## Section\n\n")
        assert result.endswith("content")

    def test_no_heading_path(self):
        chunk = {"text": "some text", "doc_id": "MyDoc", "heading_path": ""}
        result = build_reranker_text(chunk, use_heading_context=True)
        assert result == "Document: MyDoc\n\nsome text"
        assert "Section:" not in result

    def test_no_doc_id(self):
        chunk = {"text": "body", "doc_id": "", "heading_path": "## Methods"}
        result = build_reranker_text(chunk, use_heading_context=True)
        assert result == "Section: ## Methods\n\nbody"
        assert "Document:" not in result

    def test_no_doc_id_no_heading_path(self):
        """Falls back to bare text when no structural fields available."""
        chunk = {"text": "bare text", "doc_id": "", "heading_path": ""}
        result = build_reranker_text(chunk, use_heading_context=True)
        assert result == "bare text"

    def test_use_heading_context_false(self):
        """use_heading_context=False returns only chunk text — no header."""
        chunk = self._chunk()
        result = build_reranker_text(chunk, use_heading_context=False)
        assert result == "content"
        assert "Document:" not in result
        assert "Section:" not in result

    def test_missing_keys_graceful(self):
        """Chunk with no doc_id/heading_path keys at all — no KeyError."""
        chunk = {"text": "just text"}
        result = build_reranker_text(chunk, use_heading_context=True)
        assert result == "just text"

    def test_none_fields_treated_as_empty(self):
        chunk = {"text": "text", "doc_id": None, "heading_path": None}
        result = build_reranker_text(chunk, use_heading_context=True)
        assert result == "text"

    def test_heading_path_truncated_at_max_length(self):
        long_path = "A" * 300
        chunk = {"text": "body", "doc_id": "Doc", "heading_path": long_path}
        result = build_reranker_text(chunk, use_heading_context=True)
        # heading in result must be at most _HEADING_PATH_MAX_LEN chars
        lines = result.split("\n")
        section_line = next(l for l in lines if l.startswith("Section:"))
        heading_value = section_line[len("Section: "):]
        assert len(heading_value) <= 200

    def test_no_text_duplication(self):
        """text must appear exactly once in the output."""
        chunk = self._chunk(text="unique_content_xyz")
        result = build_reranker_text(chunk, use_heading_context=True)
        assert result.count("unique_content_xyz") == 1

    def test_deterministic(self):
        """Same input always produces same output."""
        chunk = self._chunk()
        assert build_reranker_text(chunk) == build_reranker_text(chunk)

    def test_old_chunk_without_heading_path_key(self):
        """Chunks ingested before heading_path existed — no crash."""
        chunk = {"text": "legacy text", "doc_id": "OldDoc"}
        result = build_reranker_text(chunk, use_heading_context=True)
        assert "legacy text" in result
        assert "OldDoc" in result

    def test_whitespace_heading_path_treated_as_empty(self):
        chunk = {"text": "text", "doc_id": "Doc", "heading_path": "   "}
        result = build_reranker_text(chunk, use_heading_context=True)
        assert "Section:" not in result
        assert result == "Document: Doc\n\ntext"


# ---------------------------------------------------------------------------
# rerank() integration — heading_path_used stamp
# ---------------------------------------------------------------------------

class TestRerankHeadingContext:
    def setup_method(self):
        reset_reranker_cache()

    def teardown_method(self):
        reset_reranker_cache()

    def _mock_reranker(self, scores):
        mock = MagicMock()
        mock.compute_score.return_value = scores
        return mock

    def test_heading_path_used_true_when_heading_present(self):
        chunks = [
            {"text": "text1", "doc_id": "Doc", "heading_path": "## Section A"},
            {"text": "text2", "doc_id": "Doc", "heading_path": "## Section B"},
        ]
        mock = self._mock_reranker([0.9, 0.7])
        with (
            patch("rag_lab.retrieval.reranker.load_reranker", return_value=mock),
            patch("rag_lab.retrieval.reranker.RERANKER_USE_HEADING_CONTEXT", True),
        ):
            results = rerank("query", chunks, top_k=2, device="cpu")
        assert all(r["heading_path_used"] is True for r in results)

    def test_heading_path_used_false_when_heading_absent(self):
        chunks = [
            {"text": "text1", "doc_id": "Doc", "heading_path": ""},
            {"text": "text2", "doc_id": "Doc"},
        ]
        mock = self._mock_reranker([0.8, 0.6])
        with (
            patch("rag_lab.retrieval.reranker.load_reranker", return_value=mock),
            patch("rag_lab.retrieval.reranker.RERANKER_USE_HEADING_CONTEXT", True),
        ):
            results = rerank("query", chunks, top_k=2, device="cpu")
        assert all(r["heading_path_used"] is False for r in results)

    def test_heading_path_used_false_when_context_disabled(self):
        chunks = [{"text": "t", "doc_id": "D", "heading_path": "## S"}]
        mock = self._mock_reranker([0.5])
        with (
            patch("rag_lab.retrieval.reranker.load_reranker", return_value=mock),
            patch("rag_lab.retrieval.reranker.RERANKER_USE_HEADING_CONTEXT", False),
        ):
            results = rerank("query", chunks, top_k=1, device="cpu")
        assert results[0]["heading_path_used"] is False

    def test_reranker_receives_heading_context_in_text(self):
        """Verify the cross-encoder actually receives the enriched text."""
        chunks = [{"text": "body", "doc_id": "Doc", "heading_path": "## Intro"}]
        mock = self._mock_reranker([0.9])
        with (
            patch("rag_lab.retrieval.reranker.load_reranker", return_value=mock),
            patch("rag_lab.retrieval.reranker.RERANKER_USE_HEADING_CONTEXT", True),
        ):
            rerank("my query", chunks, top_k=1, device="cpu")
        call_args = mock.compute_score.call_args[0][0]
        pair = call_args[0]
        assert pair[0] == "my query"
        assert "Document: Doc" in pair[1]
        assert "## Intro" in pair[1]
        assert "body" in pair[1]

    def test_reranker_text_only_when_context_disabled(self):
        """With RERANKER_USE_HEADING_CONTEXT=False, no header is sent."""
        chunks = [{"text": "body", "doc_id": "Doc", "heading_path": "## Intro"}]
        mock = self._mock_reranker([0.9])
        with (
            patch("rag_lab.retrieval.reranker.load_reranker", return_value=mock),
            patch("rag_lab.retrieval.reranker.RERANKER_USE_HEADING_CONTEXT", False),
        ):
            rerank("my query", chunks, top_k=1, device="cpu")
        call_args = mock.compute_score.call_args[0][0]
        pair = call_args[0]
        assert pair[1] == "body"

    def test_candidate_generation_unchanged(self):
        """rerank() does not modify chunk count (candidate generation not touched)."""
        chunks = [
            {"text": f"text{i}", "doc_id": "D", "heading_path": f"## S{i}"}
            for i in range(10)
        ]
        mock = self._mock_reranker(list(range(10)))
        with (
            patch("rag_lab.retrieval.reranker.load_reranker", return_value=mock),
            patch("rag_lab.retrieval.reranker.RERANKER_USE_HEADING_CONTEXT", True),
        ):
            results = rerank("q", chunks, top_k=10, device="cpu")
        assert len(results) == 10

    def test_rerank_score_attached(self):
        """rerank_score field is still present after heading context change."""
        chunks = [{"text": "t", "doc_id": "D", "heading_path": "## S"}]
        mock = self._mock_reranker([0.88])
        with (
            patch("rag_lab.retrieval.reranker.load_reranker", return_value=mock),
            patch("rag_lab.retrieval.reranker.RERANKER_USE_HEADING_CONTEXT", True),
        ):
            results = rerank("q", chunks, top_k=1, device="cpu")
        assert "rerank_score" in results[0]
        assert abs(results[0]["rerank_score"] - 0.88) < 1e-6
