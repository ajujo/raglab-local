"""Tests for rag_lab/retrieval/diversity.py."""

import pytest
from rag_lab.retrieval.diversity import apply_document_cap, apply_mmr


def _chunk(chunk_id: str, doc_id: str, rrf_score: float) -> dict:
    return {"chunk_id": chunk_id, "doc_id": doc_id, "rrf_score": rrf_score}


# ---------------------------------------------------------------------------
# apply_document_cap
# ---------------------------------------------------------------------------

class TestDocumentCap:
    def test_empty_input(self):
        assert apply_document_cap([], cap=2) == []

    def test_cap_raises_on_zero(self):
        with pytest.raises(ValueError):
            apply_document_cap([_chunk("a", "doc1", 1.0)], cap=0)

    def test_no_truncation_when_below_cap(self):
        chunks = [_chunk("a", "doc1", 1.0), _chunk("b", "doc2", 0.9)]
        assert apply_document_cap(chunks, cap=2) == chunks

    def test_limits_to_cap_per_doc(self):
        chunks = [
            _chunk("a1", "doc1", 1.0),
            _chunk("a2", "doc1", 0.9),
            _chunk("a3", "doc1", 0.8),
            _chunk("b1", "doc2", 0.7),
        ]
        result = apply_document_cap(chunks, cap=2)
        assert len(result) == 3
        doc1_ids = [c["chunk_id"] for c in result if c["doc_id"] == "doc1"]
        assert doc1_ids == ["a1", "a2"]  # first two preserved

    def test_cap_1_one_chunk_per_doc(self):
        chunks = [
            _chunk("a1", "doc1", 1.0),
            _chunk("a2", "doc1", 0.9),
            _chunk("b1", "doc2", 0.8),
            _chunk("b2", "doc2", 0.7),
        ]
        result = apply_document_cap(chunks, cap=1)
        assert len(result) == 2
        assert result[0]["chunk_id"] == "a1"
        assert result[1]["chunk_id"] == "b1"

    def test_preserves_rrf_order(self):
        chunks = [
            _chunk("z", "doc1", 1.0),
            _chunk("y", "doc2", 0.9),
            _chunk("x", "doc3", 0.8),
        ]
        result = apply_document_cap(chunks, cap=3)
        assert [c["chunk_id"] for c in result] == ["z", "y", "x"]

    def test_large_cap_passes_all(self):
        chunks = [_chunk(f"c{i}", f"doc{i}", 1.0 - i * 0.1) for i in range(5)]
        assert apply_document_cap(chunks, cap=10) == chunks


# ---------------------------------------------------------------------------
# apply_mmr
# ---------------------------------------------------------------------------

class TestMMR:
    def test_empty_input(self):
        assert apply_mmr([]) == []

    def test_lambda_raises_out_of_range(self):
        with pytest.raises(ValueError):
            apply_mmr([_chunk("a", "doc1", 1.0)], lambda_=1.5)

    def test_lambda_1_preserves_rrf_order(self):
        chunks = [
            _chunk("a1", "doc1", 1.0),
            _chunk("b1", "doc2", 0.8),
            _chunk("a2", "doc1", 0.6),
        ]
        result = apply_mmr(chunks, lambda_=1.0)
        assert [c["chunk_id"] for c in result] == ["a1", "b1", "a2"]

    def test_lambda_less_than_1_promotes_diverse_docs(self):
        # doc1 appears first and second (high score), doc2 has lower score
        # With lambda < 1.0, a2 (doc1, score=0.9) is penalised after a1 selected;
        # b1 (doc2, score=0.7) should jump ahead of a2
        chunks = [
            _chunk("a1", "doc1", 1.0),
            _chunk("a2", "doc1", 0.9),
            _chunk("b1", "doc2", 0.7),
        ]
        result = apply_mmr(chunks, lambda_=0.5)
        ids = [c["chunk_id"] for c in result]
        # After a1 is selected, b1 (new doc) beats a2 (penalised doc1)
        assert ids[0] == "a1"
        assert ids[1] == "b1"
        assert ids[2] == "a2"

    def test_k_limits_output_length(self):
        chunks = [_chunk(f"c{i}", f"doc{i}", 1.0 - i * 0.1) for i in range(10)]
        result = apply_mmr(chunks, k=3)
        assert len(result) == 3

    def test_mmr_score_field_added(self):
        chunks = [_chunk("a", "doc1", 1.0), _chunk("b", "doc2", 0.8)]
        result = apply_mmr(chunks, lambda_=0.7)
        assert all("mmr_score" in c for c in result)

    def test_original_chunks_not_mutated(self):
        chunks = [_chunk("a", "doc1", 1.0)]
        original = dict(chunks[0])
        apply_mmr(chunks, lambda_=0.7)
        assert chunks[0] == original  # original unchanged

    def test_all_same_doc_still_returns_k(self):
        chunks = [_chunk(f"c{i}", "doc1", 1.0 - i * 0.1) for i in range(5)]
        result = apply_mmr(chunks, lambda_=0.7, k=3)
        assert len(result) == 3

    def test_zero_scores_handled(self):
        chunks = [_chunk("a", "doc1", 0.0), _chunk("b", "doc2", 0.0)]
        result = apply_mmr(chunks)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# diversity_stats metric function
# ---------------------------------------------------------------------------

class TestDiversityStats:
    def test_basic_unique_docs(self):
        from rag_lab.benchmark.metrics import diversity_stats

        chunks = [
            _chunk("a", "doc1", 1.0),
            _chunk("b", "doc1", 0.9),
            _chunk("c", "doc2", 0.8),
            _chunk("d", "doc3", 0.7),
            _chunk("e", "doc3", 0.6),
        ]
        stats = diversity_stats(chunks, k_values=(5,))
        assert stats["unique_docs@5"] == 3.0
        assert stats["max_chunks_same_doc@5"] == 2.0

    def test_top5_only_first_five(self):
        from rag_lab.benchmark.metrics import diversity_stats

        chunks = [_chunk(f"c{i}", f"doc{i}", 1.0 - i * 0.1) for i in range(10)]
        stats = diversity_stats(chunks, k_values=(5,))
        assert stats["unique_docs@5"] == 5.0
        assert stats["max_chunks_same_doc@5"] == 1.0

    def test_empty_chunks(self):
        from rag_lab.benchmark.metrics import diversity_stats

        stats = diversity_stats([], k_values=(5, 10))
        assert stats["unique_docs@5"] == 0.0
        assert stats["max_chunks_same_doc@5"] == 0.0

    def test_multiple_k_values(self):
        from rag_lab.benchmark.metrics import diversity_stats

        chunks = [_chunk(f"c{i}", "doc1", 1.0 - i * 0.05) for i in range(10)]
        stats = diversity_stats(chunks, k_values=(5, 10))
        assert set(stats.keys()) == {
            "unique_docs@5", "max_chunks_same_doc@5",
            "unique_docs@10", "max_chunks_same_doc@10",
        }
        assert stats["unique_docs@5"] == 1.0
        assert stats["max_chunks_same_doc@5"] == 5.0
