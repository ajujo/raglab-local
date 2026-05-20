"""Tests for retrieval/fusion.py — weighted_rrf and rrf_three."""

import pytest
from rag_lab.retrieval.fusion import rrf_three, weighted_rrf


class TestWeightedRRF:
    def test_empty_all_returns_empty(self):
        assert weighted_rrf([], [], []) == []

    def test_dense_only(self):
        result = weighted_rrf(["a", "b", "c"], [], [])
        ids = [r["id"] for r in result]
        assert ids == ["a", "b", "c"]

    def test_bm25_only(self):
        bm25 = [{"id": "x", "bm25_score": 10.0}, {"id": "y", "bm25_score": 5.0}]
        result = weighted_rrf([], bm25, [])
        assert result[0]["id"] == "x"

    def test_overlap_scores_are_summed(self):
        result = weighted_rrf(
            ["a", "b"],
            [{"id": "a", "bm25_score": 8.0}, {"id": "c", "bm25_score": 3.0}],
            [],
        )
        assert result[0]["id"] == "a"
        assert result[0]["rrf_score"] > result[1]["rrf_score"]

    def test_provenance_flags(self):
        result = weighted_rrf(
            ["a"],
            [{"id": "b", "bm25_score": 5.0}],
            [{"id": "c", "sparse_score": 0.9}],
        )
        by_id = {r["id"]: r for r in result}
        assert by_id["a"]["in_dense_topk"] is True
        assert by_id["a"]["in_bm25_topk"] is False
        assert by_id["a"]["in_sparse_topk"] is False

        assert by_id["b"]["in_dense_topk"] is False
        assert by_id["b"]["in_bm25_topk"] is True

        assert by_id["c"]["in_sparse_topk"] is True

    def test_raw_scores_preserved(self):
        bm25 = [{"id": "a", "bm25_score": 12.5}]
        sparse = [{"id": "a", "sparse_score": 0.77}]
        result = weighted_rrf(["a"], bm25, sparse)
        assert result[0]["bm25_score"] == pytest.approx(12.5)
        assert result[0]["sparse_score"] == pytest.approx(0.77)

    def test_rrf_k_smoothing(self):
        r_k0 = weighted_rrf(["a"], [], [], k=0)
        r_k60 = weighted_rrf(["a"], [], [], k=60)
        assert r_k0[0]["rrf_score"] > r_k60[0]["rrf_score"]

    def test_result_sorted_descending(self):
        result = weighted_rrf(
            ["z", "y", "x"],
            [{"id": "x", "bm25_score": 1.0}],
            [],
        )
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_sparse_weight_zero_disables_sparse_contribution(self):
        # With sparse_w=0, a doc present only in sparse should not appear
        result_with = weighted_rrf(
            ["a"],
            [],
            [{"id": "b", "sparse_score": 0.9}],
            sparse_w=1.0,
        )
        result_without = weighted_rrf(
            ["a"],
            [],
            [{"id": "b", "sparse_score": 0.9}],
            sparse_w=0.0,
        )
        ids_with = [r["id"] for r in result_with]
        ids_without = [r["id"] for r in result_without]
        assert "b" in ids_with
        # sparse_w=0 → b contributes 0 to rrf_scores; it still appears but at score 0.0
        b_score = next((r["rrf_score"] for r in result_without if r["id"] == "b"), None)
        assert b_score == pytest.approx(0.0)

    def test_lower_sparse_weight_reduces_large_doc_dominance(self):
        # small_0 is at dense rank 5; large_0 is at sparse rank 0.
        # With k=20: dense[rank5] = 1/26 ≈ 0.038; sparse[rank0] with w=1.0 = 1/21 ≈ 0.048
        # → w=1.0: large_0 beats small_0; w=0.25: large_0=0.25/21≈0.012 < small_0=0.038
        dense = [f"filler_{i}" for i in range(5)] + ["small_0"]
        sparse = [{"id": "large_0", "sparse_score": 1.0}]

        result_high = weighted_rrf(dense, [], sparse, sparse_w=1.0, k=20)
        result_low = weighted_rrf(dense, [], sparse, sparse_w=0.25, k=20)

        rank_small_high = next(i for i, r in enumerate(result_high) if r["id"] == "small_0")
        rank_large_high = next(i for i, r in enumerate(result_high) if r["id"] == "large_0")
        rank_small_low = next(i for i, r in enumerate(result_low) if r["id"] == "small_0")
        rank_large_low = next(i for i, r in enumerate(result_low) if r["id"] == "large_0")

        assert rank_large_high < rank_small_high  # high sparse_w: large_0 beats small_0
        assert rank_small_low < rank_large_low    # low sparse_w: small_0 beats large_0

    def test_lower_rrf_k_is_more_discriminative(self):
        # With k=0, rank differences are magnified; score spread should be larger
        result_k0 = weighted_rrf(["a", "b", "c"], [], [], k=0)
        result_k60 = weighted_rrf(["a", "b", "c"], [], [], k=60)

        spread_k0 = result_k0[0]["rrf_score"] - result_k0[-1]["rrf_score"]
        spread_k60 = result_k60[0]["rrf_score"] - result_k60[-1]["rrf_score"]
        assert spread_k0 > spread_k60

    def test_dense_w_scales_contribution(self):
        result_w1 = weighted_rrf(["a"], [], [], dense_w=1.0, k=60)
        result_w2 = weighted_rrf(["a"], [], [], dense_w=2.0, k=60)
        assert result_w2[0]["rrf_score"] == pytest.approx(result_w1[0]["rrf_score"] * 2)


class TestRRFThreeBackwardCompat:
    """rrf_three must remain a valid backward-compat wrapper."""

    def test_empty_all_returns_empty(self):
        assert rrf_three([], [], []) == []

    def test_dense_only(self):
        result = rrf_three(["a", "b", "c"], [], [])
        ids = [r["id"] for r in result]
        assert ids == ["a", "b", "c"]

    def test_bm25_only(self):
        bm25 = [{"id": "x", "bm25_score": 10.0}, {"id": "y", "bm25_score": 5.0}]
        result = rrf_three([], bm25, [])
        assert result[0]["id"] == "x"

    def test_overlap_scores_are_summed(self):
        result = rrf_three(
            ["a", "b"],
            [{"id": "a", "bm25_score": 8.0}, {"id": "c", "bm25_score": 3.0}],
            [],
        )
        assert result[0]["id"] == "a"

    def test_provenance_flags(self):
        result = rrf_three(
            ["a"],
            [{"id": "b", "bm25_score": 5.0}],
            [{"id": "c", "sparse_score": 0.9}],
        )
        by_id = {r["id"]: r for r in result}
        assert by_id["a"]["in_dense_topk"] is True
        assert by_id["b"]["in_bm25_topk"] is True
        assert by_id["c"]["in_sparse_topk"] is True

    def test_raw_scores_preserved(self):
        bm25 = [{"id": "a", "bm25_score": 12.5}]
        sparse = [{"id": "a", "sparse_score": 0.77}]
        result = rrf_three(["a"], bm25, sparse)
        assert result[0]["bm25_score"] == pytest.approx(12.5)
        assert result[0]["sparse_score"] == pytest.approx(0.77)

    def test_rrf_k_smoothing(self):
        r_k0 = rrf_three(["a"], [], [], k=0)
        r_k60 = rrf_three(["a"], [], [], k=60)
        assert r_k0[0]["rrf_score"] > r_k60[0]["rrf_score"]

    def test_result_sorted_descending(self):
        result = rrf_three(
            ["z", "y", "x"],
            [{"id": "x", "bm25_score": 1.0}],
            [],
        )
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_equal_weights_match_weighted_rrf(self):
        dense = ["a", "b", "c"]
        bm25 = [{"id": "b", "bm25_score": 5.0}, {"id": "d", "bm25_score": 2.0}]
        sparse = [{"id": "c", "sparse_score": 0.8}]

        r3 = rrf_three(dense, bm25, sparse, k=60)
        rw = weighted_rrf(dense, bm25, sparse, k=60)

        assert len(r3) == len(rw)
        for a, b in zip(r3, rw):
            assert a["id"] == b["id"]
            assert a["rrf_score"] == pytest.approx(b["rrf_score"])
