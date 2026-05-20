"""Basic tests for BenchmarkRunner: loading, variant dispatch, output shape.

These tests use mocks or a tiny in-memory corpus — no GPU, no real stores.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_lab.benchmark.pipeline_variants import run_variant, VARIANT_NAMES
from rag_lab.benchmark.runner import BenchmarkRunner


# ---------------------------------------------------------------------------
# load_queries
# ---------------------------------------------------------------------------

class TestLoadQueries:
    def test_load_yaml(self, tmp_path):
        f = tmp_path / "q.yaml"
        f.write_text(
            "queries:\n"
            "  - id: q1\n"
            "    text: 'What is X?'\n"
            "    doc_relevance:\n"
            "      docA: 3\n"
        )
        queries = BenchmarkRunner.load_queries(f)
        assert len(queries) == 1
        assert queries[0]["id"] == "q1"
        assert queries[0]["doc_relevance"]["docA"] == 3

    def test_load_json(self, tmp_path):
        f = tmp_path / "q.json"
        f.write_text(json.dumps([{"id": "q1", "text": "hello"}]))
        queries = BenchmarkRunner.load_queries(f)
        assert queries[0]["id"] == "q1"

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "q.txt"
        f.write_text("something")
        with pytest.raises(ValueError, match="Unsupported"):
            BenchmarkRunner.load_queries(f)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BenchmarkRunner.load_queries(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# run_variant dispatcher
# ---------------------------------------------------------------------------

class TestRunVariantDispatcher:
    def _mock_stores(self):
        vs = MagicMock()
        vs.query.return_value = {
            "ids": ["c1", "c2"],
            "distances": [0.1, 0.2],
            "documents": ["text1", "text2"],
            "metadatas": [{}, {}],
        }
        ds = MagicMock()
        ds.get_by_ids.return_value = [
            {"chunk_id": "c1", "doc_id": "docA", "text": "hello"},
            {"chunk_id": "c2", "doc_id": "docA", "text": "world"},
        ]
        ds._conn = MagicMock()
        fts = MagicMock()
        fts.query.return_value = [{"id": "c1", "bm25_score": 5.0}]
        return vs, ds, fts

    def test_dense_returns_chunks(self):
        vs, ds, fts = self._mock_stores()
        q_dense = np.zeros(1024, dtype="float32")
        q_sparse = {}
        chunks, stats = run_variant(
            "dense", "test query", q_dense, q_sparse,
            vs, ds, fts, top_k=5, rrf_k=60, rerank_device="cpu"
        )
        assert isinstance(chunks, list)
        assert "latency_ms" in stats
        assert stats["n_dense"] > 0
        assert stats["n_bm25"] == 0

    def test_bm25_returns_chunks(self):
        vs, ds, fts = self._mock_stores()
        q_dense = np.zeros(1024, dtype="float32")
        chunks, stats = run_variant(
            "bm25", "test query", q_dense, {},
            vs, ds, fts, top_k=5, rrf_k=60, rerank_device="cpu"
        )
        assert isinstance(chunks, list)
        assert stats["n_bm25"] >= 0
        assert stats["n_dense"] == 0

    def test_invalid_variant_raises(self):
        vs, ds, fts = self._mock_stores()
        with pytest.raises(ValueError, match="Unknown variant"):
            run_variant(
                "nonexistent", "q", np.zeros(4), {},
                vs, ds, fts, top_k=5, rrf_k=60, rerank_device="cpu"
            )

    def test_all_variant_names_dispatchable(self):
        """Every variant in VARIANT_NAMES must be known to the dispatcher."""
        vs, ds, fts = self._mock_stores()
        q_dense = np.zeros(1024, dtype="float32")

        for name in VARIANT_NAMES:
            # Patch heavy calls so no real models are loaded
            with patch("rag_lab.benchmark.pipeline_variants.hybrid_search",
                       return_value=([], {"candidate_pool_size": 0, "n_dense": 0,
                                          "n_bm25": 0, "n_sparse": 0, "sparse_used": False})), \
                 patch("rag_lab.benchmark.pipeline_variants.rerank",
                       return_value=[], create=True):
                chunks, stats = run_variant(
                    name, "q", q_dense, {},
                    vs, ds, fts, top_k=5, rrf_k=60, rerank_device="cpu"
                )
            assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------

class TestToMarkdown:
    def _mock_result(self, variants):
        agg = {
            "recall@5": 0.8, "recall@10": 0.9, "recall@30": 1.0,
            "mrr": 0.75, "ndcg@10": 0.82,
            "p50": 45.0, "p95": 120.0, "p99": 200.0,
            "candidate_pool_size": 60.0,
            "dense_coverage": 0.9, "bm25_coverage": 0.7, "sparse_coverage": 0.5,
        }
        return {
            "config": {
                "top_k": 30, "rrf_k": 60,
                "n_queries": 12, "variants": variants,
            },
            "results": {v: {"aggregate": agg, "per_query": []} for v in variants},
        }

    def test_produces_markdown(self):
        result = self._mock_result(["dense", "hybrid"])
        md = BenchmarkRunner.to_markdown(result)
        assert "| dense" in md
        assert "| hybrid" in md
        assert "R@5" in md
        assert "nDCG@10" in md

    def test_all_variants_present(self):
        result = self._mock_result(VARIANT_NAMES)
        md = BenchmarkRunner.to_markdown(result)
        for v in VARIANT_NAMES:
            assert v in md


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_json(self, tmp_path):
        data = {"config": {}, "results": {}}
        out = tmp_path / "results.json"
        BenchmarkRunner.save(data, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        data = {"x": 1}
        out = tmp_path / "nested" / "deep" / "out.json"
        BenchmarkRunner.save(data, out)
        assert out.exists()
