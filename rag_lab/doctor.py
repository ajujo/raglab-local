"""RAG-Lab system health checks.

Runs 7 sequential checks and reports OK / WARN / FAIL for each.

Usage:
    python -m rag_lab.doctor                        # full check
    python -m rag_lab.doctor --checks config,docstore,chromadb
    python -m rag_lab.doctor --query "What is SDMX?"

Exit codes:
    0 = all OK
    1 = at least one WARN, no FAIL
    2 = at least one FAIL
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from rag_lab.logging_config import setup_logging
from rag_lab.maintenance.reconcile import _has_issues, reconcile
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.fts_store import FTSStore
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")

ALL_CHECKS = [
    "config",
    "docstore",
    "chromadb",
    "fts5",
    "sparse_coverage",
    "reconcile",
    "test_query",
]


@dataclass
class CheckResult:
    name: str
    status: str                     # "OK" | "WARN" | "FAIL"
    reason: Optional[str] = None
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_config() -> CheckResult:
    """Validate that all required config keys exist and have sane values."""
    try:
        from rag_lab.config import (
            EMBEDDING_DIM,
            EMBEDDING_MODEL_VERSION,
            RETRIEVAL_TOP_K,
            RRF_K,
            SPARSE_COVERAGE_THRESHOLD,
            SPARSE_FORMAT_VERSION,
        )
        issues = []
        if EMBEDDING_DIM <= 0:
            issues.append(f"EMBEDDING_DIM={EMBEDDING_DIM} must be > 0")
        if not EMBEDDING_MODEL_VERSION:
            issues.append("EMBEDDING_MODEL_VERSION is empty")
        if RETRIEVAL_TOP_K <= 0:
            issues.append(f"RETRIEVAL_TOP_K={RETRIEVAL_TOP_K} must be > 0")
        if RRF_K <= 0:
            issues.append(f"RRF_K={RRF_K} must be > 0")
        if not (0.0 <= SPARSE_COVERAGE_THRESHOLD <= 1.0):
            issues.append(f"SPARSE_COVERAGE_THRESHOLD={SPARSE_COVERAGE_THRESHOLD} must be in [0,1]")

        if issues:
            return CheckResult("config", "FAIL", "; ".join(issues))
        return CheckResult(
            "config", "OK",
            f"dim={EMBEDDING_DIM}, model_ver={EMBEDDING_MODEL_VERSION}, top_k={RETRIEVAL_TOP_K}",
        )
    except Exception as e:
        return CheckResult("config", "FAIL", f"Import error: {e}")


def check_docstore() -> CheckResult:
    """Verify the SQLite DocStore opens and contains at least one chunk."""
    try:
        ds = DocStore()
        ds.initialize()
        count = ds.count()
        ds.close()
        if count == 0:
            return CheckResult("docstore", "WARN", "DocStore is empty (0 chunks)")
        return CheckResult("docstore", "OK", f"{count} chunks")
    except Exception as e:
        return CheckResult("docstore", "FAIL", str(e))


def check_chromadb() -> CheckResult:
    """Verify ChromaDB opens and the collection is reachable."""
    try:
        vs = VectorStore()
        vs.initialize()
        count = vs._collection.count()
        if count == 0:
            return CheckResult("chromadb", "WARN", "ChromaDB collection is empty")
        return CheckResult("chromadb", "OK", f"{count} vectors")
    except Exception as e:
        return CheckResult("chromadb", "FAIL", str(e))


def check_fts5() -> CheckResult:
    """Check that chunks_fts virtual table exists and is populated."""
    try:
        ds = DocStore()
        ds.initialize()
        conn = ds._conn
        total = ds.count()
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        ds.close()
        if fts_count == 0:
            return CheckResult("fts5", "FAIL", "FTS5 table is empty — run migrate_to_v2")
        if fts_count < total:
            gap = total - fts_count
            return CheckResult(
                "fts5", "WARN",
                f"{fts_count}/{total} chunks indexed ({gap} missing) — run migrate_to_v2",
            )
        return CheckResult("fts5", "OK", f"{fts_count}/{total} chunks indexed")
    except Exception as e:
        return CheckResult("fts5", "FAIL", str(e))


def check_sparse_coverage() -> CheckResult:
    """Check that sparse BLOBs meet the configured coverage threshold."""
    try:
        from rag_lab.config import SPARSE_COVERAGE_THRESHOLD
        ds = DocStore()
        ds.initialize()
        conn = ds._conn
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN sparse_tokens IS NOT NULL THEN 1 ELSE 0 END) FROM chunks"
        ).fetchone()
        ds.close()
        total, sparse = row[0], row[1] or 0
        if total == 0:
            return CheckResult("sparse_coverage", "WARN", "DocStore is empty")
        coverage = sparse / total
        pct = int(100 * coverage)
        if coverage < SPARSE_COVERAGE_THRESHOLD:
            return CheckResult(
                "sparse_coverage", "WARN",
                f"{sparse}/{total} ({pct}%) — below threshold {int(100*SPARSE_COVERAGE_THRESHOLD)}%"
                " — run: python -m rag_lab.maintenance.backfill_sparse",
            )
        return CheckResult("sparse_coverage", "OK", f"{sparse}/{total} ({pct}%)")
    except Exception as e:
        return CheckResult("sparse_coverage", "FAIL", str(e))


def check_reconcile() -> CheckResult:
    """Run reconcile in quiet mode and report any cross-store issues."""
    try:
        result = reconcile(quiet=True)
        if not _has_issues(result):
            return CheckResult(
                "reconcile", "OK",
                f"DocStore={result['docstore_count']}, ChromaDB={result['chroma_count']}",
            )
        issues = []
        if result.get("chroma_orphans"):
            issues.append(f"{len(result['chroma_orphans'])} ChromaDB orphans")
        if result.get("missing_from_chroma"):
            issues.append(f"{len(result['missing_from_chroma'])} missing from ChromaDB")
        if result.get("duplicate_chunk_ids"):
            issues.append(f"{len(result['duplicate_chunk_ids'])} duplicate chunk_ids")
        if result.get("model_version_mismatches"):
            issues.append(f"{len(result['model_version_mismatches'])} model_version mismatches")
        if result.get("embedding_dim_mismatches"):
            issues.append(f"{len(result['embedding_dim_mismatches'])} embedding_dim mismatches")
        if result.get("sparse_format_version_mismatches"):
            issues.append(f"{len(result['sparse_format_version_mismatches'])} sparse_format_version mismatches")
        severity = "FAIL" if (
            result.get("chroma_orphans") or result.get("missing_from_chroma") or
            result.get("duplicate_chunk_ids")
        ) else "WARN"
        return CheckResult("reconcile", severity, "; ".join(issues))
    except Exception as e:
        return CheckResult("reconcile", "FAIL", str(e))


def check_test_query(query: str = "What is SDMX?") -> CheckResult:
    """Run a simple retrieval query and verify at least one result is returned."""
    try:
        from rag_lab.config import EMBEDDING_DEVICE
        from rag_lab.embedding.encoder import encode_chunks
        from rag_lab.retrieval.hybrid_search import hybrid_search

        ds = DocStore()
        ds.initialize()
        vs = VectorStore()
        vs.initialize()
        fts = FTSStore()
        fts.initialize()

        dense_emb, sparse_dict = encode_chunks(
            [{"text": query, "chunk_id": "__doctor_query__"}],
            batch_size=1,
            device=EMBEDDING_DEVICE,
        )
        query_dense = dense_emb[0]
        query_sparse = next(iter(sparse_dict.values()), {})

        results = hybrid_search(
            query, vs, ds, fts,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=3,
        )

        fts.close()
        ds.close()

        if not results:
            return CheckResult("test_query", "FAIL", f"No results for query: {query!r}")

        top = results[0]
        return CheckResult(
            "test_query", "OK",
            f"{len(results)} results — top: {top.get('doc_id','?')} "
            f"rrf={top.get('rrf_score', 0):.4f}",
        )
    except Exception as e:
        return CheckResult("test_query", "FAIL", str(e))


# ---------------------------------------------------------------------------
# Main doctor() function
# ---------------------------------------------------------------------------

def doctor(
    checks: Optional[List[str]] = None,
    query: str = "What is SDMX?",
    quiet: bool = False,
) -> dict:
    """Run system health checks.

    Args:
        checks: Subset of check names to run (default = all).
        query:  Query used for check_test_query.
        quiet:  If True, suppress all stdout output.

    Returns:
        dict with keys "results" (list of CheckResult), "overall" ("OK"|"WARN"|"FAIL").
    """
    import sys as _sys
    _module = _sys.modules[__name__]

    names = checks if checks else ALL_CHECKS
    invalid = [n for n in names if not hasattr(_module, f"check_{n}")]
    if invalid:
        raise ValueError(f"Unknown checks: {invalid}. Valid: {ALL_CHECKS}")

    sep = "─" * 55
    if not quiet:
        print(f"\n{sep}")
        print("RAG-Lab Doctor")
        print(sep)

    results: List[CheckResult] = []
    for name in names:
        fn = getattr(_module, f"check_{name}")
        if name == "test_query":
            result = fn(query=query)
        else:
            result = fn()
        results.append(result)

        if not quiet:
            icon = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}.get(result.status, "?")
            line = f"  {icon} {result.name:<20} {result.status}"
            if result.reason:
                line += f"  — {result.reason}"
            print(line)

    overall = "OK"
    for r in results:
        if r.status == "FAIL":
            overall = "FAIL"
            break
        if r.status == "WARN":
            overall = "WARN"

    if not quiet:
        print(f"\n{sep}")
        icon = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}[overall]
        print(f"  {icon} Overall: {overall}")
        print(f"{sep}\n")

    return {"results": results, "overall": overall}


if __name__ == "__main__":
    setup_logging("WARNING")
    parser = argparse.ArgumentParser(
        description="RAG-Lab system health check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0 = all OK\n"
            "  1 = at least one WARN, no FAIL\n"
            "  2 = at least one FAIL\n"
        ),
    )
    parser.add_argument(
        "--checks",
        metavar="NAME[,NAME...]",
        default=None,
        help=f"Comma-separated subset of checks to run. Available: {', '.join(ALL_CHECKS)}",
    )
    parser.add_argument(
        "--query",
        default="What is SDMX?",
        help="Query used for the test_query check (default: 'What is SDMX?')",
    )
    args = parser.parse_args()

    check_list = [c.strip() for c in args.checks.split(",")] if args.checks else None

    result = doctor(checks=check_list, query=args.query)

    overall = result["overall"]
    if overall == "FAIL":
        sys.exit(2)
    elif overall == "WARN":
        sys.exit(1)
    else:
        sys.exit(0)
