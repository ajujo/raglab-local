"""Reconcile: cross-store consistency check (DocStore vs ChromaDB).

Checks that every chunk_id in DocStore exists in ChromaDB and vice-versa.
Also reports sparse BLOB coverage and model-version / dimension mismatches.

Modes:
    python -m rag_lab.maintenance.reconcile            # report + exit code
    python -m rag_lab.maintenance.reconcile --check    # explicit CI mode (same behaviour)
    python -m rag_lab.maintenance.reconcile --repair   # remove orphaned IDs from ChromaDB
    python -m rag_lab.maintenance.reconcile --fix      # alias for --repair (backward compat)
    python -m rag_lab.maintenance.reconcile --report-json out.json

Exit codes:
    0 = all consistent
    1 = issues detected (orphans, mismatches, duplicates, etc.)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Union

from rag_lab.logging_config import setup_logging
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")


def reconcile(
    fix: bool = False,
    repair: bool = False,
    quiet: bool = False,
) -> dict:
    """Check consistency between DocStore, ChromaDB, FTS5, and sparse BLOBs.

    Args:
        fix:    Backward-compat alias for repair.
        repair: If True, remove orphaned entries from ChromaDB.
        quiet:  If True, suppress all stdout output (useful for programmatic callers).

    Returns:
        Extended dict with counts, orphan lists, and mismatch lists.
    """
    from rag_lab.config import (
        EMBEDDING_DIM,
        EMBEDDING_MODEL_VERSION,
        SPARSE_FORMAT_VERSION,
    )

    do_repair = fix or repair

    ds = DocStore()
    ds.initialize()
    conn = ds._conn

    vector_store = VectorStore()
    vector_store.initialize()

    # --- DocStore IDs (source of truth) ---
    docstore_ids = set(
        row[0] for row in conn.execute("SELECT chunk_id FROM chunks").fetchall()
    )

    # --- ChromaDB IDs ---
    try:
        chroma_result = vector_store._collection.get(include=[])
        chroma_ids = set(chroma_result["ids"])
    except Exception as e:
        logger.error(f"Failed to read ChromaDB: {e}")
        chroma_ids = set()

    # --- FTS5 coverage ---
    try:
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    except Exception:
        fts_count = 0

    # --- Sparse BLOB coverage ---
    try:
        sparse_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE sparse_tokens IS NOT NULL"
        ).fetchone()[0]
    except Exception:
        sparse_count = 0

    # --- Duplicate chunk_ids (PK constraint should prevent these; belt-and-suspenders) ---
    try:
        dupe_rows = conn.execute(
            "SELECT chunk_id FROM chunks GROUP BY chunk_id HAVING COUNT(*) > 1"
        ).fetchall()
        duplicate_chunk_ids = [r[0] for r in dupe_rows]
    except Exception:
        duplicate_chunk_ids = []

    # --- Model version mismatches ---
    try:
        rows = conn.execute(
            "SELECT chunk_id, embedding_model_version FROM chunks "
            "WHERE embedding_model_version IS NOT NULL AND embedding_model_version != '' "
            "AND embedding_model_version != ?",
            (EMBEDDING_MODEL_VERSION,),
        ).fetchall()
        model_version_mismatches = [
            {"chunk_id": r[0], "stored_version": r[1], "config_version": EMBEDDING_MODEL_VERSION}
            for r in rows
        ]
    except Exception:
        model_version_mismatches = []

    # --- Embedding dim mismatches ---
    try:
        rows = conn.execute(
            "SELECT chunk_id, embedding_dim FROM chunks "
            "WHERE embedding_dim IS NOT NULL AND embedding_dim != 0 "
            "AND embedding_dim != ?",
            (EMBEDDING_DIM,),
        ).fetchall()
        embedding_dim_mismatches = [
            {"chunk_id": r[0], "stored_dim": r[1], "config_dim": EMBEDDING_DIM}
            for r in rows
        ]
    except Exception:
        embedding_dim_mismatches = []

    # --- Sparse format version mismatches ---
    try:
        rows = conn.execute(
            "SELECT chunk_id, sparse_format_version FROM chunks "
            "WHERE sparse_format_version IS NOT NULL AND sparse_format_version != 0 "
            "AND sparse_format_version != ?",
            (SPARSE_FORMAT_VERSION,),
        ).fetchall()
        sparse_format_version_mismatches = [
            {"chunk_id": r[0], "stored_version": r[1], "config_version": SPARSE_FORMAT_VERSION}
            for r in rows
        ]
    except Exception:
        sparse_format_version_mismatches = []

    in_chroma_not_docstore = chroma_ids - docstore_ids
    in_docstore_not_chroma = docstore_ids - chroma_ids

    result = {
        "docstore_count": len(docstore_ids),
        "chroma_count": len(chroma_ids),
        "fts_count": fts_count,
        "sparse_blob_count": sparse_count,
        "chroma_orphans": sorted(in_chroma_not_docstore),
        "missing_from_chroma": sorted(in_docstore_not_chroma),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "model_version_mismatches": model_version_mismatches,
        "embedding_dim_mismatches": embedding_dim_mismatches,
        "sparse_format_version_mismatches": sparse_format_version_mismatches,
        "repaired": False,
    }

    if not quiet:
        _print_report(result, len(docstore_ids))

    if do_repair and in_chroma_not_docstore:
        ids_to_delete = list(in_chroma_not_docstore)
        vector_store._collection.delete(ids=ids_to_delete)
        result["repaired"] = True
        if not quiet:
            print(f"  ✓ Removed {len(ids_to_delete)} orphaned IDs from ChromaDB")
            sep = "─" * 50
            print(f"{sep}\n")

    ds.close()
    return result


def _print_report(result: dict, docstore_count: int) -> None:
    """Render the reconcile report to stdout."""
    sep = "─" * 50
    print(f"\n{sep}")
    print("Reconcile Report")
    print(sep)
    print(f"  DocStore (SQLite):  {result['docstore_count']:6d} chunks")
    print(f"  ChromaDB:           {result['chroma_count']:6d} chunks")
    print(f"  FTS5 index:         {result['fts_count']:6d} chunks")
    sparse = result["sparse_blob_count"]
    pct = 100 * sparse // max(docstore_count, 1)
    print(f"  Sparse BLOBs:       {sparse:6d} chunks  ({pct}% coverage)")
    print()

    has_core_issues = bool(result["chroma_orphans"] or result["missing_from_chroma"])

    if not has_core_issues:
        print("  ✓ DocStore and ChromaDB are consistent.")
    else:
        if result["chroma_orphans"]:
            print(f"  ⚠ ChromaDB has {len(result['chroma_orphans'])} orphaned IDs (not in DocStore)")
        if result["missing_from_chroma"]:
            print(f"  ⚠ DocStore has {len(result['missing_from_chroma'])} IDs missing from ChromaDB")

    if result["duplicate_chunk_ids"]:
        print(f"  ⚠ {len(result['duplicate_chunk_ids'])} duplicate chunk_ids in DocStore")

    if result["model_version_mismatches"]:
        print(f"  ⚠ {len(result['model_version_mismatches'])} chunks with stale embedding_model_version")

    if result["embedding_dim_mismatches"]:
        print(f"  ⚠ {len(result['embedding_dim_mismatches'])} chunks with mismatched embedding_dim")

    if result["sparse_format_version_mismatches"]:
        print(f"  ⚠ {len(result['sparse_format_version_mismatches'])} chunks with stale sparse_format_version")

    if sparse < docstore_count:
        missing = docstore_count - sparse
        print(f"  ℹ {missing} chunks without sparse BLOBs — run: python -m rag_lab.cli ingest --force")

    if result["fts_count"] < docstore_count:
        missing = docstore_count - result["fts_count"]
        print(f"  ℹ {missing} chunks not in FTS5 — run: python -m rag_lab.maintenance.migrate_to_v2")

    print(f"{sep}\n")


def save_report(result: dict, path: Union[str, Path]) -> None:
    """Serialize the reconcile result dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def _has_issues(result: dict) -> bool:
    return bool(
        result.get("chroma_orphans")
        or result.get("missing_from_chroma")
        or result.get("duplicate_chunk_ids")
        or result.get("model_version_mismatches")
        or result.get("embedding_dim_mismatches")
        or result.get("sparse_format_version_mismatches")
    )


if __name__ == "__main__":
    setup_logging("INFO")
    parser = argparse.ArgumentParser(
        description="Reconcile RAG-Lab stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0 = consistent\n"
            "  1 = issues detected\n"
        ),
    )
    parser.add_argument("--fix", action="store_true",
                        help="Alias for --repair (backward compat)")
    parser.add_argument("--repair", action="store_true",
                        help="Remove orphaned entries from ChromaDB")
    parser.add_argument("--check", action="store_true",
                        help="CI mode: exit 0 if consistent, 1 if issues (default behaviour)")
    parser.add_argument("--report-json", metavar="PATH", default=None,
                        help="Save JSON report to this path")
    args = parser.parse_args()

    result = reconcile(fix=args.fix, repair=args.repair)

    if args.report_json:
        save_report(result, args.report_json)
        print(f"Report saved to {args.report_json}")

    sys.exit(1 if _has_issues(result) else 0)
