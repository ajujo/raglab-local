"""Reconcile: cross-store consistency check (DocStore vs ChromaDB).

Checks that every chunk_id in DocStore exists in ChromaDB and vice-versa.
Also reports sparse BLOB coverage (chunks missing sparse data need re-ingest).

Run:
    python -m rag_lab.maintenance.reconcile          # report only
    python -m rag_lab.maintenance.reconcile --fix    # remove orphaned IDs from ChromaDB
"""

import argparse
import logging
import sys

from rag_lab.logging_config import setup_logging
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")


def reconcile(fix: bool = False) -> dict:
    """Check consistency between DocStore and ChromaDB.

    Args:
        fix: If True, remove orphaned entries from ChromaDB.

    Returns:
        Dict with counts and lists of discrepancies.
    """
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

    in_chroma_not_docstore = chroma_ids - docstore_ids
    in_docstore_not_chroma = docstore_ids - chroma_ids

    result = {
        "docstore_count": len(docstore_ids),
        "chroma_count": len(chroma_ids),
        "fts_count": fts_count,
        "sparse_blob_count": sparse_count,
        "chroma_orphans": sorted(in_chroma_not_docstore),
        "missing_from_chroma": sorted(in_docstore_not_chroma),
    }

    sep = "─" * 50
    print(f"\n{sep}")
    print("Reconcile Report")
    print(sep)
    print(f"  DocStore (SQLite):  {len(docstore_ids):6d} chunks")
    print(f"  ChromaDB:           {len(chroma_ids):6d} chunks")
    print(f"  FTS5 index:         {fts_count:6d} chunks")
    print(f"  Sparse BLOBs:       {sparse_count:6d} chunks  "
          f"({100*sparse_count//max(len(docstore_ids),1)}% coverage)")
    print()

    has_issues = bool(in_chroma_not_docstore or in_docstore_not_chroma)

    if not has_issues:
        print("  ✓ DocStore and ChromaDB are consistent.")
    else:
        if in_chroma_not_docstore:
            print(f"  ⚠ ChromaDB has {len(in_chroma_not_docstore)} orphaned IDs (not in DocStore)")
        if in_docstore_not_chroma:
            print(f"  ⚠ DocStore has {len(in_docstore_not_chroma)} IDs missing from ChromaDB")

    if sparse_count < len(docstore_ids):
        missing = len(docstore_ids) - sparse_count
        print(f"  ℹ {missing} chunks without sparse BLOBs — run: python -m rag_lab.cli ingest --force")

    if fts_count < len(docstore_ids):
        missing = len(docstore_ids) - fts_count
        print(f"  ℹ {missing} chunks not in FTS5 — run: python -m rag_lab.maintenance.migrate_to_v2")

    if fix and in_chroma_not_docstore:
        print()
        ids_to_delete = list(in_chroma_not_docstore)
        vector_store._collection.delete(ids=ids_to_delete)
        print(f"  ✓ Removed {len(ids_to_delete)} orphaned IDs from ChromaDB")

    print(f"{sep}\n")
    ds.close()
    return result


if __name__ == "__main__":
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Reconcile RAG-Lab stores")
    parser.add_argument("--fix", action="store_true",
                        help="Remove orphaned entries from ChromaDB")
    args = parser.parse_args()
    result = reconcile(fix=args.fix)
    has_issues = bool(result["chroma_orphans"] or result["missing_from_chroma"])
    sys.exit(1 if has_issues else 0)
