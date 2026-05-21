"""Full system diagnostic: coverage, consistency, and retrieval sanity check.

Shows FTS5 coverage, sparse BLOB coverage, store counts, and optionally
runs a test query with all five scores to verify end-to-end retrieval.

Usage:
    python -m rag_lab.maintenance.diagnose
    python -m rag_lab.maintenance.diagnose --query "What is SDMX?"
    python -m rag_lab.maintenance.diagnose --query "What is SDMX?" --explain
"""

import argparse
import logging
import sys

from rag_lab.logging_config import setup_logging
from rag_lab.retrieval.filters import FilterSpec, filter_stats
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.vector_store import VectorStore

logger = logging.getLogger("rag_lab")


def diagnose(
    query: str = None,
    explain: bool = False,
    doc_id: str = None,
    tags_include: list = None,
    tags_exclude: list = None,
) -> dict:
    """Run full system diagnostic.

    Args:
        query: Optional test query to exercise the retrieval pipeline.
        explain: If True and query is provided, show per-signal rank breakdown
                 for each result (requires --query).

    Returns:
        Dict with all diagnostic results.
    """
    sep = "─" * 55
    print(f"\n{sep}")
    print("RAG-Lab System Diagnostic")
    print(sep)

    ds = DocStore()
    ds.initialize()
    conn = ds._conn

    # Build filter spec (may be empty)
    _filter_spec = FilterSpec(
        doc_ids=[doc_id] if doc_id else None,
        tags_include=tags_include or None,
        tags_exclude=tags_exclude or None,
    )

    # --- DocStore ---
    total_chunks = ds.count()

    # FTS5 coverage
    try:
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    except Exception:
        fts_count = 0

    # Sparse BLOB coverage
    try:
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN sparse_tokens IS NOT NULL THEN 1 ELSE 0 END) FROM chunks"
        ).fetchone()
        total_db, sparse_count = row[0], row[1] or 0
    except Exception:
        total_db, sparse_count = total_chunks, 0

    # Per-doc breakdown
    doc_rows = conn.execute(
        "SELECT doc_id, COUNT(*) as total, "
        "SUM(CASE WHEN sparse_tokens IS NOT NULL THEN 1 ELSE 0 END) as sparse "
        "FROM chunks GROUP BY doc_id ORDER BY doc_id"
    ).fetchall()

    # Duplicate check: same doc_id + line_start + line_end is a real span collision
    dupe_count = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT doc_id, line_start, line_end, COUNT(*) as n "
        "  FROM chunks GROUP BY doc_id, line_start, line_end HAVING n > 1"
        ")"
    ).fetchone()[0]

    # Model version consistency
    model_rows = conn.execute(
        "SELECT embedding_model_name, embedding_model_version, COUNT(*) "
        "FROM chunks WHERE embedding_model_name != '' "
        "GROUP BY embedding_model_name, embedding_model_version"
    ).fetchall()

    # --- ChromaDB ---
    try:
        vs = VectorStore()
        vs.initialize()
        chroma_count = vs._collection.count()
    except Exception as e:
        chroma_count = f"ERROR: {e}"

    # --- Print ---
    print(f"\n  DocStore (SQLite):")
    print(f"    Total chunks   : {total_chunks}")
    print(f"    Unique IDs     : {total_db}")
    print(f"    Duplicates     : {dupe_count}  {'✓' if dupe_count == 0 else '⚠ PROBLEM'}")

    print(f"\n  FTS5 index:")
    fts_ok = fts_count == total_chunks
    print(f"    Count          : {fts_count}/{total_chunks}  {'✓' if fts_ok else '⚠ run migrate_to_v2'}")

    print(f"\n  Sparse BLOBs:")
    sparse_pct = 100 * sparse_count // total_chunks if total_chunks else 0
    sparse_ok = sparse_count == total_chunks
    print(f"    Count          : {sparse_count}/{total_chunks}  ({sparse_pct}%)  "
          f"{'✓' if sparse_ok else '⚠ run backfill_sparse'}")
    if not sparse_ok:
        print(f"    Missing        : {total_chunks - sparse_count} chunks")

    print(f"\n  ChromaDB:")
    if isinstance(chroma_count, int):
        chroma_ok = abs(chroma_count - total_chunks) <= 2  # allow tiny mismatch
        print(f"    Count          : {chroma_count}  {'✓' if chroma_ok else '⚠ run reconcile --fix'}")
    else:
        print(f"    Count          : {chroma_count}")

    print(f"\n  Per-document breakdown:")
    for row in doc_rows:
        doc_id, total, sparse = row
        pct = 100 * (sparse or 0) // total if total else 0
        flag = "✓" if sparse == total else f"⚠ {total - (sparse or 0)} missing"
        print(f"    {doc_id:<40} {sparse or 0:3d}/{total:3d} sparse  {flag}")

    if model_rows:
        print(f"\n  Embedding model versions in docstore:")
        for row in model_rows:
            print(f"    {row[0]}  v{row[1]}  ({row[2]} chunks)")

    # --- Optional query test ---
    result = {
        "total_chunks": total_chunks,
        "fts_count": fts_count,
        "sparse_count": sparse_count,
        "chroma_count": chroma_count,
        "duplicates": dupe_count,
        "fts_ok": fts_ok,
        "sparse_ok": sparse_ok,
    }

    if query:
        print(f"\n  Test query: {query!r}")
        if not _filter_spec.is_empty():
            stats = filter_stats(conn, _filter_spec)
            print(f"  Filters active:")
            if _filter_spec.tags_include:
                print(f"    tags_include : {_filter_spec.tags_include}")
            if _filter_spec.tags_exclude:
                print(f"    tags_exclude : {_filter_spec.tags_exclude}")
            if _filter_spec.doc_ids:
                print(f"    doc_ids      : {_filter_spec.doc_ids}")
            print(f"    → {stats['matched_documents']}/{stats['total_documents']} documents match filter")
        print("  " + "·" * 50)
        _run_test_query(query, ds, explain=explain, filter_spec=_filter_spec)

    ds.close()

    issues = []
    if dupe_count > 0:
        issues.append(f"  ⚠  {dupe_count} duplicate chunks detected")
    if not fts_ok:
        issues.append(f"  ⚠  FTS5 incomplete ({fts_count}/{total_chunks})")
    if not sparse_ok:
        issues.append(f"  ℹ  Sparse BLOBs at {sparse_pct}% — run backfill_sparse for full coverage")
    if isinstance(chroma_count, int) and not chroma_ok:
        issues.append(f"  ⚠  ChromaDB mismatch ({chroma_count} vs {total_chunks})")

    print(f"\n{sep}")
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ All checks passed.")
    print(f"{sep}\n")

    result["issues"] = issues
    return result


def _run_test_query(
    query: str,
    ds: DocStore,
    explain: bool = False,
    filter_spec=None,
) -> None:
    """Run a retrieval test and print five-score details."""
    try:
        from rag_lab.config import EMBEDDING_DEVICE
        from rag_lab.embedding.encoder import encode_chunks
        from rag_lab.storage.fts_store import FTSStore
        from rag_lab.storage.vector_store import VectorStore
        from rag_lab.retrieval.hybrid_search import hybrid_search

        vs = VectorStore()
        vs.initialize()
        fts = FTSStore()
        fts.initialize()

        dense_emb, sparse_dict = encode_chunks(
            [{"text": query, "chunk_id": "__query__"}],
            batch_size=1,
            device=EMBEDDING_DEVICE,
        )
        query_dense = dense_emb[0]
        query_sparse = next(iter(sparse_dict.values()), {})

        results = hybrid_search(
            query, vs, ds, fts,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=5,
            filter_spec=filter_spec,
        )

        if not results:
            print("  (no results)")
            return

        print(f"  Top {len(results)} results:")
        for i, chunk in enumerate(results):
            print(f"\n  [{i+1}] {chunk.get('doc_id','?')} | "
                  f"lines {chunk.get('line_start','?')}-{chunk.get('line_end','?')}")
            print(f"       rrf={chunk.get('rrf_score',0):.4f}  "
                  f"dense={chunk.get('dense_score',0):.4f}  "
                  f"bm25={chunk.get('bm25_score',0):.2f}  "
                  f"sparse={chunk.get('sparse_score',0):.4f}")
            print(f"       in_dense={chunk.get('in_dense_topk',False)}  "
                  f"in_bm25={chunk.get('in_bm25_topk',False)}  "
                  f"in_sparse={chunk.get('in_sparse_topk',False)}")
            if explain:
                _print_explain(chunk)
            print(f"       {chunk.get('text','')[:80].strip()}...")

        fts.close()

    except Exception as e:
        print(f"  Query test failed: {e}")


def _print_explain(chunk: dict) -> None:
    """Print the full rank / provenance breakdown for one result chunk."""
    def _fmt_rank(r) -> str:
        return f"rank={r}" if r is not None else "rank=–"

    dense_r = _fmt_rank(chunk.get("dense_rank"))
    bm25_r  = _fmt_rank(chunk.get("bm25_rank"))
    sparse_r = _fmt_rank(chunk.get("sparse_rank"))
    rrf_r   = _fmt_rank(chunk.get("rrf_rank"))

    print(f"       ┌─ signal ranks: dense[{dense_r}]  bm25[{bm25_r}]  sparse[{sparse_r}]")
    print(f"       │  rrf_rank={chunk.get('rrf_rank','–')}  chunk_id={chunk.get('chunk_id','?')[:16]}…")

    mmr_score = chunk.get("mmr_score")
    if mmr_score is not None:
        reordered = chunk.get("was_mmr_reordered", False)
        reorder_flag = " ← MMR reordered" if reordered else ""
        print(f"       └─ mmr_score={mmr_score:.4f}{reorder_flag}")
    else:
        print(f"       └─ MMR: not active")


if __name__ == "__main__":
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="RAG-Lab system diagnostic")
    parser.add_argument("--query", default=None, help="Optional test query")
    parser.add_argument("--explain", action="store_true",
                        help="Show per-signal rank breakdown for each result (requires --query)")
    parser.add_argument("--doc-id", default=None, metavar="DOC_ID",
                        help="Restrict query to a single document")
    parser.add_argument("--tag", action="append", dest="tags_include", metavar="TAG",
                        help="Include only documents with this tag (repeatable)")
    parser.add_argument("--exclude-tag", action="append", dest="tags_exclude", metavar="TAG",
                        help="Exclude documents with this tag (repeatable)")
    args = parser.parse_args()
    if args.explain and not args.query:
        parser.error("--explain requires --query")
    result = diagnose(
        query=args.query,
        explain=args.explain,
        doc_id=args.doc_id,
        tags_include=args.tags_include,
        tags_exclude=args.tags_exclude,
    )
    sys.exit(1 if result.get("duplicates", 0) > 0 else 0)
