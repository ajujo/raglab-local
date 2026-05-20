"""Backfill sparse BLOBs for chunks that lack them.

SAFE: only issues SQL UPDATE on existing rows — no re-chunking, no ID changes,
no duplicate risk. Idempotent: skips chunks that already have sparse data.

Usage:
    python -m rag_lab.maintenance.backfill_sparse [--device cuda] [--batch-size 8] [--dry-run]

When to use:
    Run this after migrating to schema v2 (migrate_to_v2.py) to populate the
    sparse_tokens/sparse_weights columns for chunks ingested before schema v2.
    After this script, 100% sparse coverage enables full three-way hybrid search.
"""

import argparse
import logging
import sys
import time

import numpy as np

from rag_lab.config import (
    DOCDSTORE_SQLITE_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_VERSION,
    SPARSE_FORMAT_VERSION,
)
from rag_lab.embedding.encoder import encode_chunks
from rag_lab.logging_config import setup_logging
from rag_lab.storage.docstore import DocStore

logger = logging.getLogger("rag_lab")


def backfill_sparse(
    device: str = None,
    batch_size: int = None,
    dry_run: bool = False,
) -> dict:
    """Backfill sparse BLOBs in-place for all chunks that lack them.

    Args:
        device: Embedding device ('cuda' or 'cpu'). Defaults to EMBEDDING_DEVICE.
        batch_size: Embedding batch size. Defaults to EMBEDDING_BATCH_SIZE.
        dry_run: If True, report what would be done but make no changes.

    Returns:
        Summary dict with counts.
    """
    device = device or EMBEDDING_DEVICE
    batch_size = batch_size or EMBEDDING_BATCH_SIZE

    ds = DocStore()
    ds.initialize()
    conn = ds._conn

    # Find chunks without sparse BLOBs
    rows = conn.execute(
        "SELECT chunk_id, doc_id, text FROM chunks WHERE sparse_tokens IS NULL ORDER BY doc_id"
    ).fetchall()

    total_missing = len(rows)
    print(f"\n{'─'*55}")
    print("Backfill sparse BLOBs")
    print(f"{'─'*55}")
    print(f"  Device:          {device}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Chunks to fill:  {total_missing}")

    if total_missing == 0:
        print("  ✓ All chunks already have sparse BLOBs.")
        print(f"{'─'*55}\n")
        ds.close()
        return {"total_missing": 0, "processed": 0, "errors": 0}

    if dry_run:
        print("  (DRY RUN — no changes will be made)")
        print(f"{'─'*55}\n")
        ds.close()
        return {"total_missing": total_missing, "processed": 0, "errors": 0, "dry_run": True}

    # Process in batches
    processed = 0
    errors = 0
    t_start = time.time()

    for batch_start in range(0, total_missing, batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        chunk_dicts = [
            {"chunk_id": r[0], "doc_id": r[1], "text": r[2]}
            for r in batch
        ]

        try:
            _, sparse_embeddings = encode_chunks(
                chunk_dicts,
                batch_size=batch_size,
                device=device,
            )
        except Exception as e:
            logger.error(f"Embedding batch {batch_start}–{batch_start+len(batch)} failed: {e}")
            errors += len(batch)
            continue

        # Write sparse BLOBs back to docstore
        for chunk_d in chunk_dicts:
            cid = chunk_d["chunk_id"]
            sparse = sparse_embeddings.get(cid, {})
            if sparse:
                tokens_arr = np.array(list(sparse.keys()), dtype=np.int32)
                weights_arr = np.array(list(sparse.values()), dtype=np.float32)
                conn.execute(
                    "UPDATE chunks SET sparse_tokens=?, sparse_weights=?, "
                    "embedding_model_name=?, embedding_model_version=?, "
                    "embedding_dim=?, sparse_format_version=? "
                    "WHERE chunk_id=?",
                    (
                        tokens_arr.tobytes(),
                        weights_arr.tobytes(),
                        EMBEDDING_MODEL,
                        EMBEDDING_MODEL_VERSION,
                        len(tokens_arr),  # actual sparse dim (variable)
                        SPARSE_FORMAT_VERSION,
                        cid,
                    ),
                )
                processed += 1
            else:
                errors += 1
                logger.warning(f"No sparse output for chunk {cid}")

        conn.commit()

        elapsed = time.time() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        pct = 100 * (batch_start + len(batch)) / total_missing
        print(
            f"  [{pct:5.1f}%] {batch_start + len(batch)}/{total_missing}  "
            f"{rate:.1f} chunks/s",
            end="\r",
        )

    elapsed = time.time() - t_start
    print()
    print(f"\n  ✓ Processed: {processed}  |  Errors: {errors}  |  Time: {elapsed:.1f}s")

    # Final coverage
    total, with_sparse = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN sparse_tokens IS NOT NULL THEN 1 ELSE 0 END) FROM chunks"
    ).fetchone()
    coverage = (with_sparse or 0) / total if total else 0.0
    print(f"  Sparse coverage after backfill: {with_sparse}/{total}  ({coverage:.1%})")
    print(f"{'─'*55}\n")

    ds.close()
    return {
        "total_missing": total_missing,
        "processed": processed,
        "errors": errors,
        "final_coverage": coverage,
    }


if __name__ == "__main__":
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Backfill sparse BLOBs for existing chunks")
    parser.add_argument("--device", default=None, help="Embedding device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size")
    parser.add_argument("--dry-run", action="store_true", help="Report without making changes")
    args = parser.parse_args()
    result = backfill_sparse(device=args.device, batch_size=args.batch_size, dry_run=args.dry_run)
    sys.exit(0 if result.get("errors", 0) == 0 else 1)
