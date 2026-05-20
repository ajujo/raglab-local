"""Migration to schema v2: sparse BLOBs + FTS5 virtual table.

What this script does:
  1. Runs the idempotent schema migration (adds columns, creates FTS5 table, indexes).
  2. Populates chunks_fts from the existing `chunks` table (backfill FTS5).
  3. Migrates sparse vectors from sparse_index.json (if it exists) into BLOB columns.
  4. Reports how many chunks still lack sparse data (need re-ingest to populate).

Run:
    python -m rag_lab.maintenance.migrate_to_v2

Idempotent: safe to run multiple times. Already-migrated rows are skipped.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from rag_lab.config import DOCDSTORE_SQLITE_PATH, SPARSE_INDEX_PATH
from rag_lab.logging_config import setup_logging
from rag_lab.storage.docstore import DocStore

logger = logging.getLogger("rag_lab")


def migrate() -> dict:
    """Run the full v2 migration.

    Returns:
        Summary dict with counts.
    """
    print("\n─────────────────────────────────────────")
    print("Migration: docstore → schema v2")
    print("─────────────────────────────────────────")

    # Step 1: Open DocStore — initialize() applies the idempotent schema migration
    ds = DocStore()
    ds.initialize()
    conn = ds._conn

    total_chunks = ds.count()
    print(f"  Chunks in docstore: {total_chunks}")

    # Step 2: Populate chunks_fts from existing chunks (backfill)
    already_in_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    print(f"  Chunks already in FTS5: {already_in_fts}")

    if already_in_fts < total_chunks:
        print("  Backfilling FTS5...")
        # Find chunk_ids not yet in chunks_fts
        fts_ids = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks_fts").fetchall()}
        all_ids = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks").fetchall()}
        missing_from_fts = all_ids - fts_ids

        if missing_from_fts:
            rows = conn.execute(
                f"SELECT chunk_id, doc_id, text FROM chunks WHERE chunk_id IN "
                f"({','.join('?' * len(missing_from_fts))})",
                list(missing_from_fts),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO chunks_fts(chunk_id, doc_id, text) VALUES (?,?,?)",
                    row,
                )
            conn.commit()
            print(f"  ✓ Added {len(missing_from_fts)} chunks to FTS5")
    else:
        print("  ✓ FTS5 already up to date")

    # Step 3: Migrate sparse vectors from sparse_index.json
    sparse_migrated = 0
    if SPARSE_INDEX_PATH.exists():
        print(f"\n  Reading {SPARSE_INDEX_PATH}...")
        with open(SPARSE_INDEX_PATH, encoding="utf-8") as f:
            sparse_json = json.load(f)

        print(f"  Found {len(sparse_json)} entries in sparse_index.json")

        for chunk_id, entry in sparse_json.items():
            sparse_vec = entry.get("sparse", {})
            if not sparse_vec:
                continue

            # Check if this chunk is in docstore and lacks sparse BLOBs
            row = conn.execute(
                "SELECT sparse_tokens FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                continue  # chunk not in docstore — skip
            if row[0] is not None:
                continue  # already has sparse data

            tokens_arr = np.array([int(k) for k in sparse_vec.keys()], dtype=np.int32)
            weights_arr = np.array(list(sparse_vec.values()), dtype=np.float32)
            conn.execute(
                "UPDATE chunks SET sparse_tokens=?, sparse_weights=? WHERE chunk_id=?",
                (tokens_arr.tobytes(), weights_arr.tobytes(), chunk_id),
            )
            sparse_migrated += 1

        if sparse_migrated > 0:
            conn.commit()
            print(f"  ✓ Migrated {sparse_migrated} sparse vectors from JSON")
        else:
            print("  ✓ No new sparse vectors to migrate from JSON")
    else:
        print(f"  (sparse_index.json not found at {SPARSE_INDEX_PATH} — skipping)")

    # Step 4: Report remaining gaps
    no_sparse = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE sparse_tokens IS NULL"
    ).fetchone()[0]

    print(f"\n  Chunks without sparse BLOBs: {no_sparse}")
    if no_sparse > 0:
        pct = 100 * no_sparse / total_chunks if total_chunks else 0
        print(f"  ({pct:.1f}% of total — re-ingest to populate)")
        print("  Tip: python -m rag_lab.cli ingest --force")

    fts_final = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    print(f"\n  FTS5 entries: {fts_final}/{total_chunks}")

    ds.close()

    result = {
        "total_chunks": total_chunks,
        "fts_populated": fts_final,
        "sparse_migrated_from_json": sparse_migrated,
        "chunks_without_sparse": no_sparse,
    }
    print("\n─────────────────────────────────────────")
    print("Migration complete.")
    print("─────────────────────────────────────────\n")
    return result


if __name__ == "__main__":
    setup_logging("INFO")
    result = migrate()
    sys.exit(0 if result["chunks_without_sparse"] == 0 else 0)  # non-zero only on error
