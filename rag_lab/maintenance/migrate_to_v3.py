"""Migration to schema v3: documents / tags / document_tags / sources tables.

What this script does:
  1. Opens DocStore — initialize() runs _migrate_v3() which creates the v3 metadata
     tables idempotently (INSERT OR IGNORE / CREATE TABLE IF NOT EXISTS).
  2. Populates `documents` from distinct doc_ids already present in `chunks`,
     then back-fills embedding_model_version, embedding_dim, and
     sparse_format_version from the first matching chunk row.
  3. (Optional) Migrates document paths, content hashes, and tags from
     storage/doc_manager.db if that file exists.
  4. Prints a summary report.

Run:
    python -m rag_lab.maintenance.migrate_to_v3

Idempotent: safe to run multiple times. Already-present rows are skipped or
updated without duplicates.
"""

import logging
import sqlite3
import sys
from pathlib import Path

from rag_lab.config import DOCDSTORE_SQLITE_PATH, STORAGE_DIR
from rag_lab.logging_config import setup_logging
from rag_lab.storage.docstore import DocStore
from rag_lab.storage.metadata_store import MetadataStore

logger = logging.getLogger("rag_lab")

# Legacy doc_manager database path (may not exist on every installation)
DOC_MANAGER_DB_PATH = STORAGE_DIR / "doc_manager.db"


def migrate() -> dict:
    """Run the full v3 migration.

    Returns:
        Summary dict with counts.
    """
    print("\n─────────────────────────────────────────")
    print("Migration: docstore → schema v3")
    print("─────────────────────────────────────────")

    # ------------------------------------------------------------------
    # Step 1: Open DocStore — initialize() applies all idempotent migrations
    # including _migrate_v3() which creates the metadata tables.
    # ------------------------------------------------------------------
    ds = DocStore()
    ds.initialize()
    conn = ds._conn

    # MetadataStore shares the same connection so every write lands in the
    # same SQLite transaction without a separate commit cycle.
    meta = MetadataStore(conn=conn)

    total_chunks = ds.count()
    print(f"  Chunks in docstore  : {total_chunks}")

    # ------------------------------------------------------------------
    # Step 2: Populate `documents` from existing chunks
    # ------------------------------------------------------------------
    print("\n  Populating documents from chunks...")

    # Count how many doc_ids are already in `documents` before we insert
    existing_before = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    # INSERT OR IGNORE guarantees idempotency; ingested_at is set to now only
    # for genuinely new rows (the DEFAULT clause in CREATE TABLE covers it).
    conn.execute(
        """
        INSERT OR IGNORE INTO documents(doc_id, ingested_at)
        SELECT DISTINCT doc_id, datetime('now')
        FROM chunks
        WHERE doc_id IS NOT NULL AND doc_id != ''
        """
    )

    existing_after = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    newly_inserted = existing_after - existing_before
    already_existed = existing_before

    print(f"  Documents already in table : {already_existed}")
    print(f"  Documents inserted now     : {newly_inserted}")

    # Back-fill embedding metadata from the first chunk of each document.
    # We use a correlated sub-query so the UPDATE is still idempotent:
    # re-running it simply writes the same values again.
    conn.execute(
        """
        UPDATE documents AS d SET
            embedding_model_version = (
                SELECT c.embedding_model_version
                FROM chunks c
                WHERE c.doc_id = d.doc_id
                LIMIT 1
            ),
            embedding_dim = (
                SELECT c.embedding_dim
                FROM chunks c
                WHERE c.doc_id = d.doc_id
                LIMIT 1
            ),
            sparse_format_version = (
                SELECT c.sparse_format_version
                FROM chunks c
                WHERE c.doc_id = d.doc_id
                LIMIT 1
            )
        WHERE d.doc_id IN (SELECT DISTINCT doc_id FROM chunks)
        """
    )
    conn.commit()
    print("  Embedding metadata back-filled.")

    # ------------------------------------------------------------------
    # Step 3: Migrate from legacy doc_manager.db (if it exists)
    # ------------------------------------------------------------------
    tags_migrated = 0
    docs_from_manager = 0

    if DOC_MANAGER_DB_PATH.exists():
        print(f"\n  Reading legacy doc_manager.db at {DOC_MANAGER_DB_PATH} ...")
        old_conn = sqlite3.connect(str(DOC_MANAGER_DB_PATH))
        old_conn.row_factory = sqlite3.Row

        # Migrate document metadata (path + content_hash)
        try:
            doc_rows = old_conn.execute(
                "SELECT doc_id, path, hash FROM documents"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning(f"Could not read documents from doc_manager.db: {exc}")
            doc_rows = []

        for row in doc_rows:
            doc_id = row["doc_id"]
            path = row["path"]
            content_hash = row["hash"]
            meta.upsert_document(doc_id, path=path, content_hash=content_hash)
            docs_from_manager += 1

        if docs_from_manager:
            conn.commit()

        # Migrate tags
        try:
            tag_rows = old_conn.execute(
                "SELECT doc_id, tag_name FROM tags"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning(f"Could not read tags from doc_manager.db: {exc}")
            tag_rows = []

        for row in tag_rows:
            doc_id = row["doc_id"]
            tag_name = row["tag_name"]
            # assign_tag uses INSERT OR IGNORE internally — fully idempotent
            meta.assign_tag(doc_id, tag_name)
            tags_migrated += 1

        if tags_migrated:
            conn.commit()

        old_conn.close()

        print(f"  Documents migrated from doc_manager.db : {docs_from_manager}")
        print(f"  Tags migrated from doc_manager.db      : {tags_migrated}")
    else:
        print(
            f"\n  (doc_manager.db not found at {DOC_MANAGER_DB_PATH} — skipping legacy migration)"
        )

    # ------------------------------------------------------------------
    # Step 4: Final state report
    # ------------------------------------------------------------------
    final_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    final_tags = conn.execute("SELECT COUNT(*) FROM document_tags").fetchone()[0]

    print(f"\n  Final state: {final_docs} document(s), {final_tags} tag assignment(s)")

    ds.close()

    result = {
        "total_chunks": total_chunks,
        "documents_already_existed": already_existed,
        "documents_inserted": newly_inserted,
        "docs_migrated_from_manager": docs_from_manager,
        "tags_migrated": tags_migrated,
        "final_documents": final_docs,
        "final_tag_assignments": final_tags,
    }

    print("\n─────────────────────────────────────────")
    print("Migration complete.")
    print("─────────────────────────────────────────\n")
    return result


if __name__ == "__main__":
    setup_logging("INFO")
    migrate()
    sys.exit(0)
