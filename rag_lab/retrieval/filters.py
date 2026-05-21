"""Filter specification and resolution for hybrid search."""

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("rag_lab")


@dataclass
class FilterSpec:
    doc_ids: Optional[List[str]] = None
    tags_include: Optional[List[str]] = None
    tags_exclude: Optional[List[str]] = None
    source_id: Optional[str] = None
    dataset_id: Optional[str] = None
    status: Optional[str] = "active"

    def is_empty(self) -> bool:
        """Return True if no filter criteria are set."""
        return (
            not self.doc_ids
            and not self.tags_include
            and not self.tags_exclude
            and self.source_id is None
            and self.dataset_id is None
            and self.status is None
        )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def resolve_filter(
    conn: sqlite3.Connection, spec: FilterSpec
) -> Optional[List[str]]:
    if spec.is_empty():
        return None

    if not _table_exists(conn, "documents"):
        # v3 tables not yet migrated; honour explicit doc_ids if given
        return sorted(spec.doc_ids) if spec.doc_ids else None

    params: list = []
    joins: list = []
    where_clauses: list = []
    having_clauses: list = []

    if spec.status is not None:
        where_clauses.append("d.status = ?")
        params.append(spec.status)

    if spec.source_id is not None:
        where_clauses.append("d.source_id = ?")
        params.append(spec.source_id)

    if spec.dataset_id is not None:
        where_clauses.append("d.dataset_id = ?")
        params.append(spec.dataset_id)

    if spec.tags_include:
        joins.append(
            "JOIN document_tags dt_inc ON d.doc_id = dt_inc.doc_id"
        )
        joins.append("JOIN tags t_inc ON dt_inc.tag_id = t_inc.tag_id")
        placeholders = ", ".join("?" * len(spec.tags_include))
        where_clauses.append(f"t_inc.name IN ({placeholders})")
        params.extend(spec.tags_include)
        having_clauses.append(f"COUNT(DISTINCT t_inc.name) = {len(spec.tags_include)}")

    if spec.tags_exclude:
        placeholders = ", ".join("?" * len(spec.tags_exclude))
        where_clauses.append(
            f"d.doc_id NOT IN ("
            f"  SELECT dt_ex.doc_id FROM document_tags dt_ex"
            f"  JOIN tags t_ex ON dt_ex.tag_id = t_ex.tag_id"
            f"  WHERE t_ex.name IN ({placeholders})"
            f")"
        )
        params.extend(spec.tags_exclude)

    joins_sql = "\n    ".join(joins)
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    having_sql = ("HAVING " + " AND ".join(having_clauses)) if having_clauses else ""

    query = f"""
    SELECT d.doc_id
    FROM documents d
    {joins_sql}
    {where_sql}
    GROUP BY d.doc_id
    {having_sql}
    """

    rows = conn.execute(query, params).fetchall()
    resolved = {r[0] for r in rows}

    if spec.doc_ids is not None:
        # Intersection: explicit doc_ids scoped to those that pass the other filters,
        # or if there are no other filters beyond doc_ids, restrict to known documents.
        explicit = set(spec.doc_ids)
        if where_clauses or having_clauses:
            resolved = resolved & explicit
        else:
            # Only doc_ids filter active — validate against documents table.
            resolved = resolved & explicit

    return sorted(resolved)


def filter_stats(conn: sqlite3.Connection, spec: FilterSpec) -> dict:
    """Return diagnostic stats about a filter resolution."""
    if not _table_exists(conn, "documents"):
        total = 0
    else:
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    resolved = resolve_filter(conn, spec)

    return {
        "total_documents": total,
        "matched_documents": len(resolved) if resolved is not None else total,
        "resolved_doc_ids": resolved,
    }
