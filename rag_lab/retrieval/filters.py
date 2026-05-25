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
    status: Optional[str] = "active"
    # Document classification filters — resolved as normalised tags internally
    domain: Optional[str] = None
    source_type: Optional[str] = None
    language: Optional[str] = None
    version: Optional[str] = None

    def is_empty(self) -> bool:
        """Return True if no filter criteria are set."""
        return (
            not self.doc_ids
            and not self.tags_include
            and not self.tags_exclude
            and self.source_id is None
            and self.status is None
            and self.domain is None
            and self.source_type is None
            and self.language is None
            and self.version is None
        )

    def _effective_tags_include(self) -> Optional[List[str]]:
        """Return tags_include merged with classification filters as derived tags."""
        extra: list[str] = []
        if self.domain:
            extra.append(f"domain:{self.domain.strip().lower()}")
        if self.source_type:
            extra.append(f"source_type:{self.source_type.strip().lower()}")
        if self.language:
            extra.append(f"lang:{self.language.strip().lower()}")
        if self.version:
            extra.append(f"version:{str(self.version).strip()}")
        base = list(self.tags_include or [])
        combined = base + [t for t in extra if t not in base]
        return combined if combined else None


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

    effective_tags_include = spec._effective_tags_include()
    if effective_tags_include:
        joins.append(
            "JOIN document_tags dt_inc ON d.doc_id = dt_inc.doc_id"
        )
        joins.append("JOIN tags t_inc ON dt_inc.tag_id = t_inc.tag_id")
        placeholders = ", ".join("?" * len(effective_tags_include))
        where_clauses.append(f"t_inc.name IN ({placeholders})")
        params.extend(effective_tags_include)
        having_clauses.append(f"COUNT(DISTINCT t_inc.name) = {len(effective_tags_include)}")

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
