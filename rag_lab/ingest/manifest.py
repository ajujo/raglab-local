"""Generate manifest file for ingested documents.

Creates a JSONL file with metadata about each ingested document,
including hash, path, size, and ingestion timestamp.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rag_lab.config import DATA_DIR
from rag_lab.exceptions import DocumentIngestionError

logger = logging.getLogger("rag_lab")

# `rag_lab.config.DATA_DIR` is the single source of truth and is resolved at
# call time, so monkeypatching it redirects every manifest write (this is how
# the test suite keeps its manifest out of the real `data/` directory).
# The module-level `DATA_DIR` is kept for backwards compatibility: tests that
# monkeypatch *this* attribute still take precedence.
_DATA_DIR_AT_IMPORT = DATA_DIR


def _data_dir() -> Path:
    """Return the effective data directory (config wins unless patched here)."""
    if DATA_DIR is not _DATA_DIR_AT_IMPORT:
        return DATA_DIR
    from rag_lab import config as _config
    return _config.DATA_DIR


def compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        MD5 hex digest string.
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def create_manifest(
    source_path: Path,
    cleaned_path: Path = None,
    force: bool = False,
    doc_id: str = None,
) -> Path:
    """Create or update the ingested.jsonl manifest file.

    Args:
        source_path: Path to the original source document.
        cleaned_path: Path to the cleaned document.
        force: If True, always regenerate even if already ingested.
        doc_id: Stable document identifier. Defaults to ``source_path.stem``
            for backward compatibility; callers that resolve identity from
            frontmatter must pass it explicitly (RAG-05).

    Returns:
        Path to the manifest file.

    Raises:
        DocumentIngestionError: If source file doesn't exist.
    """
    if not source_path.exists():
        raise DocumentIngestionError(f"Source file not found: {source_path}")

    manifest_path = _data_dir() / "ingested.jsonl"
    _data_dir().mkdir(parents=True, exist_ok=True)

    # Load existing manifest
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        # Use doc_id as key (separate variable: do NOT shadow the
                        # doc_id argument below).
                        entry_doc_id = entry.get("doc_id", "")
                        if entry_doc_id:
                            manifest[entry_doc_id] = entry
                    except json.JSONDecodeError:
                        continue

    # Compute hash and metadata
    resolved_doc_id = doc_id or source_path.stem
    file_hash = compute_md5(source_path)

    # Check if already ingested
    if not force and resolved_doc_id in manifest:
        existing = manifest[resolved_doc_id]
        if existing.get("hash") == file_hash:
            logger.info(f"Document already ingested: {resolved_doc_id}")
            return manifest_path

    # Create new entry
    file_size = source_path.stat().st_size
    ingested_at = datetime.now(timezone.utc).isoformat()

    entry = {
        "doc_id": resolved_doc_id,
        "hash": file_hash,
        "path": str(source_path),
        "size": file_size,
        "ingested_at": ingested_at,
    }

    if cleaned_path:
        entry["cleaned_path"] = str(cleaned_path)

    manifest[resolved_doc_id] = entry

    # Write manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Manifest updated: {resolved_doc_id} (hash: {file_hash[:8]}...)")

    return manifest_path