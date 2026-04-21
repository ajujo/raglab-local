"""Phase 1: Document ingestion and cleaning."""

from rag_lab.ingest.cleaner import clean_document
from rag_lab.ingest.manifest import create_manifest

__all__ = ["clean_document", "create_manifest"]