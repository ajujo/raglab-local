"""Phase 2: Semantic chunking of documents."""

from rag_lab.chunking.parser import parse_headings
from rag_lab.chunking.splitter import chunk_document

__all__ = ["parse_headings", "chunk_document"]