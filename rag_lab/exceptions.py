"""Custom exceptions for the RAG-Lab system."""


class RAGLabError(Exception):
    """Base exception for the RAG-Lab system."""
    pass


class DocumentIngestionError(RAGLabError):
    """Raised when document processing fails."""
    pass


class ChunkingError(RAGLabError):
    """Raised when chunking fails."""
    pass


class EmbeddingError(RAGLabError):
    """Raised when embedding generation fails (OOM, GPU, etc.)."""
    pass


class RetrievalError(RAGLabError):
    """Raised when retrieval/search fails."""
    pass


class LLMConnectionError(RAGLabError):
    """Raised when LLM server is unavailable."""
    pass