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


class LLMContextOverflowError(RAGLabError):
    """Raised when a prompt cannot fit within the model context window.

    Carries the budget details so callers can surface a recoverable error
    instead of silently truncating information.
    """

    def __init__(self, message: str, *, required_tokens: int = 0,
                 budget_tokens: int = 0, window: int = 0) -> None:
        super().__init__(message)
        self.required_tokens = required_tokens
        self.budget_tokens = budget_tokens
        self.window = window


class VersioningError(RAGLabError):
    """Raised when the document-versioning safety machinery cannot proceed.

    Raised instead of degrading silently: a failure to snapshot the previous
    version, to delete/restore vectors, or to publish the active version must
    abort the operation rather than risk data loss (RAG-01/02/04).
    """
    pass