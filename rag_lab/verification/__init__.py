"""Verification Layer for RAG-Lab.

Provides post-generation verification including citation checking,
self-consistency evaluation, and trust scoring.
"""

from rag_lab.verification.verifier import CitationResult, verify_citations_layer
from rag_lab.verification.consistency import check_consistency
from rag_lab.verification.scoring import calculate_score, ConfidenceLevel
from rag_lab.verification.pipeline import verify_and_score

__all__ = [
    "CitationResult",
    "verify_citations_layer",
    "check_consistency",
    "calculate_score",
    "ConfidenceLevel",
    "verify_and_score",
]
