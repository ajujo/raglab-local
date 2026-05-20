"""Re-export weighted_rrf from rag_lab.retrieval.fusion for backward compatibility.

The canonical implementation lives in rag_lab/retrieval/fusion.py.
This module is kept so that external scripts importing from this path continue to work.
"""

from rag_lab.retrieval.fusion import weighted_rrf  # noqa: F401
