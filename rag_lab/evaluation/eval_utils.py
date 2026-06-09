"""Evaluation utilities — helpers used only by the eval pipeline.

These helpers must NOT be imported in production code paths
(ingest, query, chat). They exist only for eval/benchmark tooling.
"""

from __future__ import annotations

import re

# Matches: [[N] Fuente: doc_id | Sección: heading | Líneas: X-Y]
# The outer brackets delimit the whole annotation; [N] is the citation number.
_INLINE_CITATION_RE = re.compile(r'\[\[\d+\][^\]]*\]')


def strip_inline_citations_for_eval(text: str) -> str:
    """Remove inline citation annotations from an answer for RAGAS evaluation.

    The pipeline appends source citations inline, e.g.:
        [[3] Fuente: SDMX_Glossary | Sección: ... | Líneas: 6283-6321]

    These annotations are useful for the user but contaminate RAGAS
    answer_relevancy: RAGAS generates synthetic questions from the answer
    text, and citation metadata skews those questions away from the
    question's actual topic.

    This function strips the annotations while preserving the substantive
    answer text. The original `answer` field is never modified — this
    produces a separate `answer_for_eval` field.

    Args:
        text: Raw answer text with inline citations.

    Returns:
        Answer text with citations stripped and whitespace normalised.
        If the input has no citations, returns a whitespace-normalised copy.
    """
    cleaned = _INLINE_CITATION_RE.sub("", text)
    # Fix space that forms before punctuation after a citation is removed,
    # e.g. "concept  [[1] ...]. Next" → "concept . Next" → "concept. Next"
    cleaned = re.sub(r" +([.,;:?!])", r"\1", cleaned)
    # Collapse any double spaces left behind
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()
