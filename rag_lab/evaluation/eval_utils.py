"""Evaluation utilities — helpers used only by the eval pipeline.

These helpers must NOT be imported in production code paths
(ingest, query, chat). They exist only for eval/benchmark tooling.
"""

from __future__ import annotations

import re

# ── Abstention detection ──────────────────────────────────────────────────────
# Matches typical LLM phrases when the answer is not in the corpus.
# Covers Spanish (primary language of the pipeline) and English.
_ABSTENTION_RE = re.compile(
    r"no\s+(?:encuentro|encuentran|se\s+menciona|se\s+especifica|se\s+detalla|"
    r"se\s+lista|se\s+incluye|está\s+disponible|están\s+disponibles|"
    r"tengo\s+información|hay\s+información|dispongo)"
    # "no está/están presente(s)/disponible(s)/incluido(s)"
    r"|no\s+(?:está|están)\s+(?:presente|disponible|incluido|especificado)"
    # English: "not found / not available / not mentioned / not covered..."
    r"|(?:not\s+(?:found|available|present|mentioned|covered|described|listed|"
    r"included|provided|specified|detailed|addressed)\b)"
    # "cannot / can't / unable to find"
    r"|(?:(?:cannot|can't|unable\s+to)\s+find)"
    # "the context/documents do not contain/include/mention/cover"
    r"|(?:(?:the\s+)?(?:provided\s+)?(?:context|documents?|sources?|corpus)\s+"
    r"(?:do(?:es)?\s+not|don't|doesn't)\s+(?:contain|include|mention|cover|provide))"
    # "no information is available/found/provided"
    r"|(?:no\s+information\s+(?:is\s+)?(?:available|found|provided))"
    # "this information is not available/present in the..."
    r"|(?:this\s+information\s+is\s+not\s+(?:available|present|found))",
    re.IGNORECASE,
)


def is_abstention(answer: str, trust_score: float | None = None) -> bool:
    """Return True if *answer* is a plausible abstention (LLM says it can't answer).

    Uses two signals in combination:
    - Pattern matching against known abstention phrases (Spanish + English).
    - A very low trust_score (< 0.25) as a corroborating signal.

    An empty or very short answer (<= 30 chars) is always treated as abstention.

    Args:
        answer: The LLM answer text to analyse.
        trust_score: Optional trust score from the verification layer.
            Scores below 0.25 are treated as strong abstention signal.
    """
    if not answer or len(answer.strip()) <= 30:
        return True
    if trust_score is not None and trust_score < 0.25:
        return True
    return bool(_ABSTENTION_RE.search(answer))


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
