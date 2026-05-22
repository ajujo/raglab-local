"""Real token counting with lazy-loaded BGE-M3 tokenizer and char-based fallback.

Public API
----------
count_tokens(text) -> int
    Count tokens using the configured tokenizer. Equivalent to the old
    `len(text) // 4` heuristic when TOKEN_COUNTING_MODE="approx" or when
    the tokenizer cannot be loaded.

reset_tokenizer_cache()
    Force reload on next call. Intended for tests only.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("rag_lab")

# Module-level cache — same pattern as encoder.py's _model/_reranker globals.
_tokenizer: Optional[Any] = None
_load_attempted: bool = False
_fallback_warned: bool = False


def _load_tokenizer() -> Optional[Any]:
    """Lazy-load the tokenizer (tokenizer files only, not the full model).

    Returns None if TOKEN_COUNTING_MODE is "approx" or if loading fails,
    triggering the char-based fallback.
    """
    global _tokenizer, _load_attempted

    if _load_attempted:
        return _tokenizer

    _load_attempted = True

    # Import config lazily to avoid circular imports at module level.
    try:
        from rag_lab.config import TOKEN_COUNTING_MODE, TOKENIZER_MODEL_NAME
    except ImportError:
        TOKEN_COUNTING_MODE = "real"
        TOKENIZER_MODEL_NAME = "BAAI/bge-m3"

    if TOKEN_COUNTING_MODE == "approx":
        logger.debug("token counting: approx mode (configured), skipping tokenizer load")
        return None

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            TOKENIZER_MODEL_NAME,
            use_fast=True,
        )
        _tokenizer = tok
        logger.debug("token counting: loaded tokenizer '%s'", TOKENIZER_MODEL_NAME)
        return _tokenizer

    except Exception as exc:  # ImportError, OSError, network error, etc.
        logger.warning(
            "token counting: tokenizer unavailable (%s); "
            "falling back to char-based approximation (len(text) // 4)",
            exc,
        )
        return None


def count_tokens(text: str) -> int:
    """Return the token count for *text*.

    Uses the BGE-M3 (XLM-RoBERTa) tokenizer when available. Falls back to
    ``max(1, len(text) // 4)`` (~4 chars/token) if the tokenizer cannot be
    loaded or if TOKEN_COUNTING_MODE is "approx".
    """
    if not text or not text.strip():
        return 1

    tok = _load_tokenizer()

    if tok is not None:
        # add_special_tokens=False: exclude CLS/SEP from the count so that the
        # result reflects content tokens, consistent with how CHUNK_MAX_TOKENS
        # is meant to be interpreted (content budget, not model input length).
        ids = tok.encode(text, add_special_tokens=False)
        return max(1, len(ids))

    # Char-based fallback — warn once per process.
    global _fallback_warned
    if not _fallback_warned:
        logger.debug("token counting: using char-based approximation (len(text) // 4)")
        _fallback_warned = True

    return max(1, len(text) // 4)


def reset_tokenizer_cache() -> None:
    """Reset cached tokenizer state so the next call reloads from scratch.

    Call this in test teardowns when you need to simulate an unavailable
    tokenizer or switch between TOKEN_COUNTING_MODE values.
    """
    global _tokenizer, _load_attempted, _fallback_warned
    _tokenizer = None
    _load_attempted = False
    _fallback_warned = False
