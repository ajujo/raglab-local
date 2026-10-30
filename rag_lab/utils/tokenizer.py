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

# Backend (generative) tokenizer cache — used for prompt budgeting (RAG-08).
_backend_tokenizer: Optional[Any] = None
_backend_attempted: bool = False


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

        # local_files_only=True: never attempt a network download.
        # If the tokenizer is not in the HuggingFace local cache it fails
        # immediately (OSError, no hang), triggering the fallback below.
        # BAAI/bge-m3 tokenizer files are always present on systems that have
        # run the embedding pipeline at least once.
        tok = AutoTokenizer.from_pretrained(
            TOKENIZER_MODEL_NAME,
            use_fast=True,
            local_files_only=True,
        )
        _tokenizer = tok
        logger.debug("token counting: loaded tokenizer '%s'", TOKENIZER_MODEL_NAME)
        return _tokenizer

    except Exception as exc:  # ImportError, OSError (not in cache), etc.
        logger.warning(
            "token counting: tokenizer not available locally (%s); "
            "falling back to char-based approximation (len(text) // 4). "
            "Set TOKEN_COUNTING_MODE='approx' to silence this warning.",
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
    global _backend_tokenizer, _backend_attempted
    _tokenizer = None
    _load_attempted = False
    _fallback_warned = False
    _backend_tokenizer = None
    _backend_attempted = False


def _load_backend_tokenizer() -> Optional[Any]:
    """Load the generative backend's tokenizer if one is configured.

    Uses ``LLM_TOKENIZER_PATH`` (local dir, offline) or ``LLM_TOKENIZER_MODEL``
    (HF id in local cache).  Returns None when neither is configured or the
    load fails, in which case :func:`count_backend_tokens` falls back to the
    conservative estimator below.
    """
    global _backend_tokenizer, _backend_attempted
    if _backend_attempted:
        return _backend_tokenizer
    _backend_attempted = True
    try:
        from rag_lab.config import LLM_TOKENIZER_PATH, LLM_TOKENIZER_MODEL
    except ImportError:
        return None
    target = LLM_TOKENIZER_PATH or LLM_TOKENIZER_MODEL
    if not target:
        return None
    try:
        from transformers import AutoTokenizer
        _backend_tokenizer = AutoTokenizer.from_pretrained(
            target, use_fast=True, local_files_only=True,
        )
        logger.debug("prompt budgeting: loaded backend tokenizer '%s'", target)
    except Exception as exc:
        logger.warning(
            "prompt budgeting: backend tokenizer '%s' unavailable (%s); "
            "using conservative estimator",
            target, exc,
        )
        _backend_tokenizer = None
    return _backend_tokenizer


def estimate_tokens(text: str) -> int:
    """Token estimate for the generative backend (RAG-08 / P0B-05).

    If a backend tokenizer is configured (``LLM_TOKENIZER_PATH`` or
    ``LLM_TOKENIZER_MODEL``) it is used, subject to the conservative
    ``LLM_TOKEN_ESTIMATE_FACTOR`` applied by the prompt builder.

    Otherwise this returns a **guaranteed upper bound**: the number of UTF-8
    bytes.  Every token produced by a byte-level BPE/Unigram tokenizer maps to
    at least one byte, so the real token count can never exceed the byte count.
    This is deliberately conservative (normal text is ~4 bytes/token) but it
    cannot under-count, so the configured context window can never be exceeded
    by an invalid request.  Configure the tokenizer to reclaim full context.
    """
    if text == "":
        return 1
    tok = _load_backend_tokenizer()
    if tok is not None:
        try:
            return max(1, len(tok.encode(text, add_special_tokens=False)))
        except Exception:
            pass
    try:
        return max(1, len(text.encode("utf-8")))
    except Exception:
        return max(1, len(text))


def count_backend_tokens(text: str) -> int:
    """Token count for budgeting: backend tokenizer if available, else a
    guaranteed byte-based upper bound."""
    return estimate_tokens(text)


def estimate_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    """Token estimate for a full prompt (system + user)."""
    return count_backend_tokens(system_prompt or "") + count_backend_tokens(user_prompt or "")
