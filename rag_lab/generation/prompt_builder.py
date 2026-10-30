"""Build prompts for the LLM.

Constructs system and user prompts for the RAG generation phase.

As of the P0 corrective work (RAG-08) prompt construction is *budgeted*: it
counts tokens against the model context window and drops the lowest-ranked
chunks until the prompt fits, leaving room for the model's output.  If even the
system prompt plus the question cannot fit, :class:`LLMContextOverflowError` is
raised instead of silently truncating.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from rag_lab.config import (
    LLM_CONTEXT_SAFETY_MARGIN,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    LLM_MIN_OUTPUT_TOKENS,
    LLM_TOKEN_ESTIMATE_FACTOR,
    LLM_THINKING_TOKEN_MULTIPLIER,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from rag_lab.exceptions import LLMContextOverflowError

logger = logging.getLogger("rag_lab")


@dataclass
class PromptPlan:
    """Result of :func:`build_prompt_plan`.

    ``used_chunks`` are the chunks actually included, in ranking order.
    ``dropped_chunks`` are the chunks excluded to satisfy the token budget
    (never a silent truncation: callers can log/report them).
    """

    system: str
    user: str
    used_chunks: List[dict] = field(default_factory=list)
    dropped_chunks: List[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0


def _render_chunk(index: int, chunk: dict) -> str:
    heading = chunk.get("heading_path", "Sin encabezado")
    text = chunk.get("text", "")
    line_start = chunk.get("line_start", "N/A")
    line_end = chunk.get("line_end", "N/A")
    return (
        f"[{index}] Fuente: {chunk.get('doc_id', 'N/A')} | "
        f"Sección: {heading} | Líneas: {line_start}-{line_end}\n---\n{text}\n"
    )


def _context_text(chunks: List[dict]) -> str:
    return "".join(_render_chunk(i, c) for i, c in enumerate(chunks, 1))


def _build_user_prompt(context: str, query: str) -> str:
    return USER_PROMPT_TEMPLATE.format(context=context, question=query)


def _resolve_token_counter() -> Callable[[str], int]:
    """Return the best available token counter, falling back to len//4.

    Uses the conservative :func:`estimate_tokens` (BGE tokenizer with a
    byte-based safety net) so pathological inputs cannot under-count.
    Imported lazily so tests can monkeypatch it and so the module never
    triggers a network download at import time.
    """
    from rag_lab.utils.tokenizer import count_backend_tokens

    return count_backend_tokens


def build_prompt_plan(
    query: str,
    chunks: List[dict],
    *,
    system_prompt: str = None,
    window: int = None,
    max_output_tokens: int = None,
    safety_margin: int = None,
    min_output_tokens: int = None,
    estimate_factor: float = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> PromptPlan:
    """Build system/user prompts that fit the model context window.

    Args:
        query: User's question.
        chunks: Retrieved chunk dicts, best first.
        system_prompt: Override system prompt (default from config).
        window: Model context window in tokens (default ``LLM_CONTEXT_WINDOW``).
        max_output_tokens: Desired answer tokens *before* the thinking
            multiplier (default ``LLM_MAX_TOKENS``).
        safety_margin: Tokens reserved for tokenizer mismatch.
        min_output_tokens: Minimum acceptable output budget.
        estimate_factor: Conservative multiplier applied to prompt estimates.

    Returns:
        :class:`PromptPlan` with the fitted prompt and chunk accounting.

    Raises:
        LLMContextOverflowError: if system prompt + question alone cannot fit.
    """
    system = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    window = window if window is not None else LLM_CONTEXT_WINDOW
    safety_margin = safety_margin if safety_margin is not None else LLM_CONTEXT_SAFETY_MARGIN
    min_output_tokens = min_output_tokens if min_output_tokens is not None else LLM_MIN_OUTPUT_TOKENS
    estimate_factor = estimate_factor if estimate_factor is not None else LLM_TOKEN_ESTIMATE_FACTOR
    desired_output = max_output_tokens if max_output_tokens is not None else LLM_MAX_TOKENS

    counter = token_counter or _resolve_token_counter()

    def _estimate(text: str) -> int:
        # Conservative inflation to absorb tokenizer mismatch.
        return int(counter(text) * estimate_factor) + 1

    # Output budget the client will request (thinking multiplier included),
    # capped so the prompt always has room.
    requested_output = desired_output * LLM_THINKING_TOKEN_MULTIPLIER
    # Budget available to the prompt while guaranteeing min output + margin.
    prompt_budget = window - safety_margin - min_output_tokens

    base_user = _build_user_prompt("", query)
    base_tokens = _estimate(system) + _estimate(base_user)
    if base_tokens > prompt_budget:
        raise LLMContextOverflowError(
            "Prompt base (system prompt + question) exceeds the model context "
            f"budget ({base_tokens} > {prompt_budget} of window {window}). "
            "Reduce the system prompt or the question length.",
            required_tokens=base_tokens,
            budget_tokens=prompt_budget,
            window=window,
        )

    running = base_tokens
    used: List[dict] = []
    dropped: List[dict] = []
    for chunk in chunks:
        rendered = _render_chunk(len(used) + 1, chunk)
        cost = _estimate(rendered)
        if running + cost <= prompt_budget:
            used.append(chunk)
            running += cost
        else:
            dropped.append(chunk)

    context = _context_text(used)
    user = _build_user_prompt(context, query)
    prompt_tokens = _estimate(system) + _estimate(user)

    # Output budget that actually fits alongside the final prompt.
    available_output = window - safety_margin - prompt_tokens
    if available_output < min_output_tokens:
        raise LLMContextOverflowError(
            "Prompt leaves no room for a minimum answer: "
            f"prompt={prompt_tokens}, window={window}, "
            f"min_output={min_output_tokens}.",
            required_tokens=prompt_tokens,
            budget_tokens=window - safety_margin - min_output_tokens,
            window=window,
        )
    output_tokens = min(requested_output, available_output)

    if dropped:
        dropped_ids = [c.get("chunk_id", "?") for c in dropped]
        logger.warning(
            "Prompt budget: %d/%d chunks included; %d dropped to fit "
            "context window (prompt≈%d tokens, output budget=%d). Dropped ids: %s",
            len(used), len(used) + len(dropped), len(dropped),
            prompt_tokens, output_tokens, dropped_ids[:20],
        )

    return PromptPlan(
        system=system,
        user=user,
        used_chunks=used,
        dropped_chunks=dropped,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )


def build_prompt(
    query: str,
    chunks: List[dict],
) -> tuple[str, str]:
    """Build system and user prompts for the LLM.

    Backwards-compatible wrapper around :func:`build_prompt_plan`; returns
    ``(system_prompt, user_prompt)``.

    Args:
        query: The user's question.
        chunks: List of retrieved chunk dicts.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    plan = build_prompt_plan(query, chunks)
    logger.info(f"Built prompt with {len(plan.used_chunks)} chunks")
    return plan.system, plan.user
