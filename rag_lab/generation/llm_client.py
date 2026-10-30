"""LLM client for generation.

Wraps the OpenAI-compatible API for SGLang/LM Studio.

Qwen 3.x models use a thinking mode that splits output into:
  - reasoning_content: internal chain-of-thought (hidden from user)
  - content: the actual response

Some servers (LM Studio) do NOT support disabling thinking mode via
chat_template_kwargs or /no_think tokens. In that case, we must:
  1. Set max_tokens high enough for reasoning + content
  2. Always prefer the content field
  3. Fall back to reasoning_content only as a last resort, cleaning it
"""

import logging
import re
from typing import Optional

from rag_lab.config import (
    LLM_BASE_URL,
    LLM_CONTEXT_SAFETY_MARGIN,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    LLM_MIN_OUTPUT_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_THINKING_TOKEN_MULTIPLIER,
)
from rag_lab.exceptions import LLMConnectionError, LLMContextOverflowError
from rag_lab.utils.tokenizer import estimate_prompt_tokens

logger = logging.getLogger("rag_lab")

# Qwen3 thinking models need extra token budget for their internal reasoning.
# With complex RAG prompts, reasoning can use 500-2000+ tokens before producing
# the actual content. We multiply the configured max_tokens to ensure there's
# room for both reasoning and the final answer.
_THINKING_TOKEN_MULTIPLIER = LLM_THINKING_TOKEN_MULTIPLIER


def _get_client():
    """Create and return an OpenAI client for the local LLM server."""
    from openai import OpenAI
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key="local",
    )


def _extract_content(choice) -> str:
    """Extract the response text from a completion choice.

    Prefers the content field. Falls back to reasoning_content only if
    content is empty, and strips the thinking-process boilerplate.

    Args:
        choice: A completion choice object from the API response.

    Returns:
        The extracted text content.
    """
    message = choice.message

    # Prefer content (the actual response after thinking)
    content = (getattr(message, 'content', None) or "").strip()

    if content:
        return content

    # Fallback: extract useful text from reasoning_content
    reasoning = (getattr(message, 'reasoning_content', None) or "").strip()

    if reasoning:
        logger.warning(
            "LLM returned empty content — thinking mode consumed all output. "
            "Using reasoning_content as fallback."
        )
        # Try to extract the actual answer from the reasoning chain.
        # Qwen3 often embeds a draft answer inside the reasoning with markers
        # like "Draft Response:", "Respuesta:", "Answer:", etc.
        cleaned = _extract_answer_from_reasoning(reasoning)
        return cleaned

    # Nothing at all
    if choice.finish_reason == "length":
        logger.error(
            "LLM finished with reason 'length' — max_tokens exhausted by "
            "thinking. Increase LLM_MAX_TOKENS in config."
        )

    return ""


def _extract_answer_from_reasoning(reasoning: str) -> str:
    """Try to extract a structured answer from reasoning_content.

    Qwen3 thinking output often contains a draft answer section.
    We try to find it; otherwise return the full reasoning (it's
    better than nothing).

    Args:
        reasoning: The raw reasoning_content string.

    Returns:
        The best-effort extracted answer.
    """
    # Common markers Qwen3 uses inside its thinking for the "final" answer
    markers = [
        r"(?:Draft Response|Respuesta|Final Answer|Answer|Borrador)[^:]*:\s*\n",
        r"(?:\d+\.\s*\*\*(?:Draft|Formulate|Synthesize)[^*]*\*\*[^:]*:\s*\n)",
    ]

    for pattern in markers:
        match = re.search(pattern, reasoning, re.IGNORECASE)
        if match:
            # Extract everything after the marker
            extracted = reasoning[match.end():].strip()
            # Stop at the next numbered section (e.g., "5. **Check")
            next_section = re.search(r'\n\d+\.\s+\*\*', extracted)
            if next_section:
                extracted = extracted[:next_section.start()].strip()
            if len(extracted) > 50:  # Only use if substantial
                return extracted

    # No good marker found — return full reasoning
    return reasoning


def generate_response(
    system_prompt: str,
    user_prompt: str,
    temperature: float = None,
    max_tokens: int = None,
    timeout: Optional[float] = None,
    force_no_thinking: bool = False,
    prompt_tokens: int = None,
    context_window: int = None,
    total_max_tokens: int = None,
) -> str:
    """Generate a response from the LLM.

    Handles Qwen 3.x thinking models by allocating extra tokens for
    the internal reasoning process. Tries to disable thinking mode
    via chat_template_kwargs (works on SGLang), but gracefully handles
    servers that ignore it (LM Studio).

    Args:
        system_prompt: System message for the LLM.
        user_prompt: User message with context and question.
        temperature: Sampling temperature (default from config).
        max_tokens: Desired max tokens for the ANSWER (default from config).
            Internally multiplied by _THINKING_TOKEN_MULTIPLIER unless
            force_no_thinking=True.
        timeout: Request timeout in seconds. None = no timeout.
        force_no_thinking: When True, skip the thinking token multiplier and
            always pass enable_thinking=False. Use for short-output calls
            (HyDE, query rewriting) where thinking is suppressed at the server.
        prompt_tokens: Estimated prompt token count. When provided (RAG-08),
            the output budget is capped so that prompt + output + safety margin
            never exceeds ``context_window``. Prevents HTTP 400
            ``exceed_context_size_error``.
        context_window: Model context window (defaults to ``LLM_CONTEXT_WINDOW``).
        total_max_tokens: Final output budget that already includes the
            thinking multiplier (as produced by PromptPlan.output_tokens).
            When set, it takes precedence over ``max_tokens`` and is still
            capped against the window when ``prompt_tokens`` is given.

    Returns:
        The LLM's response as a string.

    Raises:
        LLMContextOverflowError: if no valid output budget fits in the window.
        LLMConnectionError: If the LLM server is unavailable.
    """
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    desired_tokens = max_tokens or LLM_MAX_TOKENS

    if total_max_tokens is not None:
        # Already-multiplied budget from PromptPlan.
        actual_max_tokens = total_max_tokens
    elif force_no_thinking:
        # Thinking is suppressed — allocate exactly the desired token budget.
        actual_max_tokens = desired_tokens
    else:
        # Allocate extra budget for thinking tokens
        actual_max_tokens = desired_tokens * _THINKING_TOKEN_MULTIPLIER

    # Budget the prompt centrally so no caller can bypass the check.  When the
    # caller already budgeted with PromptPlan the value is passed in; otherwise
    # estimate it here (covers consistency, HyDE, rewriting, etc.).  A supplied
    # value can only *raise* the estimate, never lower it.
    supplied = prompt_tokens or 0
    estimated = estimate_prompt_tokens(system_prompt, user_prompt)
    prompt_tokens = max(supplied, estimated)

    window = context_window if context_window is not None else LLM_CONTEXT_WINDOW
    available = window - prompt_tokens - LLM_CONTEXT_SAFETY_MARGIN
    if available < LLM_MIN_OUTPUT_TOKENS:
        raise LLMContextOverflowError(
            "Not enough context window left for a minimum answer: "
            f"prompt≈{prompt_tokens}, window={window}, "
            f"margin={LLM_CONTEXT_SAFETY_MARGIN}. "
            "Reduce the retrieved context or the question.",
            required_tokens=prompt_tokens + LLM_MIN_OUTPUT_TOKENS,
            budget_tokens=window - LLM_CONTEXT_SAFETY_MARGIN,
            window=window,
        )
    actual_max_tokens = min(actual_max_tokens, available)

    try:
        client = _get_client()

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=actual_max_tokens,
            timeout=timeout,
            extra_body={
                # Try to disable thinking (works on SGLang, ignored by LM Studio)
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        choice = response.choices[0]
        content = _extract_content(choice)

        # Log token usage for debugging
        usage = getattr(response, 'usage', None)
        if usage:
            reasoning_tokens = 0
            details = getattr(usage, 'completion_tokens_details', None)
            if details:
                reasoning_tokens = getattr(details, 'reasoning_tokens', 0) or 0
            logger.debug(
                "Token usage — prompt: %d, completion: %d (reasoning: %d)",
                usage.prompt_tokens,
                usage.completion_tokens,
                reasoning_tokens,
            )

        if not content:
            logger.warning("LLM returned empty content")

        return content

    except LLMContextOverflowError:
        raise
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if (
            "exceed_context_size" in low
            or "exceeds the context" in low
            or ("context" in low and ("exceed" in low or "too long" in low or "maximum" in low))
            or "n_ctx" in low
        ):
            raise LLMContextOverflowError(
                f"LLM backend rejected the prompt as exceeding its context window: {msg}",
            ) from e
        raise LLMConnectionError(
            f"Failed to connect to LLM server at {LLM_BASE_URL}: {e}"
        )


def generate_response_with_thinking(
    system_prompt: str,
    user_prompt: str,
    temperature: float = None,
    max_tokens: int = None,
) -> tuple[str, str]:
    """Generate a response WITH thinking mode, returning both parts.

    Use this when you want the model's chain-of-thought reasoning
    (e.g. for HyDE or complex analysis).

    Args:
        system_prompt: System message for the LLM.
        user_prompt: User message with context and question.
        temperature: Sampling temperature (default from config).
        max_tokens: Max tokens in response (default from config).

    Returns:
        Tuple of (content, reasoning_content).

    Raises:
        LLMConnectionError: If the LLM server is unavailable.
    """
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    max_tokens = max_tokens or (LLM_MAX_TOKENS * _THINKING_TOKEN_MULTIPLIER)

    # RAG-08: budget this entry point too, so it cannot bypass the window.
    prompt_tokens = estimate_prompt_tokens(system_prompt, user_prompt)
    available = LLM_CONTEXT_WINDOW - prompt_tokens - LLM_CONTEXT_SAFETY_MARGIN
    if available < LLM_MIN_OUTPUT_TOKENS:
        raise LLMContextOverflowError(
            "Not enough context window left for a minimum answer: "
            f"prompt≈{prompt_tokens}, window={LLM_CONTEXT_WINDOW}.",
            required_tokens=prompt_tokens + LLM_MIN_OUTPUT_TOKENS,
            budget_tokens=LLM_CONTEXT_WINDOW - LLM_CONTEXT_SAFETY_MARGIN,
            window=LLM_CONTEXT_WINDOW,
        )
    max_tokens = min(max_tokens, available)

    try:
        client = _get_client()

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        message = choice.message
        content = (getattr(message, 'content', None) or "").strip()
        reasoning = (getattr(message, 'reasoning_content', None) or "").strip()

        if not content and choice.finish_reason == "length":
            logger.warning(
                "Thinking mode exhausted max_tokens (%d). "
                "Increase max_tokens or disable thinking.",
                max_tokens,
            )

        return content, reasoning

    except LLMContextOverflowError:
        raise
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if (
            "exceed_context_size" in low
            or "exceeds the context" in low
            or ("context" in low and ("exceed" in low or "too long" in low or "maximum" in low))
            or "n_ctx" in low
        ):
            raise LLMContextOverflowError(
                f"LLM backend rejected the prompt as exceeding its context window: {msg}",
            ) from e
        raise LLMConnectionError(
            f"Failed to connect to LLM server at {LLM_BASE_URL}: {e}"
        )