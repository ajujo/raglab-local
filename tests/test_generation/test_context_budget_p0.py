"""RAG-08 regression: context-window budgeting.

These tests use an injected deterministic token counter so they do not depend
on the BGE tokenizer being present.  They assert the invariants:

  * prompt + output + safety margin never exceeds the window;
  * lowest-ranked chunks are dropped (not silently truncated) to fit;
  * an impossible base prompt raises LLMContextOverflowError;
  * generate_response caps max_tokens against the window and refuses to send
    when no minimum output budget is left.
"""

from unittest.mock import Mock, patch

import pytest

from rag_lab.config import (
    LLM_CONTEXT_SAFETY_MARGIN,
    LLM_CONTEXT_WINDOW,
    LLM_MIN_OUTPUT_TOKENS,
)
from rag_lab.exceptions import LLMContextOverflowError
from rag_lab.generation.prompt_builder import build_prompt, build_prompt_plan


def _counter(text: str) -> int:
    # Deterministic, cheap: 1 token per 4 chars.
    return max(1, len(text) // 4)


def _chunk(i: int, words: int = 40) -> dict:
    return {
        "chunk_id": f"c{i}",
        "doc_id": "doc",
        "heading_path": f"H{i}",
        "text": "palabra " * words,
        "line_start": i,
        "line_end": i + 1,
    }


class TestPromptBudget:
    def test_fits_within_window(self):
        chunks = [_chunk(i) for i in range(50)]
        plan = build_prompt_plan(
            "¿Qué es SDMX?", chunks, system_prompt="sys",
            window=1024, max_output_tokens=128, safety_margin=64,
            min_output_tokens=32, token_counter=_counter,
        )
        assert plan.prompt_tokens + plan.output_tokens + 64 <= 1024

    def test_drops_lowest_ranked_chunks(self):
        chunks = [_chunk(i) for i in range(50)]
        plan = build_prompt_plan(
            "question", chunks, system_prompt="sys", window=900,
            max_output_tokens=128, safety_margin=64,
            min_output_tokens=32, token_counter=_counter,
        )
        assert plan.dropped_chunks, "expected some chunks to be dropped"
        # Kept chunks must be a prefix of the ranking (best first).
        assert plan.used_chunks == chunks[: len(plan.used_chunks)]

    def test_huge_base_raises_explicit_error(self):
        with pytest.raises(LLMContextOverflowError):
            build_prompt_plan(
                "q" * 10000, [_chunk(0)],
                window=512, max_output_tokens=64, safety_margin=32,
                min_output_tokens=16, token_counter=_counter,
            )

    def test_output_budget_never_exceeds_requested(self):
        plan = build_prompt_plan(
            "q", [_chunk(i) for i in range(5)],
            window=8192, max_output_tokens=256, safety_margin=64,
            min_output_tokens=32, token_counter=_counter,
        )
        from rag_lab.config import LLM_THINKING_TOKEN_MULTIPLIER
        assert plan.output_tokens <= 256 * LLM_THINKING_TOKEN_MULTIPLIER

    def test_compat_build_prompt_still_returns_tuple(self):
        system, user = build_prompt("What is SDMX?", [_chunk(0)])
        assert isinstance(system, str) and isinstance(user, str)
        assert "palabra" in user


class TestGenerateResponseBudget:
    def _fake_client(self, captured):
        client = Mock()

        def create(**kwargs):
            captured.update(kwargs)
            resp = Mock()
            resp.choices = [Mock()]
            resp.choices[0].message.content = "ok"
            resp.choices[0].message.reasoning_content = ""
            resp.choices[0].finish_reason = "stop"
            resp.usage = None
            return resp

        client.chat.completions.create.side_effect = create
        return client

    def test_caps_max_tokens_against_window(self):
        captured = {}
        with patch("rag_lab.generation.llm_client._get_client",
                   return_value=self._fake_client(captured)):
            from rag_lab.generation.llm_client import generate_response
            generate_response(
                "sys", "user", max_tokens=2048,
                prompt_tokens=4000, context_window=8192,
            )
        assert captured["max_tokens"] <= 8192 - 4000 - LLM_CONTEXT_SAFETY_MARGIN

    def test_total_budget_from_plan_is_respected(self):
        captured = {}
        with patch("rag_lab.generation.llm_client._get_client",
                   return_value=self._fake_client(captured)):
            from rag_lab.generation.llm_client import generate_response
            generate_response(
                "sys", "user", prompt_tokens=1000,
                context_window=8192, total_max_tokens=3000,
            )
        assert captured["max_tokens"] == 3000

    def test_no_room_raises(self):
        with patch("rag_lab.generation.llm_client._get_client",
                   return_value=self._fake_client({})):
            from rag_lab.generation.llm_client import generate_response
            with pytest.raises(LLMContextOverflowError):
                generate_response(
                    "sys", "user", prompt_tokens=8192,
                    context_window=8192,
                )

    def test_real_config_defaults_are_consistent(self):
        # The shipped defaults must leave room for a minimum answer.
        assert LLM_CONTEXT_WINDOW - LLM_CONTEXT_SAFETY_MARGIN >= LLM_MIN_OUTPUT_TOKENS
