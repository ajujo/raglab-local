"""Tests para el módulo de query rewriting."""

import pytest
from rag_lab.retrieval.query_rewriter import (
    rewrite_query,
    QUERY_REWRITER_SYSTEM_PROMPT,
    QUERY_REWRITER_USER_PROMPT_TEMPLATE,
)


class TestRewriteQuery:
    """Tests para la función rewrite_query."""

    def test_rewrite_success(self):
        def mock_llm(prompt):
            return "¿Qué es un Data Structure Definition (DSD) en SDMX?"

        result = rewrite_query("¿Qué es DSD?", mock_llm)
        assert result == "¿Qué es un Data Structure Definition (DSD) en SDMX?"

    def test_rewrite_empty_fallback(self):
        def mock_llm(prompt):
            return ""

        result = rewrite_query("¿Qué es DSD?", mock_llm)
        assert result == "¿Qué es DSD?"

    def test_rewrite_exception_fallback(self):
        def mock_llm(prompt):
            raise Exception("Simulated LLM failure")

        result = rewrite_query("¿Qué es DSD?", mock_llm)
        assert result == "¿Qué es DSD?"

    def test_rewrite_preserves_meaning(self):
        def mock_llm(prompt):
            return "What is the purpose of SDMX-ML and SDMX-EDI formats?"

        result = rewrite_query("What is SDMX-ML?", mock_llm)
        assert "SDMX-ML" in result or "SDMX-EDI" in result


class TestPrompts:
    """Tests para los prompts de rewriting."""

    def test_system_prompt_exists(self):
        assert len(QUERY_REWRITER_SYSTEM_PROMPT) > 0
        assert "búsqueda semántica" in QUERY_REWRITER_SYSTEM_PROMPT

    def test_user_prompt_template(self):
        formatted = QUERY_REWRITER_USER_PROMPT_TEMPLATE.format(question="¿Qué es DSD?")
        assert "¿Qué es DSD?" in formatted
        assert "siglas" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
