"""Tests for eval_utils: strip_inline_citations_for_eval and is_abstention."""

import pytest

from rag_lab.evaluation.eval_utils import is_abstention, strip_inline_citations_for_eval


class TestStripInlineCitationsForEval:

    # ------------------------------------------------------------------
    # Happy-path cases
    # ------------------------------------------------------------------

    def test_single_citation_at_end_of_sentence(self):
        text = (
            "SDMX es un estándar internacional "
            "[[1] Fuente: SDMX_Glossary | Sección: Intro | Líneas: 1-5]. "
            "Fue creado para facilitar el intercambio."
        )
        result = strip_inline_citations_for_eval(text)
        assert "[[1]" not in result
        assert "Fuente:" not in result
        assert "SDMX es un estándar internacional." in result
        assert "Fue creado para facilitar el intercambio." in result

    def test_multiple_citations(self):
        text = (
            "El modelo [[1] Fuente: DocA | Sección: S1 | Líneas: 1-10] "
            "incluye varias capas [[2] Fuente: DocB | Sección: S2 | Líneas: 20-30]."
        )
        result = strip_inline_citations_for_eval(text)
        assert "[[1]" not in result
        assert "[[2]" not in result
        assert "El modelo" in result
        assert "incluye varias capas." in result

    def test_no_citations_unchanged(self):
        text = "SDMX is a standard for statistical data exchange."
        result = strip_inline_citations_for_eval(text)
        assert result == text

    def test_citation_mid_sentence(self):
        text = (
            "A codelist [[3] Fuente: SDMX_Glossary | Sección: Codelist | Líneas: 100-110] "
            "defines allowed values."
        )
        result = strip_inline_citations_for_eval(text)
        assert "Fuente:" not in result
        assert "A codelist" in result
        assert "defines allowed values." in result

    def test_citation_with_long_doc_id(self):
        text = (
            "See section [[4] Fuente: SDMX_2-1_User_Guide_6 | "
            "Sección: 4.3.2 Data Sets | Líneas: 541-555] for details."
        )
        result = strip_inline_citations_for_eval(text)
        assert "Fuente:" not in result
        assert "See section" in result
        assert "for details." in result

    def test_citation_with_special_section_name(self):
        text = (
            "El estándar [[5] Fuente: SDMX_Glossary | "
            "Sección: Statistical Data and Metadata eXchange, SDMX | "
            "Líneas: 6738-6767] es ampliamente adoptado."
        )
        result = strip_inline_citations_for_eval(text)
        assert "Fuente:" not in result
        assert "El estándar" in result
        assert "es ampliamente adoptado." in result

    def test_no_double_spaces_after_strip(self):
        text = "First [[1] Fuente: A | Sección: S | Líneas: 1-2] second."
        result = strip_inline_citations_for_eval(text)
        assert "  " not in result

    def test_empty_string(self):
        assert strip_inline_citations_for_eval("") == ""

    def test_answer_is_only_citations(self):
        # Degenerate: the entire text is a citation block
        text = "[[1] Fuente: A | Sección: S | Líneas: 1-2]"
        result = strip_inline_citations_for_eval(text)
        assert result == ""

    # ------------------------------------------------------------------
    # Safety: must NOT strip normal square brackets
    # ------------------------------------------------------------------

    def test_regular_square_brackets_preserved(self):
        text = "The list contains [1, 2, 3] elements."
        result = strip_inline_citations_for_eval(text)
        assert result == text

    def test_single_number_brackets_preserved(self):
        # [1] alone (not [[1] Fuente:...]) must NOT be stripped
        text = "See reference [1] for more information."
        result = strip_inline_citations_for_eval(text)
        assert result == text

    def test_word_in_brackets_preserved(self):
        text = "The [important] concept is well defined."
        result = strip_inline_citations_for_eval(text)
        assert result == text

    def test_markdown_code_blocks_preserved(self):
        text = "Use `[key]` to access the value in `data[0]`."
        result = strip_inline_citations_for_eval(text)
        assert result == text

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_citation_at_very_start(self):
        text = "[[1] Fuente: DocA | Sección: Intro | Líneas: 1-5] This is the answer."
        result = strip_inline_citations_for_eval(text)
        assert result.startswith("This is the answer.")

    def test_consecutive_citations_no_double_space(self):
        text = (
            "Concept [[1] Fuente: A | Sección: S | Líneas: 1-2]"
            "[[2] Fuente: B | Sección: T | Líneas: 3-4] is important."
        )
        result = strip_inline_citations_for_eval(text)
        assert "  " not in result
        assert "Concept" in result
        assert "is important." in result

    def test_preserves_substantive_content_length(self):
        # After stripping, substantive content should remain
        text = (
            "SDMX stands for Statistical Data and Metadata eXchange "
            "[[1] Fuente: Glossary | Sección: S | Líneas: 1-10]. "
            "It is an ISO standard [[2] Fuente: ISO | Sección: T | Líneas: 5-15]."
        )
        result = strip_inline_citations_for_eval(text)
        assert len(result) > 50
        assert "SDMX stands for Statistical Data and Metadata eXchange." in result
        assert "It is an ISO standard." in result

    def test_idempotent(self):
        # Applying twice should give the same result as applying once
        text = (
            "A concept [[1] Fuente: DocA | Sección: S | Líneas: 1-5] explained."
        )
        once = strip_inline_citations_for_eval(text)
        twice = strip_inline_citations_for_eval(once)
        assert once == twice


class TestIsAbstention:

    def test_empty_answer_is_abstention(self):
        assert is_abstention("") is True

    def test_none_like_blank_is_abstention(self):
        assert is_abstention("   ") is True

    def test_short_answer_is_abstention(self):
        assert is_abstention("No sé.") is True

    def test_exactly_30_chars_is_abstention(self):
        assert is_abstention("A" * 30) is True

    def test_31_chars_without_pattern_is_not_abstention(self):
        assert is_abstention("A" * 31) is False

    def test_low_trust_score_is_abstention(self):
        assert is_abstention("SDMX is a data standard used worldwide.", 0.24) is True

    def test_trust_score_exactly_025_is_not_abstention_alone(self):
        # 0.25 is the boundary — NOT below it, so must not trigger by trust alone
        assert is_abstention("SDMX is a data standard used worldwide.", 0.25) is False

    def test_spanish_no_encuentro(self):
        assert is_abstention("No encuentro información sobre los valores permitidos del atributo.", None) is True

    def test_spanish_no_se_menciona(self):
        assert is_abstention("Ese dato no se menciona en los documentos disponibles.", None) is True

    def test_spanish_no_tengo_informacion(self):
        assert is_abstention("No tengo información sobre ese elemento en el corpus actual.", None) is True

    def test_spanish_no_esta_disponible(self):
        assert is_abstention("La información requerida no está disponible en el contexto.", None) is True

    def test_english_not_found(self):
        assert is_abstention("The requested information is not found in the provided documents.", None) is True

    def test_english_not_available(self):
        assert is_abstention("This information is not available in the current corpus.", None) is True

    def test_english_context_does_not_contain(self):
        assert is_abstention("The context does not contain any information about the mandatory header elements.", None) is True

    def test_english_cannot_find(self):
        assert is_abstention("I cannot find any reference to this in the provided documents.", None) is True

    def test_normal_sdmx_answer_is_not_abstention(self):
        answer = (
            "SDMX (Statistical Data and Metadata eXchange) is an ISO standard "
            "designed to facilitate the exchange of statistical information."
        )
        assert is_abstention(answer, 0.85) is False

    def test_answer_with_some_negation_not_abstention(self):
        # "not only" is a common phrase that must not match
        answer = (
            "SDMX is not only used for data exchange but also for metadata management "
            "across international statistical organizations."
        )
        assert is_abstention(answer, 0.75) is False
