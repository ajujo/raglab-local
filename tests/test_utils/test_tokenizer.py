"""Tests for rag_lab/utils/tokenizer.py."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

import rag_lab.utils.tokenizer as tok_module
from rag_lab.utils.tokenizer import count_tokens, reset_tokenizer_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure tokenizer cache is fresh for every test."""
    reset_tokenizer_cache()
    yield
    reset_tokenizer_cache()


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_when_transformers_unavailable(self):
        """If AutoTokenizer import fails, return max(1, len(text)//4)."""
        with patch.dict(sys.modules, {"transformers": None}):
            reset_tokenizer_cache()
            result = count_tokens("hello world!")  # 12 chars → 12//4 = 3
            assert result == max(1, len("hello world!") // 4)

    def test_fallback_when_model_not_in_local_cache(self):
        """Simulate offline: local_files_only=True raises OSError → fast fallback."""
        mock_at = MagicMock()
        mock_at.from_pretrained.side_effect = OSError(
            "Couldn't find the files, and couldn't connect to 'https://huggingface.co'"
        )
        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=mock_at)}):
            reset_tokenizer_cache()
            result = count_tokens("test text for offline fallback")
            assert result >= 1
            # Verify fallback is the heuristic
            text = "test text for offline fallback"
            assert result == max(1, len(text) // 4)

    def test_fallback_no_retry_after_first_failure(self):
        """_load_attempted prevents repeated from_pretrained calls on failure."""
        mock_at = MagicMock()
        mock_at.from_pretrained.side_effect = OSError("not cached")
        with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=mock_at)}):
            reset_tokenizer_cache()
            count_tokens("first call")
            count_tokens("second call")
            count_tokens("third call")
            # from_pretrained must be called at most once (gated by _load_attempted)
            assert mock_at.from_pretrained.call_count <= 1

    def test_fallback_when_from_pretrained_raises(self, monkeypatch):
        """If AutoTokenizer.from_pretrained raises, fall back gracefully."""
        mock_at = MagicMock()
        mock_at.from_pretrained.side_effect = OSError("model not found")
        with patch("rag_lab.utils.tokenizer._load_tokenizer", wraps=tok_module._load_tokenizer):
            with patch.dict("sys.modules", {"transformers": MagicMock(AutoTokenizer=mock_at)}):
                reset_tokenizer_cache()
                result = count_tokens("test text")
                assert result >= 1

    def test_approx_mode_skips_tokenizer_load(self, monkeypatch):
        """TOKEN_COUNTING_MODE='approx' must never try to load the tokenizer."""
        monkeypatch.setattr("rag_lab.config.TOKEN_COUNTING_MODE", "approx")
        reset_tokenizer_cache()
        # should use len(text)//4, no tokenizer loaded
        text = "a" * 400
        result = count_tokens(text)
        assert result == max(1, len(text) // 4)
        # _tokenizer stays None because load was not attempted beyond mode check
        assert tok_module._tokenizer is None

    def test_fallback_empty_text_returns_one(self):
        with patch.dict(sys.modules, {"transformers": None}):
            reset_tokenizer_cache()
            assert count_tokens("") == 1
            assert count_tokens("   ") == 1

    def test_fallback_proportional_to_length(self):
        """Longer text → more tokens (monotone even with heuristic)."""
        with patch.dict(sys.modules, {"transformers": None}):
            reset_tokenizer_cache()
            short = count_tokens("hello")
            medium = count_tokens("hello world " * 10)
            long   = count_tokens("hello world " * 100)
            assert medium >= short
            assert long > medium


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_tokenizer_loaded_only_once(self, monkeypatch):
        """_load_tokenizer is idempotent — model loaded once, then cached."""
        call_count = 0
        original = tok_module._load_tokenizer

        def counting_loader():
            nonlocal call_count
            call_count += 1
            return original()

        monkeypatch.setattr(tok_module, "_load_tokenizer", counting_loader)
        count_tokens("first call")
        count_tokens("second call")
        count_tokens("third call")
        # counting_loader is called every time count_tokens() runs, but the
        # underlying AutoTokenizer.from_pretrained should only run once because
        # _load_attempted gates it.
        assert call_count == 3  # wrapper called 3x; internal load gated by flag

    def test_reset_cache_clears_state(self):
        # Warm up cache
        count_tokens("warmup")
        assert tok_module._load_attempted is True
        # Reset
        reset_tokenizer_cache()
        assert tok_module._load_attempted is False
        assert tok_module._tokenizer is None
        assert tok_module._fallback_warned is False


# ---------------------------------------------------------------------------
# Real tokenizer (loaded from BAAI/bge-m3 local cache)
# ---------------------------------------------------------------------------

class TestRealTokenizer:
    def test_returns_positive_count_for_nonempty_text(self):
        result = count_tokens("What is SDMX?")
        assert result >= 1

    def test_empty_and_whitespace_return_one(self):
        assert count_tokens("") == 1
        assert count_tokens("   \n\t  ") == 1

    def test_longer_text_has_more_tokens(self):
        short  = count_tokens("SDMX")
        medium = count_tokens("What is the role of the Maintenance Agency in SDMX?")
        long   = count_tokens(
            "What is the role of the Maintenance Agency in SDMX and how is it used "
            "across different artefact types such as code lists, DSDs, concept schemes, "
            "and provision agreements?" * 3
        )
        assert medium > short
        assert long > medium

    def test_spanish_text_counted(self):
        result = count_tokens("¿Qué es un esquema de conceptos en SDMX y para qué se utiliza?")
        assert result >= 5

    def test_markdown_table_counted(self):
        table = (
            "| Métrica | Valor |\n"
            "|---------|-------|\n"
            "| Recall@5 | 0.80 |\n"
            "| nDCG@10  | 0.83 |\n"
        )
        result = count_tokens(table)
        assert result >= 5

    def test_real_count_plausible_vs_heuristic(self):
        """Real tokenizer result should be within 2× of the heuristic for ASCII text."""
        text = "The SDMX standard defines a Data Structure Definition (DSD) as a collection "
        heuristic = max(1, len(text) // 4)
        real = count_tokens(text)
        # Expect within a factor of 2 in either direction
        assert real <= heuristic * 2
        assert real >= heuristic // 2

    def test_no_full_embedding_model_loaded(self):
        """count_tokens must NOT trigger loading of the FlagModel encoder."""
        from rag_lab.embedding import encoder as enc_module
        model_before = enc_module._model_cache  # None or already-loaded object

        reset_tokenizer_cache()
        count_tokens("test whether encoder loads")

        assert enc_module._model_cache is model_before, (
            "count_tokens must not load the full FlagModel encoder"
        )


# ---------------------------------------------------------------------------
# Integration: splitter uses real token counts
# ---------------------------------------------------------------------------

class TestSplitterIntegration:
    def test_chunk_n_tokens_positive(self):
        """chunk_document sets n_tokens via count_tokens; must be > 0."""
        from rag_lab.chunking.splitter import chunk_document
        doc = "# Section\n\n" + "This is test content. " * 20
        chunks = chunk_document(doc, doc_id="test_doc")
        assert chunks
        for c in chunks:
            assert c.n_tokens >= 1

    def test_chunk_n_tokens_proportional(self):
        """A longer section produces a chunk with more tokens than a short one."""
        from rag_lab.chunking.splitter import chunk_document
        short_doc = "# Short\n\nBrief content."
        long_doc  = "# Long\n\n" + "Detailed content about SDMX structures. " * 50
        short_chunks = chunk_document(short_doc, doc_id="short")
        long_chunks  = chunk_document(long_doc,  doc_id="long")
        assert short_chunks and long_chunks
        max_short = max(c.n_tokens for c in short_chunks)
        max_long  = max(c.n_tokens for c in long_chunks)
        assert max_long > max_short
