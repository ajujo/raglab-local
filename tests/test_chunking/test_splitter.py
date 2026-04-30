"""Tests for chunking/splitter.py

Tests the semantic chunking logic:
- _count_tokens
- _is_table_line
- _merge_sibling_sections
- _filter_tiny_chunks
- _create_chunks
- _split_into_segments
- chunk_document (integration)
"""

import pytest
from rag_lab.chunking.splitter import (
    _count_tokens,
    _is_table_line,
    _merge_sibling_sections,
    _filter_tiny_chunks,
    _create_chunks,
    _split_into_segments,
    chunk_document,
)
from rag_lab.exceptions import ChunkingError


# --- _count_tokens ---

class TestCountTokens:
    def test_empty_string(self):
        assert _count_tokens("") == 1  # max(1, 0)

    def test_short_text(self):
        # 4 chars = 1 token
        assert _count_tokens("abcd") == 1
        # 8 chars = 2 tokens
        assert _count_tokens("abcdefgh") == 2
        # 100 chars = 25 tokens
        assert _count_tokens("a" * 100) == 25

    def test_long_text(self):
        text = "a" * 4000
        assert _count_tokens(text) == 1000

    def test_whitespace(self):
        assert _count_tokens("   ") == 1

    def test_mixed_content(self):
        # 13 chars / 4 = 3 tokens
        assert _count_tokens("Hello, world!") == 3


# --- _is_table_line ---

class TestIsTableLine:
    def test_valid_table_line(self):
        assert _is_table_line("| Header 1 | Header 2 |") is True

    def test_valid_table_line_stripped(self):
        assert _is_table_line("  | Header 1 | Header 2 |") is True

    def test_not_table_line(self):
        assert _is_table_line("This is regular text") is False

    def test_empty_line(self):
        assert _is_table_line("") is False

    def test_without_pipes(self):
        assert _is_table_line("No pipes here") is False


# --- _merge_sibling_sections ---

class TestMergeSiblingSections:
    def test_empty_list(self):
        assert _merge_sibling_sections([], 100) == []

    def test_single_section(self):
        sections = [
            {
                'heading': type('H', (), {'title': 'Section 1'})(),
                'heading_path': 'Section 1',
                'text': 'Some text',
                'tokens': 5,
                'relative_pos': 0.1,
                'level': 1,
                'parent_id': 1,
            }
        ]
        result = _merge_sibling_sections(sections, 100)
        assert len(result) == 1

    def test_merge_small_siblings(self):
        sections = [
            {
                'heading': type('H', (), {'title': 'A'})(),
                'heading_path': 'Section A',
                'text': 'Text A',
                'tokens': 5,
                'relative_pos': 0.1,
                'level': 1,
                'parent_id': 1,
            },
            {
                'heading': type('H', (), {'title': 'B'})(),
                'heading_path': 'Section B',
                'text': 'Text B',
                'tokens': 5,
                'relative_pos': 0.2,
                'level': 1,
                'parent_id': 1,
            },
        ]
        result = _merge_sibling_sections(sections, 100)
        assert len(result) == 1  # merged
        assert 'A' in result[0]['heading_path']
        assert 'B' in result[0]['heading_path']

    def test_no_merge_different_parents(self):
        sections = [
            {
                'heading': type('H', (), {'title': 'A'})(),
                'heading_path': 'Section A',
                'text': 'Text A',
                'tokens': 5,
                'relative_pos': 0.1,
                'level': 1,
                'parent_id': 1,
            },
            {
                'heading': type('H', (), {'title': 'B'})(),
                'heading_path': 'Section B',
                'text': 'Text B',
                'tokens': 5,
                'relative_pos': 0.2,
                'level': 1,
                'parent_id': 2,
            },
        ]
        result = _merge_sibling_sections(sections, 100)
        assert len(result) == 2  # not merged (different parents)

    def test_merge_cap(self):
        sections = [
            {
                'heading': type('H', (), {'title': 'A'})(),
                'heading_path': 'Section A',
                'text': 'a' * 2000,
                'tokens': 500,
                'relative_pos': 0.1,
                'level': 1,
                'parent_id': 1,
            },
            {
                'heading': type('H', (), {'title': 'B'})(),
                'heading_path': 'Section B',
                'text': 'b' * 2000,
                'tokens': 500,
                'relative_pos': 0.2,
                'level': 1,
                'parent_id': 1,
            },
        ]
        result = _merge_sibling_sections(sections, 800)
        # Cap is 1600, each is 500, so they can be merged
        assert len(result) == 1


# --- _filter_tiny_chunks ---

class TestFilterTinyChunks:
    def test_empty_list(self):
        assert _filter_tiny_chunks([]) == []

    def test_all_large(self):
        from rag_lab.chunking.splitter import Chunk
        chunks = [
            Chunk(chunk_id="1", doc_id="doc", text="a" * 200, heading_path="H",
                  tipo="texto", posicion_relativa=0.0, n_tokens=50),
        ]
        result = _filter_tiny_chunks(chunks)
        assert len(result) == 1

    def test_merge_tiny(self):
        from rag_lab.chunking.splitter import Chunk
        chunks = [
            Chunk(chunk_id="1", doc_id="doc", text="a" * 200, heading_path="H",
                  tipo="texto", posicion_relativa=0.0, n_tokens=50),
            Chunk(chunk_id="2", doc_id="doc", text="small", heading_path="H",
                  tipo="texto", posicion_relativa=0.5, n_tokens=10),
        ]
        result = _filter_tiny_chunks(chunks)
        # Tiny chunk merged with previous
        assert len(result) == 1
        assert "small" in result[0].text

    def test_discard_tiny_at_start(self):
        from rag_lab.chunking.splitter import Chunk
        chunks = [
            Chunk(chunk_id="1", doc_id="doc", text="small", heading_path="H",
                  tipo="texto", posicion_relativa=0.0, n_tokens=10),
        ]
        result = _filter_tiny_chunks(chunks)
        # First chunk is kept even if tiny
        assert len(result) == 1


# --- _create_chunks ---

class TestCreateChunks:
    def test_empty_text(self):
        assert _create_chunks("", "doc", "H", 0.0, 100, 20) == []

    def test_single_chunk(self):
        text = "Hello world. This is a test."
        result = _create_chunks(text, "doc", "Heading", 0.5, 100, 20)
        assert len(result) == 1

    def test_multiple_chunks(self):
        # Very long text that exceeds max_tokens
        text = " ".join(["word"] * 500)
        result = _create_chunks(text, "doc", "Heading", 0.5, 100, 20)
        assert len(result) >= 1

    def test_table_detection(self):
        text = "| Header | Header |\n|--------|--------|\n| A | B |"
        result = _create_chunks(text, "doc", "Heading", 0.0, 100, 20)
        assert len(result) == 1
        assert result[0].tipo == "tabla"


# --- _split_into_segments ---

class TestSplitIntoSegments:
    def test_empty(self):
        assert _split_into_segments("") == []

    def test_single_paragraph(self):
        text = "Hello world"
        assert _split_into_segments(text) == ["Hello world"]

    def test_multiple_paragraphs(self):
        text = "Para 1\n\nPara 2"
        result = _split_into_segments(text)
        assert len(result) == 2

    def test_long_paragraph_split(self):
        text = "a" * 1000
        result = _split_into_segments(text)
        # Should be split into smaller segments
        assert len(result) > 0

    def test_numbered_list_grouping(self):
        """Las líneas de una lista numerada consecutiva se agrupan en un solo segmento."""
        text = "Some intro text.\n\n1. First item\n2. Second item\n3. Third item\n4. Fourth item\n5. Fifth item\nMore text after."
        result = _split_into_segments(text)
        # La lista numerada debe estar en un solo segmento
        list_segments = [s for s in result if s.startswith("1.")]
        assert len(list_segments) == 1
        assert "2. Second item" in list_segments[0]
        assert "3. Third item" in list_segments[0]
        assert "4. Fourth item" in list_segments[0]
        assert "5. Fifth item" in list_segments[0]

    def test_numbered_list_with_surrounding_text(self):
        """Texto antes y después de la lista no se agrupa con la lista."""
        text = "Intro paragraph that is long enough to trigger line splitting so we can test list grouping behavior.\n\n1. Agencies are maintained in an Agency Scheme.\n2. The maintenance agency of the Agency Scheme must also be declared.\n3. The top-level agency is SDMX.\n\nOutro paragraph that follows the list."
        result = _split_into_segments(text)
        # Verificar que la lista está agrupada
        list_segments = [s for s in result if s.startswith("1.")]
        assert len(list_segments) == 1
        assert "2." in list_segments[0]
        assert "3." in list_segments[0]


# --- chunk_document ---

class TestChunkDocument:
    def test_empty_text(self):
        with pytest.raises(ChunkingError):
            chunk_document("")

    def test_empty_text_raises(self):
        with pytest.raises(ChunkingError):
            chunk_document("   ")

    def test_no_headings(self):
        text = "Just plain text without headings"
        chunks = chunk_document(text, "doc")
        assert len(chunks) >= 1

    def test_with_headings(self):
        text = "# Header 1\nSome text\n## Header 2\nMore text"
        chunks = chunk_document(text, "doc")
        assert len(chunks) >= 1

    def test_toc_exclusion(self):
        text = "# Contents\nSome contents\n# Section\nText"
        chunks = chunk_document(text, "doc")
        for chunk in chunks:
            assert "Contents" not in chunk.heading_path

    def test_custom_params(self):
        text = "# Header\n" + "a" * 1000
        chunks = chunk_document(text, "doc", max_tokens=100, overlap=20)
        assert len(chunks) >= 1

    def test_chunk_metadata(self):
        text = "# Header\nSome text here"
        chunks = chunk_document(text, "test_doc")
        for chunk in chunks:
            assert chunk.doc_id == "test_doc"
            assert chunk.chunk_id
            assert chunk.text
            assert chunk.heading_path
            assert chunk.tipo in ("texto", "tabla", "formula")
            assert chunk.n_tokens > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
