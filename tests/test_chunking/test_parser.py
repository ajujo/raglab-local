"""Tests for chunking/parser.py

Tests:
- parse_headings
- build_heading_tree
- get_heading_path
"""

import pytest
from rag_lab.chunking.parser import (
    parse_headings,
    build_heading_tree,
    get_heading_path,
    Heading,
)


class TestHeading:
    def test_path(self):
        h = Heading(level=1, title="Title", position=1)
        assert h.path == "Title"

    def test_add_child(self):
        parent = Heading(level=1, title="Parent", position=1)
        child = Heading(level=2, title="Child", position=2)
        parent.add_child(child)
        assert len(parent.children) == 1
        assert parent.children[0] == child


class TestParseHeadings:
    def test_empty_text(self):
        assert parse_headings("") == []

    def test_no_headings(self):
        text = "Just some plain text without headings"
        assert parse_headings(text) == []

    def test_single_heading(self):
        text = "# Header 1"
        headings = parse_headings(text)
        assert len(headings) == 1
        assert headings[0].level == 1
        assert headings[0].title == "Header 1"

    def test_multiple_headings(self):
        text = """# Header 1
Some text
## Header 2
More text
### Header 3
Even more text"""
        headings = parse_headings(text)
        assert len(headings) == 3
        assert headings[0].level == 1
        assert headings[1].level == 2
        assert headings[2].level == 3

    def test_heading_with_special_chars(self):
        text = "## Header with $pecial chars!"
        headings = parse_headings(text)
        assert len(headings) == 1
        assert headings[0].title == "Header with $pecial chars!"

    def test_heading_positions(self):
        text = "Line 1\n# Header\nLine 3"
        headings = parse_headings(text)
        assert len(headings) == 1
        assert headings[0].position == 2


class TestBuildHeadingTree:
    def test_empty_list(self):
        assert build_heading_tree([]) == []

    def test_single_heading(self):
        headings = [Heading(level=1, title="H1", position=1)]
        roots = build_heading_tree(headings)
        assert len(roots) == 1

    def test_nested_headings(self):
        headings = [
            Heading(level=1, title="H1", position=1),
            Heading(level=2, title="H2", position=2),
            Heading(level=1, title="H3", position=3),
        ]
        roots = build_heading_tree(headings)
        assert len(roots) == 2  # H1 and H3 are roots
        assert len(roots[0].children) == 1  # H2 is child of H1

    def test_deep_nesting(self):
        headings = [
            Heading(level=1, title="H1", position=1),
            Heading(level=2, title="H2", position=2),
            Heading(level=3, title="H3", position=3),
        ]
        roots = build_heading_tree(headings)
        assert len(roots) == 1
        assert len(roots[0].children) == 1
        assert len(roots[0].children[0].children) == 1


class TestGetHeadingPath:
    def test_simple_path(self):
        h = Heading(level=1, title="Section", position=1)
        assert get_heading_path(h) == "Section"

    def test_with_ancestors(self):
        h = Heading(level=2, title="Subsection", position=2)
        parent = Heading(level=1, title="Section", position=1)
        parent.add_child(h)
        # Now returns the full path
        assert get_heading_path(h) == "Section > Subsection"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
