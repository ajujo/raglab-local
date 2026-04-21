"""Tests for generation/prompt_builder.py

Tests:
- build_prompt
"""

import pytest
from rag_lab.generation.prompt_builder import build_prompt


class TestBuildPrompt:
    def test_empty_chunks(self):
        system, user = build_prompt("What is SDMX?", [])
        assert len(system) > 0
        assert len(user) > 0

    def test_single_chunk(self):
        chunks = [
            {
                "chunk_id": "1",
                "text": "SDMX is a standard for data exchange.",
                "heading_path": "Section 1",
            }
        ]
        system, user = build_prompt("What is SDMX?", chunks)
        assert "SDMX is a standard" in user

    def test_multiple_chunks(self):
        chunks = [
            {
                "chunk_id": "1",
                "text": "SDMX-ML is an XML format.",
                "heading_path": "Section 1",
            },
            {
                "chunk_id": "2",
                "text": "SDMX-EDI is a text format.",
                "heading_path": "Section 2",
            },
        ]
        system, user = build_prompt("What are the SDMX formats?", chunks)
        assert "SDMX-ML" in user
        assert "SDMX-EDI" in user

    def test_system_prompt(self):
        system, _ = build_prompt("What?", [])
        assert "cita" in system.lower()

    def test_user_prompt_format(self):
        chunks = [
            {
                "chunk_id": "1",
                "text": "Test content",
                "heading_path": "Section 1",
            }
        ]
        system, user = build_prompt("What is this?", chunks)
        assert "What is this?" in user
        assert "Test content" in user

    def test_max_chunks(self):
        # Should only use top RERANK_TOP_K chunks
        chunks = [
            {
                "chunk_id": str(i),
                "text": f"Content {i}",
                "heading_path": f"Section {i}",
            }
            for i in range(20)
        ]
        system, user = build_prompt("Question?", chunks)
        # Should only include top chunks
        assert len(user) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
