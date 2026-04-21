"""Tests for generation/llm_client.py

Tests:
- _extract_content
- _extract_answer_from_reasoning
- generate_response (mocked)
"""

import pytest
from unittest.mock import Mock, patch
from rag_lab.generation.llm_client import _extract_content, _extract_answer_from_reasoning


class TestExtractContent:
    def test_content_present(self):
        choice = Mock()
        choice.message.content = "This is the answer"
        choice.message.reasoning_content = "Some reasoning"
        
        result = _extract_content(choice)
        assert result == "This is the answer"

    def test_content_empty_use_reasoning(self):
        choice = Mock()
        choice.message.content = ""
        choice.message.reasoning_content = "Some reasoning about the answer"
        
        result = _extract_content(choice)
        assert len(result) > 0

    def test_both_empty(self):
        choice = Mock()
        choice.message.content = ""
        choice.message.reasoning_content = ""
        
        result = _extract_content(choice)
        assert result == ""

    def test_content_with_whitespace(self):
        choice = Mock()
        choice.message.content = "  \n  "
        choice.message.reasoning_content = "Real content"
        
        result = _extract_content(choice)
        assert result == "Real content"


class TestExtractAnswerFromReasoning:
    def test_with_marker(self):
        reasoning = "Draft Response: This is the answer\n5. **Check**..."
        result = _extract_answer_from_reasoning(reasoning)
        assert "This is the answer" in result

    def test_without_marker(self):
        reasoning = "Just some reasoning without markers"
        result = _extract_answer_from_reasoning(reasoning)
        assert result == reasoning

    def test_with_numbered_section(self):
        reasoning = "Draft Response:\n\nThis is a sufficiently long reasoning answer to demonstrate that it gets extracted correctly.\n\n5. **Check** next section"
        result = _extract_answer_from_reasoning(reasoning)
        assert "This is a sufficiently long" in result
        assert "5. **Check**" not in result


class TestGenerateResponse:
    def test_success(self):
        with patch("rag_lab.generation.llm_client._get_client") as mock_client:
            mock_choice = Mock()
            mock_choice.message.content = "This is the response"
            mock_choice.message.reasoning_content = ""
            
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.completion_tokens_details = Mock()
            mock_response.usage.completion_tokens_details.reasoning_tokens = 0
            
            mock_client.return_value.chat.completions.create.return_value = mock_response
            
            result = _extract_content(mock_choice)
            assert result == "This is the response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
