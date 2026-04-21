"""Tests for ingest/cleaner.py

Tests:
- clean_document
"""

import pytest
import tempfile
from pathlib import Path

from rag_lab.ingest.cleaner import clean_document


class TestCleanDocument:
    def test_clean_document(self, tmp_path):
        # Create a test file with base64 image
        test_file = tmp_path / "test.md"
        test_file.write_text("# Header\n\nSome text\n\n![image](data:image/png;base64,ABC123)")
        
        result = clean_document(test_file)
        assert result.exists()
        assert "base64" not in result.read_text()

    def test_clean_document_no_image(self, tmp_path):
        # Create a test file without base64 images
        test_file = tmp_path / "test.md"
        test_file.write_text("# Header\n\nSome text without images")
        
        result = clean_document(test_file)
        assert result.exists()
        assert "Some text without images" in result.read_text()

    def test_clean_document_empty(self, tmp_path):
        # Create an empty test file
        test_file = tmp_path / "test.md"
        test_file.write_text("")
        
        result = clean_document(test_file)
        assert result.exists()

    def test_clean_document_large(self, tmp_path):
        # Create a test file with multiple base64 images
        test_file = tmp_path / "test.md"
        test_file.write_text(
            "# Header\n\n"
            "Text 1\n\n"
            "![image1](data:image/png;base64,ABC123)\n\n"
            "Text 2\n\n"
            "![image2](data:image/jpeg;base64,XYZ789)"
        )
        
        result = clean_document(test_file)
        assert result.exists()
        content = result.read_text()
        assert "Text 1" in content
        assert "Text 2" in content
        assert "base64" not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
