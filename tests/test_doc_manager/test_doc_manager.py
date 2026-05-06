"""Tests for the document manager module."""

import json
import os
import sqlite3
import pytest
from pathlib import Path
from rag_lab.doc_manager.doc_store import DocManager


class TestDocManager:
    """Tests for DocManager class."""

    def test_ensure_db(self, tmp_path):
        """Test that database is created with correct schema."""
        manager = DocManager(db_path=tmp_path / "test.db")
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        tables = [t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "documents" in tables
        assert "tags" in tables
        conn.close()

    def test_add_document(self, tmp_path):
        """Test adding a new document."""
        # Create a test file
        test_file = tmp_path / "test_doc.md"
        test_file.write_text("# Test Document\nSome content here.")
        
        manager = DocManager(db_path=tmp_path / "test.db")
        result = manager.add_document(test_file, chunk_count=5)
        assert result == True

        # Verify it was added
        doc = manager.get_document("test_doc")
        assert doc is not None
        assert doc["doc_id"] == "test_doc"
        assert doc["chunk_count"] == 5

    def test_add_duplicate_document(self, tmp_path):
        """Test that duplicate documents (same hash) are rejected."""
        test_file = tmp_path / "test_doc.md"
        test_file.write_text("# Test Document\nSome content here.")
        
        manager = DocManager(db_path=tmp_path / "test.db")
        assert manager.add_document(test_file) == True
        assert manager.add_document(test_file) == False  # Duplicate

    def test_delete_document(self, tmp_path):
        """Test deleting a document."""
        test_file = tmp_path / "test_doc.md"
        test_file.write_text("# Test Document\nSome content here.")
        
        manager = DocManager(db_path=tmp_path / "test.db")
        manager.add_document(test_file)
        result = manager.delete_document("test_doc")
        assert result == True
        
        # Verify it was deleted
        doc = manager.get_document("test_doc")
        assert doc is None

    def test_delete_nonexistent_document(self, tmp_path):
        """Test deleting a document that doesn't exist."""
        manager = DocManager(db_path=tmp_path / "test.db")
        result = manager.delete_document("nonexistent")
        assert result == False

    def test_assign_and_remove_tag(self, tmp_path):
        """Test assigning and removing tags."""
        test_file = tmp_path / "test_doc.md"
        test_file.write_text("# Test Document\nSome content here.")
        
        manager = DocManager(db_path=tmp_path / "test.db")
        manager.add_document(test_file)
        
        # Assign tag
        assert manager.assign_tag("test_doc", "sdmx") == True
        assert manager.assign_tag("test_doc", "technical") == True
        
        # Verify tags
        tags = manager.get_tags("test_doc")
        assert len(tags) == 2
        assert "sdmx" in tags
        assert "technical" in tags
        
        # Remove tag
        assert manager.remove_tag("test_doc", "sdmx") == True
        tags = manager.get_tags("test_doc")
        assert len(tags) == 1
        assert "technical" in tags

    def test_list_documents(self, tmp_path):
        """Test listing all documents."""
        test_file1 = tmp_path / "doc1.md"
        test_file1.write_text("# Document 1\nContent 1.")
        test_file2 = tmp_path / "doc2.md"
        test_file2.write_text("# Document 2\nContent 2.")
        
        manager = DocManager(db_path=tmp_path / "test.db")
        manager.add_document(test_file1, chunk_count=10)
        manager.add_document(test_file2, chunk_count=20)
        manager.assign_tag("doc1", "sdmx")
        manager.assign_tag("doc2", "technical")
        
        docs = manager.list_documents()
        assert len(docs) == 2
        assert docs[0]["doc_id"] in ["doc1", "doc2"]
        assert docs[1]["doc_id"] in ["doc1", "doc2"]

    def test_search_documents(self, tmp_path):
        """Test searching for documents."""
        test_file1 = tmp_path / "doc1.md"
        test_file1.write_text("# Document 1\nContent 1.")
        test_file2 = tmp_path / "doc2.md"
        test_file2.write_text("# Document 2\nContent 2.")
        
        manager = DocManager(db_path=tmp_path / "test.db")
        manager.add_document(test_file1)
        manager.add_document(test_file2)
        manager.assign_tag("doc1", "sdmx")
        manager.assign_tag("doc2", "technical")
        
        # Search by doc_id
        results = manager.search_documents("doc1")
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"
        
        # Search by tag
        results = manager.search_documents("sdmx")
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"

    def test_list_all_tags(self, tmp_path):
        """Test listing all tags."""
        test_file = tmp_path / "test_doc.md"
        test_file.write_text("# Test Document\nSome content here.")
        
        manager = DocManager(db_path=tmp_path / "test.db")
        manager.add_document(test_file)
        manager.assign_tag("test_doc", "sdmx")
        manager.assign_tag("test_doc", "technical")
        manager.assign_tag("test_doc", "reference")
        
        tags = manager.list_all_tags()
        assert len(tags) == 3
        assert "reference" in tags
        assert "sdmx" in tags
        assert "technical" in tags

    def test_update_chunk_count(self, tmp_path):
        """Test updating chunk count for a document."""
        test_file = tmp_path / "test_doc.md"
        test_file.write_text("# Test Document\nSome content here.")
        
        manager = DocManager(db_path=tmp_path / "test.db")
        manager.add_document(test_file, chunk_count=5)
        manager.update_chunk_count("test_doc", 10)
        
        doc = manager.get_document("test_doc")
        assert doc["chunk_count"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
