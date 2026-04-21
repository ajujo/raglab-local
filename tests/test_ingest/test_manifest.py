"""Tests for ingest/manifest.py

Tests:
- create_manifest
"""

import pytest
import json
from pathlib import Path

from rag_lab.ingest.manifest import create_manifest


class TestCreateManifest:
    def test_create_manifest(self, monkeypatch, tmp_path):
        import rag_lab.ingest.manifest as m
        monkeypatch.setattr(m, "DATA_DIR", tmp_path)
        
        source = tmp_path / "source.md"
        cleaned = tmp_path / "cleaned.md"
        source.write_text("Test content")
        cleaned.write_text("Cleaned content")
        
        manifest_path = create_manifest(source, cleaned)
        
        assert manifest_path.exists()
        with open(manifest_path) as f:
            data = json.load(f)
        assert "md5" in data or "hash" in data

    def test_force_recreate(self, monkeypatch, tmp_path):
        import rag_lab.ingest.manifest as m
        monkeypatch.setattr(m, "DATA_DIR", tmp_path)
        
        source = tmp_path / "source.md"
        cleaned = tmp_path / "cleaned.md"
        source.write_text("Test content")
        cleaned.write_text("Cleaned content")
        
        # First call
        create_manifest(source, cleaned)
        
        # Second call with force=True
        create_manifest(source, cleaned, force=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
