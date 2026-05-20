"""Shared fixtures and configuration for all tests.

This module ensures that:
1. GPU is hidden during tests (CUDA_VISIBLE_DEVICES="")
2. Embedding and reranker caches are reset before each test
3. The embedding and reranker models run on CPU for tests to avoid OOM on GPU
4. Test assets directory is available

Usage:
    pytest tests/ -v
"""

import os
import pytest

# Hide GPU entirely during tests to prevent any GPU memory allocation
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "llm_required: mark test as requiring a live LLM server — skipped when unavailable",
    )


def pytest_collection_modifyitems(config, items):
    """Skip llm_required tests when the LLM server is not reachable."""
    import urllib.request
    llm_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    try:
        urllib.request.urlopen(f"{llm_url}/models", timeout=2)
        llm_available = True
    except Exception:
        llm_available = False

    if not llm_available:
        skip_marker = pytest.mark.skip(reason="LLM server not available (set LLM_BASE_URL or start server)")
        for item in items:
            if item.get_closest_marker("llm_required"):
                item.add_marker(skip_marker)

# Force CPU for all ML models in tests to avoid OOM on GPU
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"

# Explicitly patch config just in case it was already imported
try:
    import rag_lab.config as config
    config.EMBEDDING_DEVICE = "cpu"
except ImportError:
    pass


@pytest.fixture(autouse=True)
def reset_ml_caches():
    """Reset global model caches before each test.

    This ensures that each test starts with a clean slate — no cached
    models from previous tests that might be running on a different device.
    """
    # Reset embedding model cache
    from rag_lab.embedding.encoder import reset_embedding_cache
    reset_embedding_cache()

    # Reset reranker cache
    from rag_lab.retrieval.reranker import reset_reranker_cache
    reset_reranker_cache()

    yield


@pytest.fixture
def sample_text():
    """Sample text for testing chunking.

    Contains headings, paragraphs, and a table to test various chunking scenarios.

    Returns:
        str: Sample text with various markdown elements.
    """
    return """# Section 1: Introduction
This is the first section. It contains some text about the topic.

## Subsection 1.1: Details
More detailed content here.

## Subsection 1.2: More Details
Even more content.

# Section 2: Methods
Another section with different content.

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| A        | B        | C        |
| D        | E        | F        |

# Section 3: Results
Final section with results.
"""


@pytest.fixture
def sample_headings():
    """Sample headings for testing parser.

    Returns:
        list: List of heading dicts with title, level, and position.
    """
    return [
        {"title": "Section 1: Introduction", "level": 1, "position": 1},
        {"title": "Subsection 1.1: Details", "level": 2, "position": 5},
        {"title": "Subsection 1.2: More Details", "level": 2, "position": 9},
        {"title": "Section 2: Methods", "level": 1, "position": 13},
        {"title": "Section 3: Results", "level": 1, "position": 17},
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing retrieval and generation.

    Returns:
        list: List of chunk dicts with 'text' key and metadata.
    """
    return [
        {
            "chunk_id": "abc123",
            "doc_id": "sdmx_tech_notes_2.1",
            "text": "SDMX-ML is an XML-based format for data exchange.",
            "heading_path": "Section 1 > SDMX-ML",
            "tipo": "texto",
            "posicion_relativa": 0.1,
            "n_tokens": 10,
        },
        {
            "chunk_id": "def456",
            "doc_id": "sdmx_tech_notes_2.1",
            "text": "SDMX-EDI is a text-based format for statistical data.",
            "heading_path": "Section 1 > SDMX-EDI",
            "tipo": "texto",
            "posicion_relativa": 0.3,
            "n_tokens": 12,
        },
        {
            "chunk_id": "ghi789",
            "doc_id": "sdmx_tech_notes_2.1",
            "text": "The DSD defines the structure of statistical data.",
            "heading_path": "Section 2 > DSD",
            "tipo": "texto",
            "posicion_relativa": 0.5,
            "n_tokens": 11,
        },
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing.

    Returns:
        str: A sample question about SDMX standards.
    """
    return "What is SDMX-ML?"


@pytest.fixture
def test_assets_dir():
    """Path to test assets directory.

    Returns:
        pathlib.Path: Path to the test assets directory.
    """
    from pathlib import Path
    return Path(__file__).parent / "assets"
