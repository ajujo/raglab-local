"""Document cleaner: removes base64 embedded images from Markdown files.

The source document contains ~28 pages with base64-encoded images that
inflate the file 30x without adding retrievable text content.
"""

import logging
import re
from pathlib import Path

from rag_lab.config import DATA_DIR
from rag_lab.exceptions import DocumentIngestionError

logger = logging.getLogger("rag_lab")

# Pattern to match base64 image data in Markdown files
# Matches: ![alt](data:image/...;base64,VERY_LONG_STRING)
_BASE64_PATTERN = re.compile(
    r'!\[.*?\]\(data:image/[^;]*;base64,[A-Za-z0-9+/=]+\)',
    re.DOTALL,
)


def clean_document(
    source_path: Path,
    output_dir: Path = None,
) -> Path:
    """Clean a Markdown document by removing base64 embedded images.

    Args:
        source_path: Path to the source Markdown file.
        output_dir: Directory to write cleaned file. Defaults to data/cleaned/.

    Returns:
        Path to the cleaned file.

    Raises:
        DocumentIngestionError: If source file doesn't exist or is invalid.
    """
    if not source_path.exists():
        raise DocumentIngestionError(f"Source file not found: {source_path}")

    if not source_path.suffix in (".md", ".mdown", ".markdown"):
        raise DocumentIngestionError(f"Unsupported file format: {source_path.suffix}")

    output_dir = output_dir or (DATA_DIR / "cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cleaning document: {source_path.name}")

    # Read source
    raw_text = source_path.read_text(encoding="utf-8")

    # Check for base64 images
    base64_matches = _BASE64_PATTERN.findall(raw_text)
    if base64_matches:
        logger.info(f"Found {len(base64_matches)} embedded images to remove")
    else:
        logger.info("No embedded images found")

    # Remove base64 images
    cleaned_text = _BASE64_PATTERN.sub("", raw_text)

    # Write cleaned file
    output_path = output_dir / source_path.name
    output_path.write_text(cleaned_text, encoding="utf-8")

    # Log size comparison
    original_size = source_path.stat().st_size
    cleaned_size = output_path.stat().st_size
    reduction = (1 - cleaned_size / original_size) * 100 if original_size > 0 else 0
    logger.info(
        f"Cleaned: {original_size:,} -> {cleaned_size:,} bytes ({reduction:.0f}% reduction)"
    )

    return output_path