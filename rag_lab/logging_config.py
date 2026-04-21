"""Centralized logging configuration for RAG-Lab."""

import logging
import sys

from rag_lab.config import LOG_FILE, LOG_LEVEL


def setup_logging(level: str = "INFO") -> None:
    """Configure centralized logging for the entire RAG system.

    Args:
        level: Log level string (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ]

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
    )


logger = logging.getLogger("rag_lab")