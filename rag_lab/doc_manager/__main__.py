"""Main entry point for the document manager.

Launches the interactive menu by default, or the CLI when arguments are provided.
"""

import sys

from rag_lab.doc_manager.cli import app
from rag_lab.doc_manager.interactive import interactive_mode


def main():
    """Route to interactive mode or CLI based on arguments."""
    if len(sys.argv) > 1 and sys.argv[1] != "--help":
        # CLI mode when arguments are provided
        app()
    else:
        # Interactive mode by default
        interactive_mode()


if __name__ == "__main__":
    main()
