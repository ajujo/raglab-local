"""Parse Markdown headings and build a hierarchical section tree.

Extracts all headings (H1-H6) with their depth, title, and position
in the document.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("rag_lab")


@dataclass
class Heading:
    """Represents a Markdown heading in the document."""
    level: int          # Heading level (1-6)
    title: str          # Heading text
    position: int       # Line number (1-based)
    children: List['Heading'] = None
    parent: Optional['Heading'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    @property
    def path(self) -> str:
        """Get the full heading path as a string."""
        return " > ".join(self._get_path_parts())

    def _get_path_parts(self) -> List[str]:
        """Get all ancestor titles including self."""
        parts = []
        current = self
        while current:
            parts.insert(0, current.title)
            current = current.parent
        return parts

    def add_child(self, heading: 'Heading') -> None:
        """Add a heading as a child of this heading."""
        self.children.append(heading)
        heading.parent = self


def parse_headings(text: str) -> List[Heading]:
    """Parse all Markdown headings from the text.

    Args:
        text: The full text of the Markdown document.

    Returns:
        List of Heading objects in order of appearance.
    """
    headings = []
    lines = text.split('\n')

    # Match lines that start with # (Markdown heading)
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

    for line_num, line in enumerate(lines, start=1):
        match = heading_pattern.match(line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append(Heading(
                level=level,
                title=title,
                position=line_num,
            ))

    logger.info(f"Parsed {len(headings)} headings")
    return headings


def build_heading_tree(headings: List[Heading]) -> List[Heading]:
    """Build a hierarchical tree from flat list of headings.

    Each heading becomes a child of the nearest heading with a lower level
    that appears before it.

    Args:
        headings: Flat list of Heading objects.

    Returns:
        List of root-level headings (level 1 or top-level).
    """
    if not headings:
        return []

    # Group headings by their parent
    roots = []
    stack = []  # Stack of (level, heading) for tracking hierarchy

    for heading in headings:
        # Pop from stack until we find a parent with lower level
        while stack and stack[-1][0] >= heading.level:
            stack.pop()

        if stack:
            # Add as child of current top of stack
            stack[-1][1].add_child(heading)
        else:
            # This is a root-level heading
            roots.append(heading)

        stack.append((heading.level, heading))

    return roots


def get_heading_path(heading: Heading) -> str:
    """Get the full path of a heading including all ancestors.

    Args:
        heading: The heading to get the path for.

    Returns:
        String like "3.2 > SDMX Information Model > ..."
    """
    return heading.path