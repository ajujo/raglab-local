"""Semantic chunking: divide documents into coherent fragments.

Rules:
- Never cross an H2 heading or higher
- If section < max_tokens → single chunk
- If section >= max_tokens → sub-chunks with overlap
- Complete Markdown tables → always in a single chunk
- Each chunk carries metadata: doc_id, chunk_id, heading_path, tipo, posicion_relativa
- Chunks below CHUNK_MIN_TOKENS are merged or discarded
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import List

from rag_lab.chunking.parser import Heading, parse_headings, build_heading_tree
from rag_lab.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP, CHUNK_MIN_TOKENS
from rag_lab.exceptions import ChunkingError

logger = logging.getLogger("rag_lab")


@dataclass
class Chunk:
    """Represents a single chunk of text from the document."""
    chunk_id: str
    doc_id: str
    text: str
    heading_path: str
    tipo: str  # "texto", "tabla", "formula"
    posicion_relativa: float
    n_tokens: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "heading_path": self.heading_path,
            "tipo": self.tipo,
            "posicion_relativa": self.posicion_relativa,
            "n_tokens": self.n_tokens,
        }


def _count_tokens(text: str) -> int:
    """Approximate token count for the full text.

    Uses ~4 characters per token, applied to the entire text (not per-word).
    This is a reasonable approximation for English/technical text.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


def _is_table_line(line: str) -> bool:
    """Check if a line is part of a Markdown table."""
    stripped = line.strip()
    return '|' in stripped and stripped.startswith('|')


def _build_heading_path(heading: Heading, parent_map: dict) -> str:
    """Build the full hierarchical heading path.

    Args:
        heading: The heading to build the path for.
        parent_map: Dict mapping heading to its parent.

    Returns:
        Path string like "Section 3 > 3.2 Model > 3.2.1 Intro"
    """
    parts = []
    current = heading
    while current is not None:
        parts.append(current.title)
        current = parent_map.get(id(current))
    parts.reverse()
    return " > ".join(parts)


def _build_parent_map(roots: List[Heading]) -> dict:
    """Build a mapping from heading id to its parent heading.

    Args:
        roots: Top-level headings.

    Returns:
        Dict mapping id(heading) -> parent Heading (or None for roots).
    """
    parent_map = {}

    def _walk(heading: Heading, parent=None):
        parent_map[id(heading)] = parent
        for child in heading.children:
            _walk(child, heading)

    for root in roots:
        _walk(root, None)

    return parent_map


def _collect_all_headings(roots: List[Heading]) -> List[Heading]:
    """Flatten the heading tree into a list ordered by position.

    Args:
        roots: Top-level headings.

    Returns:
        All headings in document order.
    """
    result = []

    def _walk(heading: Heading):
        result.append(heading)
        for child in heading.children:
            _walk(child)

    for root in roots:
        _walk(root)

    result.sort(key=lambda h: h.position)
    return result


def chunk_document(
    text: str,
    doc_id: str = "sdmx_tech_notes_2.1",
    max_tokens: int = None,
    overlap: int = None,
) -> List[Chunk]:
    """Divide a document into semantic chunks.

    Processes every heading at all levels (not just roots), extracting
    content between each heading and the next. Applies overlap between
    sub-chunks and filters out tiny fragments.

    Args:
        text: Cleaned text of the document.
        doc_id: Document identifier.
        max_tokens: Maximum tokens per chunk (default from config).
        overlap: Overlap tokens between chunks (default from config).

    Returns:
        List of Chunk objects.

    Raises:
        ChunkingError: If text is empty or invalid.
    """
    if not text or not text.strip():
        raise ChunkingError("The document text is empty or invalid")

    max_tokens = max_tokens or CHUNK_MAX_TOKENS
    overlap = overlap or CHUNK_OVERLAP

    # Parse headings and build tree
    headings = parse_headings(text)
    roots = build_heading_tree(headings)

    if not headings:
        logger.warning("No headings found, treating document as single section")
        return _create_chunks(text, doc_id, "Document", 0.0, max_tokens, overlap)

    # Flatten all headings in document order
    all_headings = _collect_all_headings(roots)
    parent_map = _build_parent_map(roots)

    lines = text.split('\n')
    total_lines = len(lines)

    # Step 1: Collect all sections with their text and metadata
    sections = []
    for idx, heading in enumerate(all_headings):
        start_line = heading.position - 1  # Convert to 0-based

        if idx + 1 < len(all_headings):
            end_line = all_headings[idx + 1].position - 1
        else:
            end_line = total_lines

        section_text = '\n'.join(lines[start_line:end_line]).strip()
        if not section_text:
            continue

        heading_path = _build_heading_path(heading, parent_map)

        # Skip Table of Contents sections
        title_lower = heading.title.strip().lower()
        if title_lower in ('contents', 'table of contents', 'índice', 'toc'):
            logger.debug(f"Skipping TOC section: {heading_path}")
            continue

        # Collapse excessive whitespace
        section_text = re.sub(r'\n{3,}', '\n\n', section_text)

        relative_pos = start_line / total_lines

        sections.append({
            'heading': heading,
            'heading_path': heading_path,
            'text': section_text,
            'tokens': _count_tokens(section_text),
            'relative_pos': relative_pos,
            'level': heading.level,
            'parent_id': id(parent_map.get(id(heading))),
        })

    # Step 2: Merge small sibling sections into combined chunks.
    # When consecutive headings at the same level under the same parent
    # each produce small sections, they likely cover related sub-topics
    # (e.g., Reporting Year, Reporting Semester, ..., Reporting Day).
    # Merging them ensures a single query can retrieve the full picture.
    merged_sections = _merge_sibling_sections(sections, max_tokens)

    # Step 3: Create chunks from merged sections
    all_chunks = []
    for section in merged_sections:
        chunks = _create_chunks(
            section['text'], doc_id, section['heading_path'],
            section['relative_pos'], max_tokens, overlap,
        )
        all_chunks.extend(chunks)

    # Filter out tiny chunks (merge or discard)
    filtered = _filter_tiny_chunks(all_chunks)

    logger.info(
        f"Created {len(filtered)} chunks from document "
        f"({len(all_chunks)} before filtering, {len(all_chunks) - len(filtered)} removed)"
    )
    return filtered


def _merge_sibling_sections(
    sections: List[dict],
    max_tokens: int,
) -> List[dict]:
    """Merge small consecutive sibling sections into combined sections.

    When consecutive headings at the same level under the same parent
    each have small text, they are merged so a single chunk contains
    the full context (e.g., all Reporting Periods in one chunk).

    Merging stops when adding the next sibling would exceed max_tokens.

    Args:
        sections: List of section dicts with text, heading, level, parent_id.
        max_tokens: Maximum tokens for a merged section.

    Returns:
        List of sections (some merged, some unchanged).
    """
    if not sections:
        return []

    result = []
    i = 0

    while i < len(sections):
        current = sections[i]

        # Try to merge with consecutive siblings at the same level/parent.
        # We merge aggressively: even large siblings are grouped together,
        # because _create_chunks will split the result with overlap.
        # Cap at 2x max_tokens to avoid mega-chunks of unrelated siblings.
        merge_cap = max_tokens * 2
        group = [current]
        group_tokens = current['tokens']
        j = i + 1

        while j < len(sections):
            candidate = sections[j]
            # Must be same level and same parent to be siblings
            if (candidate['level'] == current['level']
                    and candidate['parent_id'] == current['parent_id']
                    and group_tokens + candidate['tokens'] <= merge_cap):
                group.append(candidate)
                group_tokens += candidate['tokens']
                j += 1
            else:
                break

        if len(group) > 1:
            # Merge the group
            merged_text = '\n\n'.join(s['text'] for s in group)
            merged_path = group[0]['heading_path']
            # Show range in heading path if multiple siblings merged
            if group[-1]['heading_path'] != group[0]['heading_path']:
                first_title = group[0]['heading'].title.strip()
                last_title = group[-1]['heading'].title.strip()
                parent_path_parts = merged_path.rsplit(' > ', 1)
                if len(parent_path_parts) > 1:
                    merged_path = f"{parent_path_parts[0]} > {first_title} … {last_title}"
                else:
                    merged_path = f"{first_title} … {last_title}"

            result.append({
                'heading': group[0]['heading'],
                'heading_path': merged_path,
                'text': merged_text,
                'tokens': _count_tokens(merged_text),
                'relative_pos': group[0]['relative_pos'],
                'level': current['level'],
                'parent_id': current['parent_id'],
            })
            logger.debug(
                f"Merged {len(group)} sibling sections: {merged_path} "
                f"({group_tokens} tokens)"
            )
            i = j
        else:
            result.append(current)
            i += 1

    return result


def _filter_tiny_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Remove or merge chunks that are too small.

    Chunks below CHUNK_MIN_TOKENS are merged with the previous chunk
    if possible, otherwise discarded.

    Args:
        chunks: List of chunks to filter.

    Returns:
        Filtered list.
    """
    if not chunks:
        return []

    filtered = []
    for chunk in chunks:
        if chunk.n_tokens >= CHUNK_MIN_TOKENS:
            filtered.append(chunk)
        elif filtered:
            # Merge with previous chunk
            prev = filtered[-1]
            merged_text = prev.text + "\n\n" + chunk.text
            prev.text = merged_text
            prev.n_tokens = _count_tokens(merged_text)
            # Update chunk_id for the merged chunk
            prev.chunk_id = hashlib.md5(
                merged_text[:100].encode()
            ).hexdigest()[:12]
        else:
            # First chunk and it's tiny — keep it anyway (better than losing content)
            filtered.append(chunk)

    return filtered


def _create_chunks(
    text: str,
    doc_id: str,
    heading_path: str,
    posicion_relativa: float,
    max_tokens: int,
    overlap: int,
) -> List[Chunk]:
    """Create chunks from text, respecting token limits with real overlap.

    Args:
        text: The text to chunk.
        doc_id: Document identifier.
        heading_path: Heading path for the chunk.
        posicion_relativa: Relative position in document.
        max_tokens: Max tokens per chunk.
        overlap: Overlap between chunks in tokens.

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    # Check for tables — if the section contains table lines, keep table content whole
    text_lines = text.split('\n')
    non_empty_lines = [l for l in text_lines if l.strip()]
    has_table = any(_is_table_line(l) for l in non_empty_lines)

    if has_table:
        # This section contains a table — keep as single chunk
        chunk_id = hashlib.md5(text[:100].encode()).hexdigest()[:12]
        return [Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text.strip(),
            heading_path=heading_path,
            tipo="tabla",
            posicion_relativa=posicion_relativa,
            n_tokens=_count_tokens(text),
        )]

    # If the whole section fits in one chunk, return it directly
    total_tokens = _count_tokens(text)
    if total_tokens <= max_tokens:
        chunk_id = hashlib.md5(text[:100].encode()).hexdigest()[:12]
        return [Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text.strip(),
            heading_path=heading_path,
            tipo="texto",
            posicion_relativa=posicion_relativa,
            n_tokens=total_tokens,
        )]

    # Split into overlapping chunks
    # Work with sentences/paragraphs for cleaner boundaries
    segments = _split_into_segments(text)

    chunks = []
    current_segments = []
    current_tokens = 0

    i = 0
    while i < len(segments):
        seg = segments[i]
        seg_tokens = _count_tokens(seg)

        if current_tokens + seg_tokens > max_tokens and current_segments:
            # Emit current chunk
            chunk_text = '\n'.join(current_segments).strip()
            if chunk_text:
                chunk_id = hashlib.md5(chunk_text[:100].encode()).hexdigest()[:12]
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    heading_path=heading_path,
                    tipo="texto",
                    posicion_relativa=posicion_relativa,
                    n_tokens=_count_tokens(chunk_text),
                ))

            # Backtrack for overlap: keep the last N tokens worth of segments
            overlap_segments = []
            overlap_tokens = 0
            for seg_back in reversed(current_segments):
                seg_back_tokens = _count_tokens(seg_back)
                if overlap_tokens + seg_back_tokens > overlap:
                    break
                overlap_segments.insert(0, seg_back)
                overlap_tokens += seg_back_tokens

            if len(overlap_segments) == len(current_segments):
                # Ensure we make progress if overlap covers all current segments
                overlap_segments = overlap_segments[1:]
                overlap_tokens = sum(_count_tokens(s) for s in overlap_segments)

            current_segments = overlap_segments
            current_tokens = overlap_tokens
            # Don't increment i — reprocess the current segment
            continue

        current_segments.append(seg)
        current_tokens += seg_tokens
        i += 1

    # Emit the last chunk
    if current_segments:
        chunk_text = '\n'.join(current_segments).strip()
        if chunk_text:
            chunk_id = hashlib.md5(chunk_text[:100].encode()).hexdigest()[:12]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                heading_path=heading_path,
                tipo="texto",
                posicion_relativa=posicion_relativa,
                n_tokens=_count_tokens(chunk_text),
            ))

    return chunks


def _split_into_segments(text: str) -> List[str]:
    """Split text into natural segments (paragraphs and sentences).

    Prefers paragraph boundaries (double newline), then falls back
    to single lines. This produces cleaner chunk boundaries than
    splitting by individual words.

    Args:
        text: Input text.

    Returns:
        List of text segments.
    """
    # Split by paragraphs first (double newline)
    paragraphs = re.split(r'\n\s*\n', text)

    segments = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a paragraph is very long, split by lines
        if _count_tokens(para) > 200:
            for line in para.split('\n'):
                line = line.strip()
                if line:
                    segments.append(line)
        else:
            segments.append(para)

    return segments