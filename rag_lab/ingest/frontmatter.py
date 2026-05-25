"""YAML frontmatter parser for Markdown documents.

Extracts and normalises the canonical contract fields:
  doc_id, title, domain, source_type, language, version, tags

Also produces derived tags for FilterSpec use:
  domain:X, source_type:X, lang:X, version:X
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FrontmatterData:
    """Parsed frontmatter from a Markdown document."""

    doc_id: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None
    source_type: Optional[str] = None
    language: Optional[str] = None
    version: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def derived_tags(self) -> List[str]:
        """Normalised tags derived from structured fields for FilterSpec use."""
        out: list[str] = []
        if self.domain:
            out.append(f"domain:{self.domain.strip().lower()}")
        if self.source_type:
            out.append(f"source_type:{self.source_type.strip().lower()}")
        if self.language:
            out.append(f"lang:{self.language.strip().lower()}")
        if self.version:
            out.append(f"version:{str(self.version).strip()}")
        return out

    @property
    def all_tags(self) -> List[str]:
        """Union of explicit tags and derived tags (deduped, sorted)."""
        return sorted(set(self.tags) | set(self.derived_tags))


def parse_frontmatter(text: str) -> FrontmatterData:
    """Extract and normalise YAML frontmatter from a Markdown document.

    Returns an empty FrontmatterData if no frontmatter block is present or
    if PyYAML is not installed.  Never raises.

    Args:
        text: Full text of the Markdown file.

    Returns:
        FrontmatterData with parsed fields (may be all-None if absent).
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return FrontmatterData()

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return FrontmatterData()

    frontmatter_text = "\n".join(lines[1:end_idx])
    try:
        import yaml
        raw = yaml.safe_load(frontmatter_text) or {}
    except Exception:
        return FrontmatterData()

    if not isinstance(raw, dict):
        return FrontmatterData()

    doc_id = _str_or_none(raw.get("doc_id"))
    title = _str_or_none(raw.get("title"))
    domain = _str_or_none(raw.get("domain"))
    source_type = _str_or_none(raw.get("source_type"))
    language = _str_or_none(raw.get("language"))
    version = _str_or_none(raw.get("version"))
    tags = _normalise_tags(raw.get("tags"))

    return FrontmatterData(
        doc_id=doc_id,
        title=title,
        domain=domain,
        source_type=source_type,
        language=language,
        version=version,
        tags=tags,
        raw=raw,
    )


def extract_h1_title(text: str) -> Optional[str]:
    """Return the first H1 heading from a Markdown document, or None."""
    for line in text.splitlines():
        m = re.match(r'^#\s+(.*)', line)
        if m:
            return m.group(1).strip() or None
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _normalise_tags(value: Any) -> List[str]:
    """Convert frontmatter tags value to a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        tag = item.strip()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out
