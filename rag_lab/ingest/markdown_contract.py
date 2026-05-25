"""Markdown canonical contract: validation config and check functions."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rag_lab.ingest.validation import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    count_tokens_approx,
)

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class MarkdownValidationConfig:
    max_section_tokens: int = 1600
    max_estimated_chunks: int = 200
    max_table_rows: int = 200
    max_line_length: int = 500
    min_content_tokens: int = 50
    require_title: bool = True
    check_heading_hierarchy: bool = True


def validate_markdown(
    path: Path,
    config: Optional[MarkdownValidationConfig] = None,
) -> ValidationReport:
    """Validate a Markdown file against the canonical contract.

    Args:
        path: Path to the Markdown file.
        config: Thresholds (defaults to MarkdownValidationConfig()).

    Returns:
        ValidationReport with all issues found.
    """
    if config is None:
        config = MarkdownValidationConfig()

    report = ValidationReport(path=path)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="encoding_error",
            message=f"File is not valid UTF-8: {exc}",
        ))
        return report

    if not text.strip():
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="empty_file",
            message="File is empty or contains only whitespace.",
        ))
        return report

    lines = text.splitlines()

    total_tokens = count_tokens_approx(text)
    if total_tokens < config.min_content_tokens:
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARN,
            code="min_content",
            message=(
                f"Document has very little content (~{total_tokens} tokens, "
                f"threshold {config.min_content_tokens})."
            ),
        ))

    _check_frontmatter(lines, report)

    if config.require_title:
        _check_title(lines, report)

    if config.check_heading_hierarchy:
        _check_heading_hierarchy(lines, report)

    _check_section_lengths(lines, config, report)
    _check_table_sizes(lines, config, report)
    _check_long_lines(lines, config, report)
    _check_estimated_chunks(text, config, report)

    return report


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_frontmatter(lines: List[str], report: ValidationReport) -> None:
    """Check YAML frontmatter validity and canonical contract fields if present."""
    if not lines or lines[0].strip() != "---":
        # No frontmatter — doc_id cannot be extracted
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARN,
            code="frontmatter_missing",
            message=(
                "No YAML frontmatter found. "
                "Add a '--- ... ---' block with at least doc_id and title."
            ),
        ))
        return

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARN,
            code="frontmatter_unclosed",
            message="YAML frontmatter block opened with '---' but never closed.",
            line_number=1,
        ))
        return

    if not _YAML_AVAILABLE:
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            code="yaml_unavailable",
            message="PyYAML not installed — frontmatter content not validated.",
            line_number=1,
        ))
        return

    frontmatter_text = "\n".join(lines[1:end_idx])
    try:
        parsed = _yaml.safe_load(frontmatter_text)
    except Exception as exc:
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="frontmatter_invalid_yaml",
            message=f"YAML frontmatter is not valid: {exc}",
            line_number=1,
        ))
        return

    if not isinstance(parsed, dict):
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="frontmatter_not_mapping",
            message="YAML frontmatter must be a key-value mapping, not a list or scalar.",
            line_number=1,
        ))
        return

    _check_frontmatter_fields(parsed, report)


def _check_frontmatter_fields(parsed: dict, report: ValidationReport) -> None:
    """Validate canonical contract fields within a parsed frontmatter dict."""
    # Scope guard: dataset/dataset_id are prohibited
    for banned in ("dataset", "dataset_id"):
        if banned in parsed:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="frontmatter_scope_violation",
                message=(
                    f"Field '{banned}' is not part of the Markdown document contract. "
                    "RAG-Lab is a document-only system — no dataset/tabular fields."
                ),
                line_number=1,
            ))

    # doc_id: required
    doc_id = parsed.get("doc_id")
    if not doc_id or not str(doc_id).strip():
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="frontmatter_missing_doc_id",
            message=(
                "Frontmatter is missing required field 'doc_id'. "
                "Every document must have a unique stable identifier."
            ),
            line_number=1,
        ))

    # title: recommended (H1 check is separate, but warn here if absent)
    if not parsed.get("title") or not str(parsed.get("title", "")).strip():
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARN,
            code="frontmatter_missing_title",
            message=(
                "Frontmatter is missing 'title'. "
                "Will fall back to first H1 heading if present."
            ),
            line_number=1,
        ))

    # domain / source_type / language: recommended
    for field_name in ("domain", "source_type", "language"):
        if not parsed.get(field_name) or not str(parsed.get(field_name, "")).strip():
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARN,
                code=f"frontmatter_missing_{field_name}",
                message=(
                    f"Frontmatter is missing recommended field '{field_name}'. "
                    "This field improves document classification and filtering."
                ),
                line_number=1,
            ))

    # tags: must be a list of non-empty strings without duplicates
    tags_raw = parsed.get("tags")
    if tags_raw is not None:
        if not isinstance(tags_raw, list):
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="frontmatter_tags_not_list",
                message=(
                    f"'tags' must be a YAML list, got {type(tags_raw).__name__}."
                ),
                line_number=1,
            ))
        else:
            seen: set = set()
            for i, tag in enumerate(tags_raw):
                if not isinstance(tag, str):
                    report.issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="frontmatter_tag_not_string",
                        message=f"Tag at position {i} is not a string: {tag!r}.",
                        line_number=1,
                    ))
                    continue
                tag_stripped = tag.strip()
                if not tag_stripped:
                    report.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARN,
                        code="frontmatter_tag_empty",
                        message=f"Tag at position {i} is empty or whitespace-only.",
                        line_number=1,
                    ))
                elif tag_stripped != tag:
                    report.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARN,
                        code="frontmatter_tag_whitespace",
                        message=f"Tag {tag!r} has leading/trailing whitespace.",
                        line_number=1,
                    ))
                if tag_stripped in seen:
                    report.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARN,
                        code="frontmatter_tag_duplicate",
                        message=f"Tag {tag_stripped!r} appears more than once.",
                        line_number=1,
                    ))
                seen.add(tag_stripped)


def _check_title(lines: List[str], report: ValidationReport) -> None:
    """Warn if no H1 heading is present."""
    for line in lines:
        if re.match(r'^#\s+\S', line):
            return
    report.issues.append(ValidationIssue(
        severity=ValidationSeverity.WARN,
        code="missing_title",
        message="No H1 heading found. Consider adding a top-level title.",
    ))


def _check_heading_hierarchy(lines: List[str], report: ValidationReport) -> None:
    """Warn when heading levels skip (e.g., H1 → H3 without H2)."""
    prev_level = 0
    for i, line in enumerate(lines, start=1):
        m = re.match(r'^(#{1,6})\s+', line)
        if not m:
            continue
        level = len(m.group(1))
        if prev_level > 0 and level > prev_level + 1:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARN,
                code="heading_hierarchy_skip",
                message=(
                    f"Heading level jumps from H{prev_level} to H{level}: "
                    f"'{line.strip()[:60]}'"
                ),
                line_number=i,
            ))
        prev_level = level


def _check_section_lengths(
    lines: List[str],
    config: MarkdownValidationConfig,
    report: ValidationReport,
) -> None:
    """Warn if any section exceeds max_section_tokens."""
    heading_positions = [
        i for i, line in enumerate(lines)
        if re.match(r'^#{1,6}\s+', line)
    ]
    heading_positions.append(len(lines))

    for idx in range(len(heading_positions) - 1):
        start = heading_positions[idx]
        end = heading_positions[idx + 1]
        section_text = "\n".join(lines[start:end])
        tokens = count_tokens_approx(section_text)
        if tokens > config.max_section_tokens:
            heading_label = lines[start].strip()[:60]
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARN,
                code="section_too_long",
                message=(
                    f"Section is ~{tokens} tokens (max {config.max_section_tokens}): "
                    f"'{heading_label}'"
                ),
                line_number=start + 1,
            ))


def _check_table_sizes(
    lines: List[str],
    config: MarkdownValidationConfig,
    report: ValidationReport,
) -> None:
    """Warn if a table has too many data rows."""
    in_table = False
    table_start = 0
    row_count = 0

    def _flush(line_num: int) -> None:
        if in_table and row_count > config.max_table_rows:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARN,
                code="large_table",
                message=(
                    f"Table has {row_count} rows "
                    f"(max {config.max_table_rows})."
                ),
                line_number=table_start,
            ))

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and "|" in stripped

        if is_table_line and not in_table:
            in_table = True
            table_start = i
            row_count = 1
        elif is_table_line:
            # Separator rows (|---|---|) don't count as data rows
            if not re.match(r'^\|[\s\-:|]+\|', stripped):
                row_count += 1
        elif in_table:
            _flush(i)
            in_table = False
            row_count = 0

    _flush(len(lines))


def _check_long_lines(
    lines: List[str],
    config: MarkdownValidationConfig,
    report: ValidationReport,
) -> None:
    """Info notice for very long lines."""
    for i, line in enumerate(lines, start=1):
        if len(line) > config.max_line_length:
            report.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="long_line",
                message=(
                    f"Line {i} is {len(line)} chars "
                    f"(threshold {config.max_line_length})."
                ),
                line_number=i,
            ))


def _check_estimated_chunks(
    text: str,
    config: MarkdownValidationConfig,
    report: ValidationReport,
) -> None:
    """Warn if estimated chunk count exceeds the threshold."""
    from rag_lab.config import CHUNK_MAX_TOKENS

    total_tokens = count_tokens_approx(text)
    estimated = max(1, total_tokens // CHUNK_MAX_TOKENS)
    if estimated > config.max_estimated_chunks:
        report.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARN,
            code="estimated_chunks_high",
            message=(
                f"Document will produce ~{estimated} chunks "
                f"(threshold {config.max_estimated_chunks}). "
                "Consider splitting the document."
            ),
        ))
