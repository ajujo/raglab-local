"""Core data structures for Markdown validation results."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ValidationSeverity(Enum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    line_number: Optional[int] = None


@dataclass
class ValidationReport:
    path: Path
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARN]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    @property
    def is_valid(self) -> bool:
        return not self.has_errors

    def summary(self) -> str:
        n_e = len(self.errors)
        n_w = len(self.warnings)
        n_i = sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)
        parts = []
        if n_e:
            parts.append(f"{n_e} error{'s' if n_e != 1 else ''}")
        if n_w:
            parts.append(f"{n_w} warning{'s' if n_w != 1 else ''}")
        if n_i:
            parts.append(f"{n_i} info")
        return ", ".join(parts) if parts else "OK"


def count_tokens_approx(text: str) -> int:
    """Approximate token count (~4 chars per token)."""
    return max(1, len(text) // 4)
