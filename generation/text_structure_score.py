"""Text-structure scoring — regex-based structural signal on recent output.
"""

from __future__ import annotations

import re

_TEXT_STRUCTURE_PATTERNS: list[re.Pattern] = []  # lazily compiled


def _compile_text_structure_patterns() -> list[re.Pattern]:
    """Compile regex patterns for structural output detection (once)."""
    if not _TEXT_STRUCTURE_PATTERNS:
        _TEXT_STRUCTURE_PATTERNS.extend(
            [
                re.compile(r"[{}()\[\];]"),  # brackets, semicolons
                re.compile(r"[=+\-*/^].*\d"),  # math operators + numbers
                re.compile(  # code keywords
                    r"^\s*(def |class |import |from |if |for |while |return )",
                    re.MULTILINE,
                ),
                re.compile(r"^\s*[-*\d]+[.)]\s", re.MULTILINE),  # list items
                re.compile(r'"[^"]+"\s*:'),  # JSON-like keys
            ]
        )
    return _TEXT_STRUCTURE_PATTERNS


def compute_text_structure_score(text: str) -> float:
    """Fraction of structural patterns detected in *text* (0–1)."""
    patterns = _compile_text_structure_patterns()
    hits = sum(1 for p in patterns if p.search(text))
    return hits / len(patterns)
