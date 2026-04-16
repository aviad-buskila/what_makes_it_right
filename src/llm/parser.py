from __future__ import annotations

import re

VALID_ANSWERS = {"A", "B", "C", "D"}

# Patterns ordered from most to least specific
_PATTERNS = [
    re.compile(r"(?:the\s+)?answer\s+is\s*:?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"^([A-D])\s*[\)\.]", re.MULTILINE),
    re.compile(r"^([A-D])\s*$", re.MULTILINE),
    re.compile(r"\b([A-D])\b"),
]


def extract_answer(response: str) -> str | None:
    """Extract the answer letter from an LLM response.

    Returns the letter (A-D) or None if parsing fails.
    """
    text = response.strip()

    # Quick check: response starts with a single letter
    if text and text[0] in VALID_ANSWERS and (len(text) == 1 or not text[1].isalpha()):
        return text[0]

    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            letter = match.group(1).upper()
            if letter in VALID_ANSWERS:
                return letter

    return None
