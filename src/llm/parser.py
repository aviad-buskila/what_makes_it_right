from __future__ import annotations

import ast
import re

VALID_ANSWERS = {"A", "B", "C", "D"}

# Matches explicit answer statements: "answer is X", "answer: X", "**Answer:** X",
# "the correct answer is X", "final answer is X", etc. Case-insensitive.
# Requires the word "answer" to avoid matching arbitrary letters in prose.
_EXPLICIT_ANSWER = re.compile(
    r"answer[\s\*:\-\.\)\(]*(?:is\s+)?[\s\*:\-\.\)\(]*([A-D])(?![A-Za-z])",
    re.IGNORECASE,
)

# A single letter alone on its own line.
_BARE_LETTER_LINE = re.compile(r"(?:^|\n)\s*([A-D])\s*(?:$|\n)")

# "X)" or "X." at the start of a line (only trusted for short responses).
_LETTER_PREFIX = re.compile(r"^\s*([A-D])\s*[\)\.:]", re.MULTILINE)

_SHORT_RESPONSE_CHARS = 40


def normalize_answer_letter(value: str | None) -> str | None:
    """Normalize answer letters to uppercase A-D, else None."""
    if value is None:
        return None
    letter = str(value).strip().upper()
    if letter in VALID_ANSWERS:
        return letter
    return None


def extract_answer(response: str) -> str | None:
    """Extract the answer letter (A–D) from an LLM response.

    Strategy, in order:
    1. Take the LAST explicit ``answer ... X`` phrase. Models often critique
       distractors first and state the conclusion at the end, so the last
       explicit-answer match is the most reliable signal.
    2. Accept a bare leading letter (e.g. ``"D\\n\\nExplanation..."``). Reject
       long responses that start with ``"X)"`` or ``"X."`` — those are
       distractor analysis, not the answer.
    3. A letter alone on its own line, last occurrence.
    4. ``"X)"`` or ``"X."`` at line start, but only for short responses.
    5. Last standalone ``A|B|C|D`` anywhere as a final fallback.
    """
    text = response.strip()
    if not text:
        return None

    # 0. Preferred: strict structured output {"answer":"A"} (or python dict-like).
    m = re.search(r'["\']answer["\']\s*:\s*["\']?\s*([A-Da-d])\s*["\']?', text)
    if m:
        return normalize_answer_letter(m.group(1))
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return normalize_answer_letter(parsed.get("answer"))
        except (ValueError, SyntaxError):
            pass

    # 1. Last explicit-answer phrase wins.
    best_letter: str | None = None
    best_pos = -1
    for m in _EXPLICIT_ANSWER.finditer(text):
        if m.start() > best_pos:
            best_pos = m.start()
            best_letter = normalize_answer_letter(m.group(1))
    if best_letter is not None:
        return best_letter

    # 2. Genuine terse leading letter.
    first = text[0]
    if first.upper() in VALID_ANSWERS and (len(text) == 1 or not text[1].isalpha()):
        # Distrust long "X) ..." / "X. ..." prefixes: those are distractor analysis.
        if text[1:2] not in {")", "."} or len(text) <= _SHORT_RESPONSE_CHARS:
            return first.upper()

    # 3. Letter alone on its own line (prefer last — conclusions come last).
    line_matches = list(_BARE_LETTER_LINE.finditer(text))
    if line_matches:
        return normalize_answer_letter(line_matches[-1].group(1))

    # 4. Short "X)" / "X." line prefix.
    if len(text) <= _SHORT_RESPONSE_CHARS:
        m = _LETTER_PREFIX.search(text)
        if m:
            return normalize_answer_letter(m.group(1))

    # 5. Last-resort: final isolated letter anywhere in the response.
    isolated = list(re.finditer(r"\b([A-D])\b", text))
    if isolated:
        return normalize_answer_letter(isolated[-1].group(1))

    return None
