"""Tests for src.llm.parser.extract_answer.

Run directly: ``python tests/test_parser.py`` (no pytest required).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.parser import extract_answer


CASES: list[tuple[str, str | None, str]] = [
    # (response, expected, description)
    ("A", "A", "single bare letter"),
    ("D\n\nThis is a best practice because of reasons.", "D", "bare letter + explanation"),
    ("A)", "A", "terse letter with paren"),
    ("A) short answer here.", "A", "short response with letter prefix"),
    ("The answer is B.", "B", "explicit answer phrase"),
    ("the correct answer is C", "C", "correct-answer phrase"),
    ("Final answer: D", "D", "final-answer colon"),
    ("**Answer:** A", "A", "markdown bold answer"),
    # The critical regression case: model critiques A first, states D at the end.
    (
        "A) This is not a best practice for creating strong passwords because "
        "common words are easily guessed. The correct answer is D) Including a "
        "mix of uppercase, lowercase, numbers, and symbols.",
        "D",
        "distractor analysis then correct answer",
    ),
    (
        "A) foo is wrong. B) bar is also wrong. C) baz too. The answer is D.",
        "D",
        "analyze all distractors then state answer",
    ),
    (
        "The answer is A. Actually wait, the correct answer is D.",
        "D",
        "self-correction: last answer wins",
    ),
    (
        "Identification refers to the process by which a user claims their identity.\n"
        "Answer: A",
        "A",
        "explanation then answer tag",
    ),
    ("", None, "empty response"),
    ("I have no idea.", None, "no letter anywhere"),
]


def main() -> int:
    failures = []
    for response, expected, desc in CASES:
        got = extract_answer(response)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {desc}: got={got!r} expected={expected!r}")
        if not ok:
            failures.append((desc, response, expected, got))

    print()
    print(f"{len(CASES) - len(failures)}/{len(CASES)} passed")
    if failures:
        print("\nFailures:")
        for desc, response, expected, got in failures:
            print(f"  - {desc}")
            print(f"    response:  {response!r}")
            print(f"    expected:  {expected!r}")
            print(f"    got:       {got!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
