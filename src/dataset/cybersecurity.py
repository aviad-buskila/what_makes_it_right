"""Cybersecurity multiple-choice dataset loaders.

Primary:  CyberMetric (Tihanyi et al., 2024) — 80/500/2000/10000 MCQs
          fetched as raw JSON from the official project repo.
Secondary: SecQA (Liu, 2023) — HuggingFace-hosted, textbook-grounded MCQs
          useful for isolating retrieval lift.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from src.dataset.loader import Question, VALID_ANSWERS, _validate_correct_answer

CYBERMETRIC_BASE_URL = (
    "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/"
)
CYBERMETRIC_FILES = {
    80: "CyberMetric-80-v1.json",
    500: "CyberMetric-500-v1.json",
    2000: "CyberMetric-2000-v1.json",
    10000: "CyberMetric-10000-v1.json",
}


def _download_json(url: str, cache_path: Path) -> dict:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        cache_path.write_bytes(data)
    return json.loads(cache_path.read_text(encoding="utf-8"))


def load_cybermetric(
    variant: int = 2000,
    max_questions: int | None = None,
    random_seed: int = 42,
    cache_dir: str | None = "data/cybermetric",
) -> list[Question]:
    """Load CyberMetric MCQs. ``variant`` ∈ {80, 500, 2000, 10000}."""
    if variant not in CYBERMETRIC_FILES:
        raise ValueError(
            f"Unknown CyberMetric variant {variant}. "
            f"Pick one of {sorted(CYBERMETRIC_FILES)}."
        )

    file_name = CYBERMETRIC_FILES[variant]
    url = CYBERMETRIC_BASE_URL + file_name
    cache_root = Path(cache_dir or "data/cybermetric")
    payload = _download_json(url, cache_root / file_name)

    raw_items = (
        payload["questions"] if isinstance(payload, dict) and "questions" in payload
        else payload
    )
    if not isinstance(raw_items, list):
        raise ValueError(
            "Unexpected CyberMetric payload: expected a list or "
            "{'questions': [...]}, got a different structure."
        )
    questions: list[Question] = []
    for idx, row in enumerate(raw_items):
        answers = row.get("answers") or {}
        # CyberMetric ships uppercase A/B/C/D; normalize case-insensitively so
        # mirrors or future variants don't silently drop every row.
        normalized = {str(letter).upper(): str(text).strip()
                      for letter, text in answers.items()}
        options = {letter: normalized[letter]
                   for letter in ("A", "B", "C", "D") if letter in normalized}
        if len(options) != 4:
            continue
        gold = str(row.get("solution", "")).strip().upper()
        if gold not in VALID_ANSWERS:
            continue
        q = Question(
            id=f"cybermetric_{variant}_{idx}",
            question_text=str(row.get("question", "")).strip(),
            options=options,
            correct_answer=gold,
            metadata={"source": "cybermetric", "variant": str(variant),
                      "index": str(idx)},
        )
        _validate_correct_answer(q.id, q.correct_answer)
        questions.append(q)

    if not questions:
        raise RuntimeError(
            f"CyberMetric loader parsed 0 questions from {file_name}. "
            "The upstream JSON structure may have changed — expected items "
            "with 'question', 'answers' (A/B/C/D), and 'solution' fields."
        )

    if max_questions is not None and max_questions < len(questions):
        rng = random.Random(random_seed)
        questions = rng.sample(questions, max_questions)
    return questions


def load_secqa(
    version: str = "v2",
    split: str = "test",
    max_questions: int | None = None,
    random_seed: int = 42,
    cache_dir: str | None = None,
) -> list[Question]:
    """Load SecQA (zefang-liu/secqa) from HuggingFace.

    Versions: "v1" (easier, 110 test) or "v2" (harder, 100 test).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "zefang-liu/secqa",
        name=version,
        split=split,
        cache_dir=cache_dir,
    )
    questions: list[Question] = []
    for idx, row in enumerate(ds):
        options = {
            "A": str(row["A"]).strip(),
            "B": str(row["B"]).strip(),
            "C": str(row["C"]).strip(),
            "D": str(row["D"]).strip(),
        }
        gold = str(row["Answer"]).strip().upper()
        q = Question(
            id=f"secqa_{version}_{split}_{idx}",
            question_text=str(row["Question"]).strip(),
            options=options,
            correct_answer=gold,
            metadata={"source": "secqa", "version": version, "split": split,
                      "index": str(idx)},
        )
        _validate_correct_answer(q.id, q.correct_answer)
        questions.append(q)

    if max_questions is not None and max_questions < len(questions):
        rng = random.Random(random_seed)
        questions = rng.sample(questions, max_questions)
    return questions
