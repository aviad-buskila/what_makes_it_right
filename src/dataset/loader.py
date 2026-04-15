from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset


@dataclass
class Question:
    id: str
    question_text: str
    options: dict[str, str]  # {"A": "...", "B": "...", ...}
    correct_answer: str  # letter: "A", "B", "C", or "D"
    metadata: dict[str, str] = field(default_factory=dict)


def load_medqa(
    split: str = "test",
    max_questions: int | None = None,
    random_seed: int = 42,
    cache_dir: str | None = None,
) -> list[Question]:
    """Load MedQA USMLE 4-option questions from HuggingFace."""
    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split=split,
        cache_dir=cache_dir,
    )

    questions: list[Question] = []
    option_keys = ["A", "B", "C", "D"]

    for idx, row in enumerate(ds):
        options = {key: row["options"][key] for key in option_keys if key in row["options"]}
        q = Question(
            id=f"medqa_{split}_{idx}",
            question_text=row["question"],
            options=options,
            correct_answer=row["answer"],
            metadata={"split": split, "index": str(idx)},
        )
        questions.append(q)

    if max_questions is not None and max_questions < len(questions):
        rng = random.Random(random_seed)
        questions = rng.sample(questions, max_questions)

    return questions


def save_questions(questions: list[Question], path: Path) -> None:
    """Save questions to JSONL for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(q.__dict__) + "\n")


def load_questions(path: Path) -> list[Question]:
    """Load questions from JSONL."""
    questions = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            questions.append(Question(**data))
    return questions
