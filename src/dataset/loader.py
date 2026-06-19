from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset

VALID_ANSWERS = {"A", "B", "C", "D"}


@dataclass
class Question:
    id: str
    question_text: str
    options: dict[str, str]  # {"A": "...", "B": "...", ...}
    correct_answer: str  # letter: "A", "B", "C", or "D"
    metadata: dict[str, str] = field(default_factory=dict)


def _normalize_answer_to_letter(answer: str, options: dict[str, str]) -> str:
    """Return answer as letter A-D when possible.

    Some MedQA variants store the gold answer as answer text, not as option letter.
    """
    candidate = (answer or "").strip()
    if candidate in options:
        return candidate

    # Match by exact option text first.
    for key, value in options.items():
        if candidate == value.strip():
            return key

    # Fallback: keep original value if no mapping is found.
    return candidate


def _validate_correct_answer(question_id: str, correct_answer: str) -> None:
    """Fail fast when a question's gold label is not A/B/C/D."""
    if correct_answer not in VALID_ANSWERS:
        raise ValueError(
            f"Invalid correct_answer for {question_id}: {correct_answer!r}. "
            "Expected one of A/B/C/D."
        )


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
            correct_answer=_normalize_answer_to_letter(row["answer"], options),
            metadata={"split": split, "index": str(idx)},
        )
        _validate_correct_answer(q.id, q.correct_answer)
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
            # Backward compatibility for previously saved files
            # where correct_answer may be answer text.
            data["correct_answer"] = _normalize_answer_to_letter(
                data.get("correct_answer", ""),
                data.get("options", {}),
            )
            q = Question(**data)
            _validate_correct_answer(q.id, q.correct_answer)
            questions.append(q)
    return questions


def load_dataset_by_source(
    source: str,
    max_questions: int | None = None,
    random_seed: int = 42,
    cache_dir: str | None = None,
    split: str = "test",
    variant: str | int | None = None,
) -> list[Question]:
    """Dispatch to a dataset loader by name.

    Supported sources:
      - ``medqa``       → MedQA-USMLE-4-options (default medical benchmark)
      - ``pubmedqa``    → PubMedQA labeled (yes/no/maybe → A/B/C)
      - ``medmcqa``     → MedMCQA (``variant`` selects split; default validation)
      - ``mmlu``        → MMLU clinical subsets (``variant`` = subject or None=all)
      - ``cybermetric`` → CyberMetric (``variant`` ∈ {80, 500, 2000, 10000})
      - ``secqa``       → SecQA (``variant`` ∈ {"v1", "v2"})
    """
    key = source.strip().lower()
    if key == "medqa":
        return load_medqa(
            split=split,
            max_questions=max_questions,
            random_seed=random_seed,
            cache_dir=cache_dir,
        )
    if key == "pubmedqa":
        from src.dataset.medical_extra import load_pubmedqa

        return load_pubmedqa(
            split=split if split in {"train"} else "train",
            max_questions=max_questions,
            random_seed=random_seed,
            cache_dir=cache_dir,
        )
    if key == "medmcqa":
        from src.dataset.medical_extra import load_medmcqa

        chosen_split = str(variant) if variant is not None else "validation"
        return load_medmcqa(
            split=chosen_split,
            max_questions=max_questions,
            random_seed=random_seed,
            cache_dir=cache_dir,
        )
    if key == "mmlu":
        from src.dataset.medical_extra import load_mmlu_medical

        subjects = None
        if variant is not None:
            subjects = tuple(str(variant).split(",")) if "," in str(variant) else str(variant)
        return load_mmlu_medical(
            subjects=subjects,
            split=split,
            max_questions=max_questions,
            random_seed=random_seed,
            cache_dir=cache_dir,
        )
    if key == "cybermetric":
        from src.dataset.cybersecurity import load_cybermetric

        chosen_variant = int(variant) if variant is not None else 2000
        return load_cybermetric(
            variant=chosen_variant,
            max_questions=max_questions,
            random_seed=random_seed,
            cache_dir=cache_dir or "data/cybermetric",
        )
    if key == "secqa":
        from src.dataset.cybersecurity import load_secqa

        chosen_version = str(variant) if variant is not None else "v2"
        return load_secqa(
            version=chosen_version,
            split=split,
            max_questions=max_questions,
            random_seed=random_seed,
            cache_dir=cache_dir,
        )
    raise ValueError(
        f"Unknown dataset source {source!r}. "
        "Use one of: medqa, pubmedqa, medmcqa, mmlu, cybermetric, secqa."
    )
