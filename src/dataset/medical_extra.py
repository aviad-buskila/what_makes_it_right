"""Additional medical multiple-choice benchmarks for the generalization study.

All loaders normalise to the same four-option (or three-option, for PubMedQA)
:class:`~src.dataset.loader.Question` schema used by MedQA, so the prompt,
decoding, and scoring code path is identical across benchmarks.

Benchmarks
----------
* ``pubmedqa`` — PubMedQA (Jin et al., 2019). yes/no/maybe over a research
  abstract; mapped to options A=yes, B=no, C=maybe.
* ``medmcqa``  — MedMCQA (Pal et al., 2022) validation split (the test split
  ships without public labels).
* ``mmlu``     — MMLU (Hendrycks et al., 2021) clinical/medical subsets.
"""

from __future__ import annotations

import random

from src.dataset.loader import Question, VALID_ANSWERS, _validate_correct_answer

_LETTERS = ["A", "B", "C", "D"]

# Clinical/medical MMLU subjects used for the generalization study.
MMLU_MEDICAL_SUBJECTS = (
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
)


def _subsample(questions: list[Question], max_questions: int | None, seed: int) -> list[Question]:
    if max_questions is not None and max_questions < len(questions):
        rng = random.Random(seed)
        return rng.sample(questions, max_questions)
    return questions


def load_pubmedqa(
    split: str = "train",
    max_questions: int | None = None,
    random_seed: int = 42,
    cache_dir: str | None = None,
) -> list[Question]:
    """Load PubMedQA labeled subset (``pqa_labeled``, 1k items, single split).

    The research abstract is prepended to the question so the model has the
    context the benchmark assumes. final_decision in {yes, no, maybe} maps to
    options A/B/C.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "qiaojin/PubMedQA",
        name="pqa_labeled",
        split=split,
        cache_dir=cache_dir,
    )
    decision_to_letter = {"yes": "A", "no": "B", "maybe": "C"}
    options = {"A": "yes", "B": "no", "C": "maybe"}

    questions: list[Question] = []
    for idx, row in enumerate(ds):
        contexts = row.get("context", {})
        if isinstance(contexts, dict):
            context_text = " ".join(contexts.get("contexts", []) or [])
        else:
            context_text = ""
        question_text = str(row.get("question", "")).strip()
        if context_text:
            question_text = f"Context: {context_text}\n\nQuestion: {question_text}"

        gold = decision_to_letter.get(str(row.get("final_decision", "")).strip().lower())
        if gold is None:
            continue
        q = Question(
            id=f"pubmedqa_{split}_{idx}",
            question_text=question_text,
            options=dict(options),
            correct_answer=gold,
            metadata={"source": "pubmedqa", "split": split, "index": str(idx)},
        )
        _validate_correct_answer(q.id, q.correct_answer)
        questions.append(q)

    return _subsample(questions, max_questions, random_seed)


def load_medmcqa(
    split: str = "validation",
    max_questions: int | None = None,
    random_seed: int = 42,
    cache_dir: str | None = None,
) -> list[Question]:
    """Load MedMCQA (4-option). Defaults to the labeled validation split.

    Note: this is the *question/answer* MedMCQA split. The RAG corpus uses the
    MedMCQA *explanation* field of the train split, which is disjoint from the
    items evaluated here, so there is no answer leakage.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "openlifescienceai/medmcqa",
        split=split,
        cache_dir=cache_dir,
    )
    questions: list[Question] = []
    for idx, row in enumerate(ds):
        options = {
            "A": str(row.get("opa", "")).strip(),
            "B": str(row.get("opb", "")).strip(),
            "C": str(row.get("opc", "")).strip(),
            "D": str(row.get("opd", "")).strip(),
        }
        cop = row.get("cop")
        if cop is None or int(cop) not in (0, 1, 2, 3):
            continue  # unlabeled / hidden-label item
        gold = _LETTERS[int(cop)]
        if not all(options.values()):
            continue
        q = Question(
            id=f"medmcqa_{split}_{idx}",
            question_text=str(row.get("question", "")).strip(),
            options=options,
            correct_answer=gold,
            metadata={"source": "medmcqa", "split": split, "index": str(idx)},
        )
        _validate_correct_answer(q.id, q.correct_answer)
        questions.append(q)

    return _subsample(questions, max_questions, random_seed)


def load_mmlu_medical(
    subjects: str | tuple[str, ...] | None = None,
    split: str = "test",
    max_questions: int | None = None,
    random_seed: int = 42,
    cache_dir: str | None = None,
) -> list[Question]:
    """Load the clinical/medical subsets of MMLU (4-option).

    ``subjects`` may be a single subject name, a tuple of subjects, or ``None``
    to load all of :data:`MMLU_MEDICAL_SUBJECTS`.
    """
    from datasets import load_dataset

    if subjects is None:
        chosen = MMLU_MEDICAL_SUBJECTS
    elif isinstance(subjects, str):
        chosen = (subjects,)
    else:
        chosen = tuple(subjects)

    questions: list[Question] = []
    for subject in chosen:
        ds = load_dataset("cais/mmlu", subject, split=split, cache_dir=cache_dir)
        for idx, row in enumerate(ds):
            choices = row.get("choices") or []
            if len(choices) != 4:
                continue
            options = {letter: str(text).strip() for letter, text in zip(_LETTERS, choices)}
            answer = row.get("answer")
            if answer is None or int(answer) not in (0, 1, 2, 3):
                continue
            gold = _LETTERS[int(answer)]
            q = Question(
                id=f"mmlu_{subject}_{split}_{idx}",
                question_text=str(row.get("question", "")).strip(),
                options=options,
                correct_answer=gold,
                metadata={"source": "mmlu", "subject": subject, "split": split,
                          "index": str(idx)},
            )
            _validate_correct_answer(q.id, q.correct_answer)
            questions.append(q)

    if not questions:
        raise RuntimeError(
            f"MMLU loader parsed 0 questions for subjects={chosen!r}. "
            "Check the subject names against the cais/mmlu configs."
        )
    return _subsample(questions, max_questions, random_seed)
