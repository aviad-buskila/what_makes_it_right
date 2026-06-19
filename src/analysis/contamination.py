"""N-gram contamination / overlap probe.

Checks how much of a benchmark's items overlap (verbatim n-grams) with a text
corpus. Two uses:

1. RAG-leakage check: overlap between the evaluation questions and the retrieval
   corpus, to confirm the corpus is a knowledge proxy and not an answer key.
2. Pretraining-contamination proxy: overlap between the evaluation questions and
   an external reference text supplied by the user.

Tokenisation is whitespace/word-based and lowercase; this is a standard
surface-overlap probe and does not require any heavy dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.dataset.loader import Question

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def build_corpus_ngrams(corpus_texts: list[str], n: int = 13) -> set[tuple[str, ...]]:
    """Build the set of all n-grams present in the corpus."""
    grams: set[tuple[str, ...]] = set()
    for text in corpus_texts:
        grams |= _ngrams(_tokens(text), n)
    return grams


@dataclass
class ContaminationReport:
    n: int
    n_questions: int
    contaminated_questions: int
    contamination_rate: float
    mean_max_overlap: float  # mean over questions of (matched n-grams / question n-grams)


def question_overlap(
    questions: list[Question],
    corpus_ngrams: set[tuple[str, ...]],
    n: int = 13,
    include_options: bool = True,
    overlap_threshold: float = 0.0,
) -> ContaminationReport:
    """Compute n-gram overlap between questions and a pre-built corpus n-gram set.

    A question is counted as "contaminated" if the fraction of its n-grams that
    appear in the corpus strictly exceeds ``overlap_threshold``.
    """
    contaminated = 0
    overlap_fractions: list[float] = []
    for q in questions:
        text = q.question_text
        if include_options:
            text = text + " " + " ".join(q.options.values())
        q_grams = _ngrams(_tokens(text), n)
        if not q_grams:
            overlap_fractions.append(0.0)
            continue
        matched = len(q_grams & corpus_ngrams)
        frac = matched / len(q_grams)
        overlap_fractions.append(frac)
        if frac > overlap_threshold:
            contaminated += 1

    n_q = len(questions)
    return ContaminationReport(
        n=n,
        n_questions=n_q,
        contaminated_questions=contaminated,
        contamination_rate=(contaminated / n_q) if n_q else 0.0,
        mean_max_overlap=(sum(overlap_fractions) / len(overlap_fractions))
        if overlap_fractions else 0.0,
    )
