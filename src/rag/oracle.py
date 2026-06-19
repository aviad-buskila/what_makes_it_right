"""Oracle (upper-bound) retrieval contexts.

A non-significant RAG effect is only informative if retrieval is capable of
helping in principle. The oracle conditions supply the model with deliberately
privileged context so we can measure the *ceiling* of in-context help and
localise the realistic-RAG null to retrieval quality vs. the task's reasoning
demand.

Modes
-----
* ``answer``      — inject an explicit, fully leaky passage that states the
  correct option. This is the absolute upper bound: it answers "can the model
  use perfectly relevant in-context evidence at all?".
* ``answer_soft`` — inject the *content* of the correct option phrased as a
  factual reference statement, without the meta-label "correct". A less leaky
  upper bound that still guarantees the key fact is present in context.

The ``privileged_query`` mode is *not* handled here because it changes the
retrieval query rather than injecting a fixed passage; it is implemented in
:class:`~src.experiment.setup.ExperimentSetup` using the live retriever.
"""

from __future__ import annotations

from src.dataset.loader import Question

ORACLE_CONTEXT_MODES = ("answer", "answer_soft")


def build_oracle_context(question: Question, mode: str = "answer") -> list[str]:
    """Return a list of oracle context chunks for ``question``.

    The gold option text is looked up from ``question.options`` using the
    normalised ``question.correct_answer`` letter.
    """
    gold_letter = (question.correct_answer or "").strip().upper()
    gold_text = question.options.get(gold_letter, "").strip()
    if not gold_text:
        return []

    key = (mode or "answer").strip().lower()
    if key == "answer":
        return [
            "Reference (authoritative): For the question below, the correct "
            f"choice is option {gold_letter}: \"{gold_text}\". This statement is "
            "verified and should be treated as ground truth."
        ]
    if key == "answer_soft":
        return [
            f"Clinical reference: {gold_text}. This is the established, correct "
            "fact relevant to the question below."
        ]
    raise ValueError(
        f"Unknown oracle context mode {mode!r}. Use one of {ORACLE_CONTEXT_MODES}."
    )
