"""Lightweight tests for new revision modules that have no heavy dependencies.

Covers oracle context construction, the n-gram contamination probe, the embedder
factory dispatch, and dataset-source dispatch errors.

Run directly: ``python tests/test_revision_modules.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.contamination import build_corpus_ngrams, question_overlap
from src.dataset.loader import Question
from src.llm.embeddings import HFEmbedder, MedCPTEmbedder, OllamaEmbedder, get_embedder
from src.rag.oracle import build_oracle_context


def main() -> int:
    failures: list[str] = []

    q = Question(
        id="medqa_test_0",
        question_text="A 55-year-old man presents with chest pain. Best next step?",
        options={"A": "Aspirin", "B": "Ibuprofen", "C": "Acetaminophen", "D": "Warfarin"},
        correct_answer="A",
    )

    # 1. Oracle context references the gold option text.
    for mode in ("answer", "answer_soft"):
        ctx = build_oracle_context(q, mode=mode)
        ok = len(ctx) == 1 and "Aspirin" in ctx[0]
        if not ok:
            failures.append(f"oracle mode {mode} did not include gold text: {ctx}")
    print(f"[{'PASS' if not failures else 'FAIL'}] oracle context construction")

    # 2. Contamination probe finds an overlap when the question text is in corpus.
    corpus = ["A 55-year-old man presents with chest pain. Best next step?"]
    grams = build_corpus_ngrams(corpus, n=8)
    rep = question_overlap([q], grams, n=8)
    if rep.contaminated_questions != 1:
        failures.append(f"expected 1 contaminated question, got {rep}")
    # A clean question should not overlap.
    clean = Question(id="medqa_test_1", question_text="What is the capital of France?",
                     options={"A": "Paris", "B": "Rome", "C": "Berlin", "D": "Madrid"},
                     correct_answer="A")
    rep2 = question_overlap([clean], grams, n=8)
    if rep2.contaminated_questions != 0:
        failures.append(f"clean question wrongly flagged: {rep2}")
    print(f"[{'PASS' if 'contaminat' not in ''.join(failures) else 'FAIL'}] "
          f"contamination probe (rate={rep.contamination_rate:.0%}/{rep2.contamination_rate:.0%})")

    # 3. Embedder factory dispatch (no model loading for hf/medcpt).
    if not isinstance(get_embedder("nomic-embed-text", "ollama"), OllamaEmbedder):
        failures.append("ollama backend did not return OllamaEmbedder")
    if not isinstance(get_embedder("BAAI/bge-large-en-v1.5", "hf"), HFEmbedder):
        failures.append("hf backend did not return HFEmbedder")
    if not isinstance(get_embedder("medcpt", "medcpt"), MedCPTEmbedder):
        failures.append("medcpt backend did not return MedCPTEmbedder")
    try:
        get_embedder("x", "bogus")
        failures.append("bogus backend should raise ValueError")
    except ValueError:
        pass
    print(f"[{'PASS' if 'backend' not in ''.join(failures) else 'FAIL'}] embedder factory")

    # 4. Dataset dispatch raises for unknown sources.
    from src.dataset.loader import load_dataset_by_source
    try:
        load_dataset_by_source("nope")
        failures.append("unknown dataset source should raise ValueError")
    except ValueError:
        pass
    print(f"[{'PASS' if 'dataset' not in ''.join(failures) else 'FAIL'}] dataset dispatch")

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All revision-module tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
