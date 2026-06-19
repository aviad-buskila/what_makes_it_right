"""CLI: n-gram contamination / overlap probe for a benchmark.

Reports verbatim n-gram overlap between the evaluation questions and:
  * the RAG retrieval corpus (leakage check), and
  * an optional external reference text file (pretraining-contamination proxy).

Example:
    python scripts/contamination_probe.py --config config.yaml --n 13
    python scripts/contamination_probe.py --config config.yaml \\
        --reference path/to/reference_text.txt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.contamination import build_corpus_ngrams, question_overlap
from src.dataset.loader import load_dataset_by_source
from src.experiment.config import load_config
from src.rag.index import load_corpus_by_source


def main():
    parser = argparse.ArgumentParser(description="N-gram contamination probe")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n", type=int, default=13, help="n-gram length")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="subset of questions to probe (default: all)")
    parser.add_argument("--reference", default=None,
                        help="optional external reference text file to probe against")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Loading benchmark: {config.dataset.source} (variant={config.dataset.variant})")
    questions = load_dataset_by_source(
        source=config.dataset.source,
        max_questions=args.max_questions,
        random_seed=config.random_seed,
        cache_dir=config.dataset.cache_dir,
        split=config.dataset.split,
        variant=config.dataset.variant,
    )
    print(f"  {len(questions)} questions.")

    # 1. Overlap with the retrieval corpus (leakage check).
    print(f"\nLoading retrieval corpus: {config.rag.corpus_source}")
    corpus = load_corpus_by_source(
        source=config.rag.corpus_source,
        cache_dir=config.dataset.cache_dir,
        max_corpus_size=config.rag.max_corpus_size,
        include=tuple(config.rag.corpus_include) if config.rag.corpus_include else None,
    )
    corpus_texts = [row.get("text", "") for row in corpus]
    print(f"  {len(corpus_texts)} corpus documents. Building {args.n}-grams...")
    corpus_grams = build_corpus_ngrams(corpus_texts, n=args.n)
    print(f"  {len(corpus_grams)} unique {args.n}-grams in corpus.")

    rep = question_overlap(questions, corpus_grams, n=args.n)
    print("\n=== Question vs RAG-corpus overlap (leakage check) ===")
    print(f"  contaminated questions: {rep.contaminated_questions}/{rep.n_questions} "
          f"({rep.contamination_rate:.2%})")
    print(f"  mean per-question {args.n}-gram overlap fraction: {rep.mean_max_overlap:.4f}")

    # 2. Overlap with an external reference (pretraining proxy).
    if args.reference:
        ref_text = Path(args.reference).read_text(encoding="utf-8", errors="ignore")
        ref_grams = build_corpus_ngrams([ref_text], n=args.n)
        ref_rep = question_overlap(questions, ref_grams, n=args.n)
        print("\n=== Question vs external reference overlap (pretraining proxy) ===")
        print(f"  reference: {args.reference}")
        print(f"  contaminated questions: {ref_rep.contaminated_questions}/"
              f"{ref_rep.n_questions} ({ref_rep.contamination_rate:.2%})")
        print(f"  mean per-question {args.n}-gram overlap fraction: "
              f"{ref_rep.mean_max_overlap:.4f}")


if __name__ == "__main__":
    main()
