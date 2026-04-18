"""Download and cache an MCQ dataset (medical or cybersecurity)."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.loader import load_dataset_by_source, save_questions


def main():
    parser = argparse.ArgumentParser(description="Download an MCQ dataset")
    parser.add_argument(
        "--source",
        default="medqa",
        choices=["medqa", "cybermetric", "secqa"],
        help="Dataset source",
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument(
        "--variant",
        default=None,
        help="CyberMetric size (80/500/2000/10000) or SecQA version (v1/v2)",
    )
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (defaults to data/<source>/questions.jsonl)",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path(
        f"data/{args.source}/questions.jsonl"
    )

    print(f"Loading dataset source={args.source} split={args.split}...")
    questions = load_dataset_by_source(
        source=args.source,
        max_questions=args.max_questions,
        random_seed=args.seed,
        split=args.split,
        variant=args.variant,
    )
    print(f"Loaded {len(questions)} questions.")

    save_questions(questions, output_path)
    print(f"Saved to {output_path}")

    # Show a sample
    if questions:
        q = questions[0]
        print("\nSample question:")
        print(f"  Q: {q.question_text[:160]}...")
        for k, v in q.options.items():
            print(f"  {k}) {v[:80]}")
        print(f"  Correct: {q.correct_answer}")


if __name__ == "__main__":
    main()
