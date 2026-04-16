"""Download and cache the MedQA dataset."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.loader import load_medqa, save_questions


def main():
    parser = argparse.ArgumentParser(description="Download MedQA dataset")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--output", default="data/medqa/questions.jsonl", help="Output path")
    args = parser.parse_args()

    print(f"Loading MedQA ({args.split} split)...")
    questions = load_medqa(
        split=args.split,
        max_questions=args.max_questions,
        random_seed=args.seed,
    )
    print(f"Loaded {len(questions)} questions.")

    output_path = Path(args.output)
    save_questions(questions, output_path)
    print(f"Saved to {output_path}")

    # Show a sample
    if questions:
        q = questions[0]
        print(f"\nSample question:")
        print(f"  Q: {q.question_text[:120]}...")
        for k, v in q.options.items():
            print(f"  {k}) {v[:80]}")
        print(f"  Correct: {q.correct_answer}")


if __name__ == "__main__":
    main()
