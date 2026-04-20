"""Re-parse an existing results JSONL with the current answer extractor and
regenerate the report.

Use this after changing ``src.llm.parser.extract_answer`` to avoid re-running
thousands of LLM calls. Reads ``results/<experiment>.jsonl``, re-parses every
``model_response`` with the current parser, recomputes ``is_correct``, writes
``results/<experiment>_rescored.jsonl``, and regenerates the markdown report
against the rescored data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.report import generate_report
from src.llm.parser import extract_answer


def rescore_file(src_path: Path, dst_path: Path) -> dict:
    """Re-parse responses in ``src_path`` and write rescored records to ``dst_path``.

    Returns a stats dict with change counts per setup.
    """
    changes: dict[str, dict[str, int]] = {}
    total = 0
    with src_path.open() as src, dst_path.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            setup = rec.get("setup_name", "?")
            stats = changes.setdefault(
                setup,
                {"n": 0, "answer_changed": 0, "correct_before": 0, "correct_after": 0},
            )
            stats["n"] += 1

            old_ans = rec.get("extracted_answer")
            old_correct = bool(rec.get("is_correct", False))
            stats["correct_before"] += int(old_correct)

            response = rec.get("model_response") or ""
            new_ans = extract_answer(response)
            correct_letter = rec.get("correct_answer")
            new_correct = new_ans is not None and new_ans == correct_letter

            if new_ans != old_ans:
                stats["answer_changed"] += 1

            rec["extracted_answer"] = new_ans
            rec["is_correct"] = new_correct
            stats["correct_after"] += int(new_correct)

            dst.write(json.dumps(rec) + "\n")

    return {"total": total, "by_setup": changes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiment",
        required=True,
        help="Experiment name (without .jsonl). Source file: results/<experiment>.jsonl",
    )
    ap.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing the source jsonl and where outputs are written.",
    )
    ap.add_argument(
        "--suffix",
        default="_rescored",
        help="Suffix appended to the rescored jsonl filename.",
    )
    ap.add_argument(
        "--report-dir",
        default=None,
        help="Directory for the regenerated report (default: same as results-dir).",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    src_path = results_dir / f"{args.experiment}.jsonl"
    dst_path = results_dir / f"{args.experiment}{args.suffix}.jsonl"

    if not src_path.exists():
        print(f"ERROR: source file not found: {src_path}", file=sys.stderr)
        return 1

    print(f"Rescoring {src_path} -> {dst_path}")
    stats = rescore_file(src_path, dst_path)

    print(f"\nRescored {stats['total']} records. Changes by setup:")
    print(f"{'setup':<28}{'n':>6}{'changed':>10}{'acc_before':>13}{'acc_after':>13}{'delta':>10}")
    for setup, s in sorted(stats["by_setup"].items()):
        n = s["n"]
        before = s["correct_before"] / n if n else 0.0
        after = s["correct_after"] / n if n else 0.0
        print(
            f"{setup:<28}{n:>6}{s['answer_changed']:>10}"
            f"{before:>13.4f}{after:>13.4f}{after - before:>+10.4f}"
        )

    report_dir = Path(args.report_dir) if args.report_dir else results_dir
    df = pd.read_json(dst_path, lines=True)
    df = df.drop_duplicates(subset=["question_id", "setup_name", "repetition"], keep="last")
    print(f"\nRegenerating report in {report_dir}/report.md ...")
    generate_report(df, output_dir=str(report_dir))
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
