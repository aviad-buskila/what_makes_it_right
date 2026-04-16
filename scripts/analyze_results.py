"""Analyze experiment results and generate report."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.report import generate_report
from src.storage.results import ResultsStore


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--experiment", default="medical_mcq_comparison", help="Experiment name")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="results", help="Output directory for report")
    args = parser.parse_args()

    store = ResultsStore(args.results_dir)
    df = store.load(args.experiment)

    if df.empty:
        print(f"No results found for experiment '{args.experiment}'.")
        print(f"Looked in: {args.results_dir}/{args.experiment}.jsonl")
        return

    print(f"Loaded {len(df)} results.")
    print(f"  Setups: {df['setup_name'].nunique()}")
    print(f"  Questions: {df['question_id'].nunique()}")
    print(f"  Total calls: {len(df)}")

    report = generate_report(df, output_dir=args.output_dir)
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
