"""Full statistical analysis for one or more experiments.

Loads one or more experiment JSONL files, merges them (so setups from different
runs — e.g. different embedders, corpora, scales, or quantization levels — can be
compared head to head), and reports:

  * majority-vote accuracy with Wilson 95% CIs,
  * exact pairwise McNemar tests with Holm and BH correction and odds ratios,
  * TOST equivalence tests for each RAG toggle against a margin,
  * achieved power to detect a target odds ratio.

Example:
    python scripts/statistical_analysis.py --experiment medical_mcq_comparison_full_0.1_3r
    python scripts/statistical_analysis.py \\
        --experiment run_nomic --experiment run_medcpt --tost-margin 0.02
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.metrics import accuracy_by_setup, wilson_ci
from src.analysis.statistics import (
    all_pairwise_tests,
    mcnemar_power,
    observed_discordant_rate,
    rag_effect_tests,
    tost_from_setups,
)
from src.storage.results import ResultsStore


def _load_merged(experiments: list[str], results_dir: str) -> pd.DataFrame:
    store = ResultsStore(results_dir)
    frames = []
    for name in experiments:
        df = store.load(name)
        if df.empty:
            print(f"Warning: no results for experiment '{name}'.")
            continue
        # Disambiguate identical setup names coming from different runs.
        if len(experiments) > 1:
            df = df.copy()
            df["setup_name"] = df["setup_name"] + f"::{name}"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis across experiments")
    parser.add_argument("--experiment", action="append", required=True,
                        help="experiment name (repeatable to merge multiple runs)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tost-margin", type=float, default=0.02,
                        help="equivalence margin (absolute accuracy) for TOST")
    parser.add_argument("--target-or", type=float, default=1.5,
                        help="target odds ratio for the power calculation")
    args = parser.parse_args()

    df = _load_merged(args.experiment, args.results_dir)
    if df.empty:
        print("No results to analyze.")
        return

    print(f"Loaded {len(df)} rows across {df['setup_name'].nunique()} setups, "
          f"{df['question_id'].nunique()} questions.\n")

    # --- Accuracy + Wilson CI ---
    acc = accuracy_by_setup(df)
    acc["ci_lower"] = acc.apply(lambda r: wilson_ci(int(r["correct"]), int(r["total"]))[0], axis=1)
    acc["ci_upper"] = acc.apply(lambda r: wilson_ci(int(r["correct"]), int(r["total"]))[1], axis=1)
    print("## Accuracy (majority vote) with Wilson 95% CI")
    print(acc[["setup_name", "accuracy", "ci_lower", "ci_upper", "correct", "total"]]
          .to_string(index=False))

    # --- Exact pairwise McNemar with correction + odds ratios ---
    pairwise = all_pairwise_tests(df)
    print("\n## Pairwise exact McNemar (with Holm/BH correction and odds ratio)")
    cols = ["setup_a", "setup_b", "a_only_correct", "b_only_correct",
            "p_value", "p_holm", "p_bh", "odds_ratio"]
    print(pairwise[[c for c in cols if c in pairwise.columns]].to_string(index=False))

    # --- TOST equivalence + power for RAG toggles ---
    rag_tests = rag_effect_tests(df)
    if not rag_tests.empty:
        print(f"\n## RAG-effect equivalence (TOST, margin=+/-{args.tost_margin}) and power")
        rows = []
        for _, r in rag_tests.iterrows():
            tost = tost_from_setups(df, r["setup_a"], r["setup_b"],
                                    margin=args.tost_margin)
            disc = observed_discordant_rate(df, r["setup_a"], r["setup_b"])
            power = mcnemar_power(int(r["n"]), disc, target_odds_ratio=args.target_or)
            rows.append({
                "base": r["setup_a"],
                "+RAG": r["setup_b"],
                "delta_acc": tost["delta"],
                "ci_low": tost["ci_low"],
                "ci_high": tost["ci_high"],
                "p_tost": tost["p_tost"],
                "equivalent": tost["equivalent"],
                "power@OR{}".format(args.target_or): round(power, 3),
            })
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("\n(No base/+RAG setup pairs found for equivalence testing.)")


if __name__ == "__main__":
    main()
