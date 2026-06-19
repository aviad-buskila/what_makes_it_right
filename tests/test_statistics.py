"""Tests for src.analysis.statistics (exact McNemar, corrections, TOST, power).

Run directly: ``python tests/test_statistics.py`` (requires scipy, pandas).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis.statistics import (
    all_pairwise_tests,
    bh_correction,
    holm_correction,
    mcnemar_power,
    mcnemar_test,
    paired_accuracy_tost,
)


def _approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def _make_df(setup_correct: dict[str, list[bool]]) -> pd.DataFrame:
    """Build a single-repetition results frame from per-setup correctness lists."""
    rows = []
    n = len(next(iter(setup_correct.values())))
    for i in range(n):
        for setup, flags in setup_correct.items():
            # Encode correctness by making extracted==correct when flag is True.
            correct_letter = "A"
            chosen = "A" if flags[i] else "B"
            rows.append({
                "question_id": f"q{i}",
                "setup_name": setup,
                "extracted_answer": chosen,
                "correct_answer": correct_letter,
            })
    return pd.DataFrame(rows)


def main() -> int:
    failures: list[str] = []

    # 1. Exact McNemar from known discordant counts (the paper's headline pair).
    # Construct a df with a_only=144 (A right, B wrong) and b_only=231 (B right,
    # A wrong), plus some concordant pairs.
    a_flags, b_flags = [], []
    a_flags += [True] * 144 + [False] * 231 + [True] * 100 + [False] * 100
    b_flags += [False] * 144 + [True] * 231 + [True] * 100 + [False] * 100
    df = _make_df({"A": a_flags, "B": b_flags})
    res = mcnemar_test(df, "A", "B")
    if res["a_only_correct"] != 144 or res["b_only_correct"] != 231:
        failures.append(f"discordant counts wrong: {res}")
    # Exact p computed offline = 8.2e-6.
    if not _approx(res["p_exact"], 8.2e-6, tol=5e-6):
        failures.append(f"exact p mismatch: {res['p_exact']} (expected ~8.2e-6)")
    if not _approx(res["odds_ratio"], 231 / 144, tol=1e-2):
        failures.append(f"odds ratio mismatch: {res['odds_ratio']}")
    print(f"[{'PASS' if not failures else 'FAIL'}] exact McNemar: "
          f"a={res['a_only_correct']} b={res['b_only_correct']} "
          f"p={res['p_exact']:.2e} OR={res['odds_ratio']}")

    # 2. Holm and BH correction on the paper's six p-values.
    pvals = [0.5644, 0.1631, 0.00492, 8.2e-6, 3.08e-4, 2.30e-3]
    holm = holm_correction(pvals)
    bh = bh_correction(pvals)
    # Holm of the smallest (8.2e-6 * 6) and largest stays <= 1, monotone.
    if not _approx(holm[3], 8.2e-6 * 6, tol=1e-5):
        failures.append(f"holm smallest wrong: {holm[3]}")
    if any(holm[i] > 1.0 + 1e-9 for i in range(len(holm))):
        failures.append("holm exceeded 1.0")
    if not all(bh[i] <= 1.0 + 1e-9 for i in range(len(bh))):
        failures.append("bh exceeded 1.0")
    print(f"[{'PASS' if 'holm' not in ''.join(failures) else 'FAIL'}] "
          f"corrections: holm={[round(x,4) for x in holm]}")

    # 3. all_pairwise_tests adds correction columns.
    pw = all_pairwise_tests(df)
    for col in ("p_value", "p_holm", "p_bh", "odds_ratio"):
        if col not in pw.columns:
            failures.append(f"all_pairwise_tests missing column {col}")

    # 4. TOST equivalence. NB: with ~300 discordant pairs the paired-difference
    # SE is ~1.4pp, so a +/-2pp margin is NOT achievable (the study is
    # underpowered for that margin). A +/-5pp margin cleanly separates a tiny
    # effect (equivalent) from the +6.8pp fine-tuning effect (not equivalent).
    tost = paired_accuracy_tost(a_only=150, b_only=156, n=1273, margin=0.05)
    if not tost["equivalent"]:
        failures.append(f"TOST should find equivalence for tiny effect at 5pp: {tost}")
    tost2 = paired_accuracy_tost(a_only=144, b_only=231, n=1273, margin=0.05)
    if tost2["equivalent"]:
        failures.append(f"TOST should reject equivalence for +6.8pp effect at 5pp: {tost2}")
    print(f"[{'PASS' if 'TOST' not in ''.join(failures) else 'FAIL'}] "
          f"TOST small={tost['equivalent']} large={tost2['equivalent']}")

    # 5. Power increases with N.
    p_small = mcnemar_power(200, 0.3, target_odds_ratio=1.5)
    p_large = mcnemar_power(2000, 0.3, target_odds_ratio=1.5)
    if not (0.0 <= p_small <= p_large <= 1.0):
        failures.append(f"power not monotone in N: {p_small} -> {p_large}")
    print(f"[{'PASS' if 'power' not in ''.join(failures) else 'FAIL'}] "
          f"power N=200:{p_small:.3f} N=2000:{p_large:.3f}")

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All statistics tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
