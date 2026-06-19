from __future__ import annotations

import itertools
import math

import pandas as pd
from scipy.stats import binomtest, norm

from src.llm.parser import normalize_answer_letter


# ---------------------------------------------------------------------------
# Majority-vote correctness helper
# ---------------------------------------------------------------------------

def _majority_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame of majority-vote correctness per (question, setup)."""
    df = df.copy()
    df["extracted_answer"] = df["extracted_answer"].map(normalize_answer_letter)
    df["correct_answer"] = df["correct_answer"].map(normalize_answer_letter)

    def _majority_correct(group):
        majority = group["extracted_answer"].mode()
        if majority.empty:
            return False
        return majority.iloc[0] == group["correct_answer"].iloc[0]

    return (
        df.groupby(["question_id", "setup_name"])
        .apply(_majority_correct, include_groups=False)
        .reset_index()
        .rename(columns={0: "correct"})
    )


# ---------------------------------------------------------------------------
# Paired McNemar test (exact) with effect size
# ---------------------------------------------------------------------------

def mcnemar_test(df: pd.DataFrame, setup_a: str, setup_b: str) -> dict:
    """McNemar's test comparing two setups on the same questions.

    Uses majority-vote correctness per question. Reports both the *exact*
    binomial two-sided p-value (the default, preferred for journal reporting)
    and the continuity-corrected chi-square statistic for backward
    compatibility, plus the discordant odds ratio as an effect size.
    """
    votes = _majority_correctness(df)

    a = votes[votes["setup_name"] == setup_a].set_index("question_id")["correct"]
    b = votes[votes["setup_name"] == setup_b].set_index("question_id")["correct"]

    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    # Discordant cells.
    # a_only: A correct, B wrong;  b_only: B correct, A wrong.
    a_only = int(((a == True) & (b == False)).sum())
    b_only = int(((a == False) & (b == True)).sum())
    n = int(len(common))

    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {
            "setup_a": setup_a,
            "setup_b": setup_b,
            "n": n,
            "a_only_correct": a_only,
            "b_only_correct": b_only,
            "statistic": 0.0,
            "p_value": 1.0,
            "p_exact": 1.0,
            "odds_ratio": float("nan"),
            "significant": False,
        }

    # Exact two-sided binomial test on the discordant pairs (H0: p = 0.5).
    p_exact = binomtest(min(a_only, b_only), n_discordant, 0.5,
                        alternative="two-sided").pvalue

    # Continuity-corrected chi-square (kept for backward compatibility).
    statistic = (abs(a_only - b_only) - 1) ** 2 / n_discordant

    # Discordant odds ratio: >1 favours setup_b, <1 favours setup_a.
    odds_ratio = (b_only / a_only) if a_only > 0 else float("inf")

    return {
        "setup_a": setup_a,
        "setup_b": setup_b,
        "n": n,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "statistic": round(statistic, 4),
        # p_value remains the headline value used across the codebase; it is now
        # the *exact* binomial p-value.
        "p_value": round(p_exact, 6),
        "p_exact": round(p_exact, 6),
        "odds_ratio": round(odds_ratio, 4) if math.isfinite(odds_ratio) else odds_ratio,
        "significant": p_exact < 0.05,
    }


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm step-down family-wise corrected p-values (monotone, capped at 1)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    corrected = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, adj)  # enforce monotonicity
        corrected[idx] = running
    return corrected


def bh_correction(pvalues: list[float]) -> list[float]:
    """Benjamini--Hochberg FDR corrected p-values (monotone, capped at 1)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    corrected = [0.0] * m
    prev = 1.0
    # Walk from largest p-value to smallest, enforcing monotonicity.
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        adj = min(1.0, pvalues[idx] * m / (rank + 1))
        prev = min(prev, adj)
        corrected[idx] = prev
    return corrected


def all_pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run McNemar's test for all pairs of setups, with Holm and BH correction."""
    setups = sorted(df["setup_name"].unique())
    results = [mcnemar_test(df, a, b) for a, b in itertools.combinations(setups, 2)]
    out = pd.DataFrame(results)
    if out.empty:
        return out
    pvals = out["p_value"].tolist()
    out["p_holm"] = [round(p, 6) for p in holm_correction(pvals)]
    out["p_bh"] = [round(p, 6) for p in bh_correction(pvals)]
    out["significant_holm"] = out["p_holm"] < 0.05
    return out


def rag_effect_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Test the effect of RAG for each comparable setup pair.

    Supports setup names with suffixes like ``@t0.5`` by matching:
      model@tX <-> model+RAG@tX
    """
    setups = sorted(df["setup_name"].unique())
    results = []
    for s in setups:
        if "+RAG" not in s and "+Oracle" not in s:
            if "@t" in s:
                base_name, temp_suffix = s.split("@t", 1)
                rag_name = f"{base_name}+RAG@t{temp_suffix}"
            else:
                rag_name = s + "+RAG"
            if rag_name in setups:
                results.append(mcnemar_test(df, s, rag_name))
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Equivalence testing (TOST) for a paired accuracy difference
# ---------------------------------------------------------------------------

def paired_accuracy_tost(
    a_only: int,
    b_only: int,
    n: int,
    margin: float = 0.02,
    alpha: float = 0.05,
) -> dict:
    """Two one-sided tests for equivalence of two paired accuracies.

    The effect is the marginal accuracy difference ``delta = (b_only - a_only)/n``
    (positive favours setup B). Equivalence within ``+/- margin`` is concluded
    when the ``(1 - 2*alpha)`` confidence interval lies entirely inside
    ``[-margin, +margin]`` (the standard TOST <-> CI equivalence).

    Returns the point estimate, the equivalence CI, the two one-sided p-values,
    and an ``equivalent`` flag.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    delta = (b_only - a_only) / n
    # SE of the difference of two correlated proportions (McNemar form).
    var = (a_only + b_only - (b_only - a_only) ** 2 / n) / (n ** 2)
    se = math.sqrt(var) if var > 0 else 0.0

    z = norm.ppf(1 - alpha)  # one-sided critical value
    ci_low = delta - z * se
    ci_high = delta + z * se

    if se == 0.0:
        p_lower = 0.0 if delta > -margin else 1.0
        p_upper = 0.0 if delta < margin else 1.0
    else:
        # H0_lower: delta <= -margin   ->  reject if delta sufficiently > -margin
        p_lower = 1 - norm.cdf((delta + margin) / se)
        # H0_upper: delta >=  margin   ->  reject if delta sufficiently < margin
        p_upper = norm.cdf((delta - margin) / se)

    p_tost = max(p_lower, p_upper)
    equivalent = (ci_low >= -margin) and (ci_high <= margin)

    return {
        "delta": round(delta, 6),
        "se": round(se, 6),
        "margin": margin,
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "p_lower": round(p_lower, 6),
        "p_upper": round(p_upper, 6),
        "p_tost": round(p_tost, 6),
        "equivalent": bool(equivalent),
    }


def tost_from_setups(
    df: pd.DataFrame,
    setup_a: str,
    setup_b: str,
    margin: float = 0.02,
    alpha: float = 0.05,
) -> dict:
    """Convenience: run TOST equivalence directly from two setup names."""
    res = mcnemar_test(df, setup_a, setup_b)
    tost = paired_accuracy_tost(
        a_only=res["a_only_correct"],
        b_only=res["b_only_correct"],
        n=res["n"],
        margin=margin,
        alpha=alpha,
    )
    return {"setup_a": setup_a, "setup_b": setup_b, **tost}


# ---------------------------------------------------------------------------
# Power analysis for the paired (McNemar) comparison
# ---------------------------------------------------------------------------

def mcnemar_power(
    n: int,
    discordant_rate: float,
    target_odds_ratio: float = 1.5,
    alpha: float = 0.05,
) -> float:
    """Approximate power of an exact McNemar test.

    Given ``n`` paired items and an expected ``discordant_rate`` (fraction of
    items where the two setups disagree), computes the power to reject
    ``H0: psi = 0.5`` on the discordant pairs when the true split corresponds to
    ``target_odds_ratio`` (psi = OR / (1 + OR)). Uses a normal approximation to
    the binomial sign test on the expected number of discordant pairs.
    """
    m = max(1.0, n * discordant_rate)  # expected discordant pairs
    psi = target_odds_ratio / (1.0 + target_odds_ratio)
    z_alpha = norm.ppf(1 - alpha / 2)

    se0 = math.sqrt(0.25 * m)            # sd of count under H0 (p=0.5)
    se1 = math.sqrt(psi * (1 - psi) * m) # sd under H1
    if se1 == 0:
        return 1.0
    mean_diff = abs(psi - 0.5) * m
    # Power = P(|count - m/2| > z_alpha * se0) under H1 (one dominant tail).
    z = (mean_diff - z_alpha * se0) / se1
    return float(max(0.0, min(1.0, norm.cdf(z))))


def observed_discordant_rate(df: pd.DataFrame, setup_a: str, setup_b: str) -> float:
    """Fraction of items on which the two setups disagree (majority vote)."""
    res = mcnemar_test(df, setup_a, setup_b)
    if res["n"] == 0:
        return 0.0
    return (res["a_only_correct"] + res["b_only_correct"]) / res["n"]
