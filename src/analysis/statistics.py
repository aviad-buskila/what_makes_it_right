from __future__ import annotations

import itertools

import pandas as pd
from scipy.stats import chi2

from src.llm.parser import normalize_answer_letter


def mcnemar_test(df: pd.DataFrame, setup_a: str, setup_b: str) -> dict:
    """Perform McNemar's test comparing two setups on the same questions.

    Uses majority-vote correctness per question.
    """
    df = df.copy()
    df["extracted_answer"] = df["extracted_answer"].map(normalize_answer_letter)
    df["correct_answer"] = df["correct_answer"].map(normalize_answer_letter)

    # Get majority vote correctness per question for each setup
    def _majority_correct(group):
        majority = group["extracted_answer"].mode()
        if majority.empty:
            return False
        return majority.iloc[0] == group["correct_answer"].iloc[0]

    votes = (
        df.groupby(["question_id", "setup_name"])
        .apply(_majority_correct, include_groups=False)
        .reset_index()
        .rename(columns={0: "correct"})
    )

    a = votes[votes["setup_name"] == setup_a].set_index("question_id")["correct"]
    b = votes[votes["setup_name"] == setup_b].set_index("question_id")["correct"]

    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    # Contingency: b=discordant pairs
    # b_only_correct: B correct, A wrong
    # a_only_correct: A correct, B wrong
    a_only = ((a == True) & (b == False)).sum()
    b_only = ((a == False) & (b == True)).sum()

    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {
            "setup_a": setup_a,
            "setup_b": setup_b,
            "a_only_correct": int(a_only),
            "b_only_correct": int(b_only),
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    # McNemar's chi-squared (with continuity correction)
    statistic = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "setup_a": setup_a,
        "setup_b": setup_b,
        "a_only_correct": int(a_only),
        "b_only_correct": int(b_only),
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def all_pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run McNemar's test for all pairs of setups."""
    setups = sorted(df["setup_name"].unique())
    results = []
    for a, b in itertools.combinations(setups, 2):
        results.append(mcnemar_test(df, a, b))
    return pd.DataFrame(results)


def rag_effect_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Test the effect of RAG for each model (with vs without)."""
    setups = sorted(df["setup_name"].unique())
    results = []
    for s in setups:
        if "+RAG" not in s:
            rag_name = s + "+RAG"
            if rag_name in setups:
                results.append(mcnemar_test(df, s, rag_name))
    return pd.DataFrame(results)
