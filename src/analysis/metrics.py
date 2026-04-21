from __future__ import annotations

import math

import pandas as pd

from src.llm.parser import normalize_answer_letter


def accuracy_by_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy per setup using majority vote across repetitions."""
    df = df.copy()
    df["extracted_answer"] = df["extracted_answer"].map(normalize_answer_letter)
    df["correct_answer"] = df["correct_answer"].map(normalize_answer_letter)

    # For each (question, setup), take majority vote
    votes = (
        df.groupby(["question_id", "setup_name"])["extracted_answer"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
    )
    votes = votes.rename(columns={"extracted_answer": "majority_answer"})

    # Merge correct answer
    correct = df[["question_id", "correct_answer"]].drop_duplicates()
    votes = votes.merge(correct, on="question_id")
    votes["majority_correct"] = votes["majority_answer"] == votes["correct_answer"]

    # Accuracy per setup
    acc = (
        votes.groupby("setup_name")["majority_correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy", "sum": "correct", "count": "total"})
        .reset_index()
    )
    return acc


def mean_accuracy_by_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean per-question accuracy (fraction of reps correct)."""
    per_q = (
        df.groupby(["question_id", "setup_name"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "question_accuracy"})
    )
    return (
        per_q.groupby("setup_name")["question_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"question_accuracy": "mean_accuracy"})
    )


def consistency_by_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Compute consistency: for each question, fraction of reps giving same answer."""
    def _consistency(group):
        if len(group) == 0:
            return 0.0
        counts = group["extracted_answer"].value_counts(dropna=False)
        if counts.empty:
            return 0.0
        most_common_count = counts.iloc[0]
        return most_common_count / len(group)

    per_q = (
        df.groupby(["question_id", "setup_name"])
        .apply(_consistency, include_groups=False)
        .reset_index()
        .rename(columns={0: "consistency"})
    )
    return (
        per_q.groupby("setup_name")["consistency"]
        .mean()
        .reset_index()
    )


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return 0.0, 0.0
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_failure_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fraction of responses where answer extraction failed."""
    df_copy = df.copy()
    df_copy["extracted_answer"] = df_copy["extracted_answer"].map(normalize_answer_letter)
    df_copy["parse_failed"] = df_copy["extracted_answer"].isna()
    return (
        df_copy.groupby("setup_name")["parse_failed"]
        .mean()
        .reset_index()
        .rename(columns={"parse_failed": "parse_failure_rate"})
    )
