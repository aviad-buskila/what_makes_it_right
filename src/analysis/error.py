from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def question_verdict_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (question × setup) pivot of majority-vote correctness.

    Columns: question_id, correct_answer, question_snippet, one bool column
    per setup, and a 'models_correct' tally.
    """
    # Majority vote per (question, setup)
    votes = (
        df.groupby(["question_id", "setup_name"])["extracted_answer"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={"extracted_answer": "majority_answer"})
    )
    correct_map = df[["question_id", "correct_answer"]].drop_duplicates()
    votes = votes.merge(correct_map, on="question_id")
    votes["correct"] = votes["majority_answer"] == votes["correct_answer"]

    pivot = votes.pivot(index="question_id", columns="setup_name", values="correct")
    pivot = pivot.merge(correct_map.set_index("question_id"), left_index=True, right_index=True)
    pivot["models_correct"] = pivot.drop(columns="correct_answer").sum(axis=1)
    pivot = pivot.reset_index().sort_values("models_correct")
    return pivot


def answer_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Return answer-choice frequency per setup.

    Shows how often each setup picks A/B/C/D/null compared to the true
    label distribution, making position bias visible.
    """
    choices = ["A", "B", "C", "D"]

    rows = []
    for setup, grp in df.groupby("setup_name"):
        total = len(grp)
        true_dist = grp["correct_answer"].value_counts(normalize=True)
        pred_dist = grp["extracted_answer"].value_counts(normalize=True)
        row = {"setup_name": setup}
        for c in choices:
            row[f"pred_{c}"] = pred_dist.get(c, 0.0)
            row[f"true_{c}"] = true_dist.get(c, 0.0)
        row["pred_null"] = grp["extracted_answer"].isna().sum() / total
        rows.append(row)

    return pd.DataFrame(rows)


def rag_delta_per_question(df: pd.DataFrame) -> pd.DataFrame:
    """Per-question, per-model: did RAG help (+1), hurt (-1), or change nothing (0)?

    Returns one row per (question_id, model_name) with columns:
      base_correct, rag_correct, delta, effect label.
    """
    votes = (
        df.groupby(["question_id", "setup_name", "model_name", "has_rag"])["extracted_answer"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={"extracted_answer": "majority_answer"})
    )
    correct_map = df[["question_id", "correct_answer"]].drop_duplicates()
    votes = votes.merge(correct_map, on="question_id")
    votes["correct"] = votes["majority_answer"] == votes["correct_answer"]

    base = votes[~votes["has_rag"]].rename(columns={"correct": "base_correct"})[
        ["question_id", "model_name", "base_correct"]
    ]
    rag = votes[votes["has_rag"]].rename(columns={"correct": "rag_correct"})[
        ["question_id", "model_name", "rag_correct"]
    ]

    merged = base.merge(rag, on=["question_id", "model_name"])
    merged["delta"] = merged["rag_correct"].astype(int) - merged["base_correct"].astype(int)
    merged["effect"] = merged["delta"].map({1: "RAG helped", -1: "RAG hurt", 0: "no change"})
    return merged.sort_values(["model_name", "delta"])


def rag_effect_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rag_delta_per_question into counts per model."""
    return (
        delta_df.groupby(["model_name", "effect"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def difficulty_spectrum(pivot: pd.DataFrame, n_hardest: int = 10) -> pd.DataFrame:
    """Return the N hardest and N easiest questions ranked by models_correct."""
    setup_cols = [c for c in pivot.columns if c not in ("question_id", "correct_answer", "models_correct")]
    cols = ["question_id", "correct_answer", "models_correct"] + setup_cols
    ranked = pivot[cols].sort_values("models_correct")
    hardest = ranked.head(n_hardest).copy()
    easiest = ranked.tail(n_hardest).copy()
    hardest["difficulty"] = "hardest"
    easiest["difficulty"] = "easiest"
    return pd.concat([hardest, easiest], ignore_index=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_answer_bias(bias_df: pd.DataFrame, path: Path) -> None:
    choices = ["A", "B", "C", "D"]
    setups = bias_df["setup_name"].tolist()

    pred_data = bias_df[[f"pred_{c}" for c in choices]].values
    true_data = bias_df[[f"true_{c}" for c in choices]].values

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(setups) * 0.6 + 2)), sharey=True)
    for ax, data, title in zip(axes, [pred_data, true_data], ["Predicted distribution", "True label distribution"]):
        im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=0.6)
        ax.set_xticks(range(len(choices)))
        ax.set_xticklabels(choices)
        ax.set_yticks(range(len(setups)))
        ax.set_yticklabels(setups)
        ax.set_title(title)
        for i in range(len(setups)):
            for j in range(len(choices)):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Answer Choice Distribution: predicted vs true labels")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rag_effect(effect_df: pd.DataFrame, path: Path) -> None:
    effect_cols = [c for c in ["RAG helped", "no change", "RAG hurt"] if c in effect_df.columns]
    plot_df = effect_df.set_index("model_name")[effect_cols]

    colors = {"RAG helped": "#2ecc71", "no change": "#bdc3c7", "RAG hurt": "#e74c3c"}
    plot_df.plot(
        kind="barh",
        stacked=True,
        color=[colors.get(c, "#888") for c in plot_df.columns],
        figsize=(9, max(3, len(plot_df) * 0.7 + 1.5)),
    )
    plt.xlabel("Number of questions")
    plt.title("RAG effect per question per model")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_difficulty_histogram(pivot: pd.DataFrame, path: Path) -> None:
    n_setups = pivot["models_correct"].max()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pivot["models_correct"], bins=range(0, int(n_setups) + 2), align="left", color="#3498db", edgecolor="white")
    ax.set_xlabel("Number of setups that answered correctly")
    ax.set_ylabel("Number of questions")
    ax.set_title("Question difficulty distribution")
    ax.set_xticks(range(0, int(n_setups) + 1))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
