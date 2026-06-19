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


def rag_conflict_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Explain RAG-changed items using the stored retrieved context.

    For every (question, model) pair where toggling RAG changed majority-vote
    correctness, characterise the retrieved context by term overlap with the
    gold option vs. the distractors:

      * ``gold_overlap``       — recall of gold-option terms in the context
      * ``distractor_overlap`` — mean recall of distractor-option terms
      * ``evidence``           — "supports_gold" if gold_overlap is highest,
                                 "supports_distractor" if a distractor is higher,
                                 "no_evidence" if the context has no option terms.

    This lets the discussion distinguish RAG hurting because the retrieved
    passage *conflicts* with the gold answer (supports a distractor) from RAG
    failing because the passage was simply *irrelevant* (no_evidence).
    """
    delta = rag_delta_per_question(df)
    changed = delta[delta["delta"] != 0]
    if changed.empty:
        return pd.DataFrame()

    rag_rows = df[(df["has_rag"] == True)].dropna(subset=["retrieved_context"])
    context_by_q: dict[str, list[str]] = {}
    for _, r in rag_rows.iterrows():
        ctx = r["retrieved_context"]
        if isinstance(ctx, list) and ctx and r["question_id"] not in context_by_q:
            context_by_q[r["question_id"]] = ctx

    # Reconstruct option text per question from any row carrying it (if present).
    # Falls back gracefully when option text is not stored in results.
    rows = []
    for _, row in changed.iterrows():
        qid = row["question_id"]
        ctx_chunks = context_by_q.get(qid, [])

        record = {
            "question_id": qid,
            "model_name": row["model_name"],
            "effect": row["effect"],
            "n_chunks": len(ctx_chunks),
        }
        # Option text is not in the results store; evidence labelling requires it.
        # When unavailable we still report effect + chunk count so the breakdown
        # of helped/hurt with vs without retrieved context is computable.
        record["had_context"] = len(ctx_chunks) > 0
        rows.append(record)

    return pd.DataFrame(rows)


def rag_conflict_summary(conflict_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise RAG-changed items by effect and whether context was present."""
    if conflict_df.empty:
        return pd.DataFrame()
    return (
        conflict_df.groupby(["model_name", "effect", "had_context"])
        .size()
        .reset_index(name="count")
    )


def rag_conflict_analysis_with_options(
    df: pd.DataFrame,
    questions: list,
) -> pd.DataFrame:
    """Richer conflict analysis when the original questions (with option text)
    are available.

    ``questions`` is a list of :class:`~src.dataset.loader.Question`. Provides the
    ``gold_overlap`` / ``distractor_overlap`` / ``evidence`` labelling described
    in :func:`rag_conflict_analysis`.
    """
    from statistics import mean

    from src.rag.evaluation import _term_overlap

    q_by_id = {q.id: q for q in questions}
    delta = rag_delta_per_question(df)
    changed = delta[delta["delta"] != 0]
    if changed.empty:
        return pd.DataFrame()

    rag_rows = df[(df["has_rag"] == True)].dropna(subset=["retrieved_context"])
    context_by_q: dict[str, list[str]] = {}
    for _, r in rag_rows.iterrows():
        ctx = r["retrieved_context"]
        if isinstance(ctx, list) and ctx and r["question_id"] not in context_by_q:
            context_by_q[r["question_id"]] = ctx

    rows = []
    for _, row in changed.iterrows():
        qid = row["question_id"]
        q = q_by_id.get(qid)
        if q is None:
            continue
        context_blob = "\n\n".join(context_by_q.get(qid, []))
        gold_recall, _, _ = _term_overlap(q.options.get(q.correct_answer, ""), context_blob)
        distractor_recalls = [
            _term_overlap(text, context_blob)[0]
            for letter, text in q.options.items()
            if letter != q.correct_answer
        ]
        distractor_mean = mean(distractor_recalls) if distractor_recalls else 0.0
        max_distractor = max(distractor_recalls) if distractor_recalls else 0.0

        if gold_recall == 0 and max_distractor == 0:
            evidence = "no_evidence"
        elif gold_recall >= max_distractor:
            evidence = "supports_gold"
        else:
            evidence = "supports_distractor"

        rows.append({
            "question_id": qid,
            "model_name": row["model_name"],
            "effect": row["effect"],
            "gold_overlap": round(gold_recall, 4),
            "distractor_overlap": round(distractor_mean, 4),
            "evidence": evidence,
        })
    return pd.DataFrame(rows)


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
