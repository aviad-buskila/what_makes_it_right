"""Per-model error analysis: question verdicts, answer bias, RAG delta, difficulty."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.error import (
    answer_bias,
    difficulty_spectrum,
    plot_answer_bias,
    plot_difficulty_histogram,
    plot_rag_effect,
    question_verdict_pivot,
    rag_delta_per_question,
    rag_effect_summary,
)
from src.storage.results import ResultsStore


def main():
    parser = argparse.ArgumentParser(description="Error analysis for experiment results")
    parser.add_argument("--experiment", required=True, help="Experiment name (matches results/<name>.jsonl)")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="results", help="Output directory for report")
    parser.add_argument("--n-hardest", type=int, default=10, help="Number of hardest/easiest questions to show")
    args = parser.parse_args()

    store = ResultsStore(args.results_dir)
    df = store.load(args.experiment)

    if df.empty:
        print(f"No results found for experiment '{args.experiment}'.")
        return

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} rows — {df['question_id'].nunique()} questions, {df['setup_name'].nunique()} setups.")

    # --- Compute ---
    pivot = question_verdict_pivot(df)
    bias = answer_bias(df)
    delta = rag_delta_per_question(df)
    effect_summary = rag_effect_summary(delta)
    difficulty = difficulty_spectrum(pivot, n_hardest=args.n_hardest)

    # --- Plots ---
    plot_answer_bias(bias, out / "answer_bias.png")
    plot_rag_effect(effect_summary, out / "rag_effect_per_question.png")
    plot_difficulty_histogram(pivot, out / "difficulty_histogram.png")
    print("Plots saved.")

    # --- Report ---
    lines = [
        "# Error Analysis Report",
        "",
        f"**Experiment**: `{args.experiment}`  ",
        f"**Questions**: {df['question_id'].nunique()}  ",
        f"**Setups**: {df['setup_name'].nunique()}",
        "",
        "---",
        "",
        "## Question Difficulty Distribution",
        "",
        "How many setups answered each question correctly.",
        "",
        "![Difficulty histogram](difficulty_histogram.png)",
        "",
    ]

    # Difficulty counts table
    setup_cols = [c for c in pivot.columns if c not in ("question_id", "correct_answer", "models_correct")]
    n_setups = len(setup_cols)
    diff_counts = pivot["models_correct"].value_counts().sort_index().reset_index()
    diff_counts.columns = ["setups_correct", "n_questions"]
    diff_counts["pct"] = (diff_counts["n_questions"] / len(pivot) * 100).round(1)
    lines += [
        diff_counts.to_markdown(index=False),
        "",
        f"- **All wrong** (0/{n_setups}): {(pivot['models_correct'] == 0).sum()} questions",
        f"- **All correct** ({n_setups}/{n_setups}): {(pivot['models_correct'] == n_setups).sum()} questions",
        "",
        "---",
        "",
        "## Hardest Questions (fewest setups correct)",
        "",
    ]

    hardest = difficulty[difficulty["difficulty"] == "hardest"][
        ["question_id", "correct_answer", "models_correct"] + setup_cols
    ]
    # Render booleans as ✓/✗ for readability
    display = hardest.copy()
    for col in setup_cols:
        display[col] = display[col].map({True: "✓", False: "✗", None: "?"})
    lines += [display.to_markdown(index=False), ""]

    lines += [
        "---",
        "",
        "## RAG Effect Per Question",
        "",
        "For each model: how many questions did RAG help, hurt, or leave unchanged?",
        "",
        "![RAG effect](rag_effect_per_question.png)",
        "",
        effect_summary.to_markdown(index=False),
        "",
    ]

    # Per-model breakdown
    for model in delta["model_name"].unique():
        model_delta = delta[delta["model_name"] == model]
        helped = model_delta[model_delta["effect"] == "RAG helped"]["question_id"].tolist()
        hurt = model_delta[model_delta["effect"] == "RAG hurt"]["question_id"].tolist()
        lines += [
            f"### {model}",
            f"- RAG helped on: {helped if helped else 'none'}",
            f"- RAG hurt on:   {hurt if hurt else 'none'}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Answer Choice Bias",
        "",
        "Predicted vs true answer-choice frequencies. Uniform = 0.25 per choice.",
        "",
        "![Answer bias](answer_bias.png)",
        "",
        bias[[c for c in bias.columns if c.startswith("pred_") or c == "setup_name"]].to_markdown(index=False, floatfmt=".3f"),
        "",
    ]

    report_text = "\n".join(lines)
    report_path = out / "error_analysis.md"
    report_path.write_text(report_text)
    print(f"Report written to {report_path}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
