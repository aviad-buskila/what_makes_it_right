from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis.metrics import (
    accuracy_by_setup,
    consistency_by_setup,
    mean_accuracy_by_setup,
    parse_failure_rate,
    wilson_ci,
)
from src.analysis.statistics import all_pairwise_tests, rag_effect_tests


def generate_report(df: pd.DataFrame, output_dir: str = "results") -> str:
    """Generate a markdown report with charts from experiment results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    acc = accuracy_by_setup(df)
    mean_acc = mean_accuracy_by_setup(df)
    cons = consistency_by_setup(df)
    parse_fail = parse_failure_rate(df)
    pairwise = all_pairwise_tests(df)
    rag_tests = rag_effect_tests(df)

    # Add CIs to accuracy table
    acc["ci_lower"] = acc.apply(lambda r: wilson_ci(int(r["correct"]), int(r["total"]))[0], axis=1)
    acc["ci_upper"] = acc.apply(lambda r: wilson_ci(int(r["correct"]), int(r["total"]))[1], axis=1)

    # Merge all metrics
    summary = acc.merge(mean_acc, on="setup_name").merge(cons, on="setup_name").merge(parse_fail, on="setup_name")

    # Generate charts
    _plot_accuracy_bars(summary, out / "accuracy_comparison.png")
    _plot_consistency_bars(summary, out / "consistency_comparison.png")
    if not pairwise.empty:
        _plot_significance_heatmap(pairwise, out / "pairwise_significance.png")

    # Build markdown
    inferred_domain = _infer_domain_label(df)

    lines = [
        "# Experiment Report: What Makes It Right?",
        "",
        "## Research Question",
        (
            f"What is more crucial for {inferred_domain} multiple-choice QA accuracy: "
            "model size, domain expertise, or retrieved knowledge?"
        ),
        "",
        "## Setup Summary",
        f"- **Domain (inferred)**: {inferred_domain}",
        f"- **Questions**: {df['question_id'].nunique()}",
        f"- **Repetitions per question**: {df.groupby(['question_id', 'setup_name']).size().mode().iloc[0]}",
        f"- **Total LLM calls**: {len(df)}",
        "",
        "## Accuracy Results (Majority Vote)",
        "",
        summary[["setup_name", "accuracy", "ci_lower", "ci_upper", "correct", "total"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "![Accuracy Comparison](accuracy_comparison.png)",
        "",
        "## Mean Per-Question Accuracy",
        "",
        summary[["setup_name", "mean_accuracy"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Consistency (Agreement Across Repetitions)",
        "",
        summary[["setup_name", "consistency"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "![Consistency Comparison](consistency_comparison.png)",
        "",
        "## Parse Failure Rate",
        "",
        summary[["setup_name", "parse_failure_rate"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Pairwise Statistical Tests (McNemar's)",
        "",
    ]

    if not pairwise.empty:
        lines.append(pairwise.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        lines.append("![Pairwise Significance](pairwise_significance.png)")
    else:
        lines.append("No pairwise tests computed.")

    lines.extend([
        "",
        "## RAG Effect Tests",
        "",
    ])

    if not rag_tests.empty:
        lines.append(rag_tests.to_markdown(index=False, floatfmt=".4f"))
    else:
        lines.append("No RAG effect tests computed.")

    report_text = "\n".join(lines)
    report_path = out / "report.md"
    report_path.write_text(report_text)
    print(f"Report written to {report_path}")
    return report_text


def _plot_accuracy_bars(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for name in summary["setup_name"]:
        if "medgemma" in name.lower():
            colors.append("#e74c3c" if "+RAG" not in name else "#c0392b")
        elif "gpt-oss" in name.lower():
            colors.append("#3498db" if "+RAG" not in name else "#2980b9")
        else:
            colors.append("#2ecc71" if "+RAG" not in name else "#27ae60")

    bars = ax.bar(summary["setup_name"], summary["accuracy"], color=colors)
    ax.errorbar(
        summary["setup_name"],
        summary["accuracy"],
        yerr=[
            summary["accuracy"] - summary["ci_lower"],
            summary["ci_upper"] - summary["accuracy"],
        ],
        fmt="none",
        color="black",
        capsize=5,
    )
    ax.set_ylabel("Accuracy (Majority Vote)")
    ax.set_title("MCQ Accuracy by Setup")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_consistency_bars(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary["setup_name"], summary["consistency"], color="#9b59b6")
    ax.set_ylabel("Consistency (Agreement Rate)")
    ax.set_title("Answer Consistency Across Repetitions")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_significance_heatmap(pairwise: pd.DataFrame, path: Path) -> None:
    setups = sorted(set(pairwise["setup_a"]) | set(pairwise["setup_b"]))
    matrix = pd.DataFrame(1.0, index=setups, columns=setups)
    for _, row in pairwise.iterrows():
        matrix.loc[row["setup_a"], row["setup_b"]] = row["p_value"]
        matrix.loc[row["setup_b"], row["setup_a"]] = row["p_value"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        ax=ax,
    )
    ax.set_title("Pairwise McNemar's Test p-values")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _infer_domain_label(df: pd.DataFrame) -> str:
    """Best-effort domain label inferred from question ids."""
    ids = set(df["question_id"].astype(str).str.lower().tolist())
    if any(q.startswith("cybermetric_") or q.startswith("secqa_") for q in ids):
        return "cybersecurity"
    if any(q.startswith("medqa_") for q in ids):
        return "medical"
    return "the configured domain"
