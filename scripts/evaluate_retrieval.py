"""Retrieval-only evaluation, debugging, and tuning CLI.

Three modes:

  default          python scripts/evaluate_retrieval.py --config config_cyber.yaml
                   Evaluate the configured retriever against the question set
                   and write per-query diagnostics, aggregate metrics, and
                   plots under results/retrieval/<experiment_name>/.

  --sweep          Sweep over top_k x max_distance without rebuilding the
                   index. Writes sweep_report.md ranking configurations by
                   retrieval-only accuracy and discrimination.

  --query TEXT     Ad-hoc debug: show the retriever's behaviour on a single
                   arbitrary query.

  --question-id    Ad-hoc debug: rerun a specific question from the dataset
                   with full diagnostics printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.loader import Question, load_dataset_by_source, load_questions, save_questions
from src.experiment.config import load_config
from src.llm.client import ensure_model
from src.rag.evaluation import (
    aggregate,
    evaluate_query,
    infer_source,
    plot_distance_histogram,
    plot_source_distribution,
)
from src.rag.retriever import Retriever


def _default_questions_path(source: str) -> Path:
    return Path(f"data/{source}/questions.jsonl")


def _load_questions(config, explicit_path: str | None) -> list[Question]:
    qpath = Path(explicit_path) if explicit_path else _default_questions_path(
        config.dataset.source
    )
    if qpath.exists():
        return load_questions(qpath)
    print(f"Questions file not found at {qpath}, downloading...")
    questions = load_dataset_by_source(
        source=config.dataset.source,
        max_questions=config.max_questions,
        random_seed=config.random_seed,
        cache_dir=config.dataset.cache_dir,
        split=config.dataset.split,
        variant=config.dataset.variant,
    )
    save_questions(questions, qpath)
    return questions


def _build_retriever(config, max_distance: float | None = None,
                     top_k: int | None = None) -> Retriever:
    ensure_model(config.rag.embedding_model)
    return Retriever(
        persist_dir=config.rag.persist_dir,
        collection_name=config.rag.collection_name,
        embedding_model=config.rag.embedding_model,
        top_k=top_k if top_k is not None else config.rag.top_k,
        timeout_per_query=config.timeout_per_query,
        max_distance=max_distance if max_distance is not None
        else config.rag.max_distance,
    )


def _write_per_query(path: Path, diagnostics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in diagnostics:
            f.write(json.dumps(asdict(d)) + "\n")


def _write_metrics(path: Path, metrics, rag_cfg) -> None:
    payload = asdict(metrics)
    payload["config"] = {
        "collection_name": rag_cfg.collection_name,
        "persist_dir": rag_cfg.persist_dir,
        "embedding_model": rag_cfg.embedding_model,
        "top_k": rag_cfg.top_k,
        "max_distance": rag_cfg.max_distance,
        "chunk_size": rag_cfg.chunk_size,
        "chunk_overlap": rag_cfg.chunk_overlap,
        "corpus_source": rag_cfg.corpus_source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _print_aggregate(metrics) -> None:
    print(f"\n=== retrieval metrics (n={metrics.n_queries}) ===")
    print(f"coverage                   : {metrics.coverage:.3f}")
    print(f"retrieval-only accuracy    : {metrics.retrieval_only_accuracy:.3f}  "
          f"(chance = 0.25 for 4-option MCQ)")
    print(f"mean correct-term recall   : {metrics.mean_correct_term_recall:.3f}")
    print(f"mean discrimination        : {metrics.mean_discrimination:+.3f}")
    print(f"mean top-1 distance        : {metrics.mean_top1_distance:.3f}")
    print(f"  p10/median/p90           : "
          f"{metrics.p10_top1_distance:.3f} / "
          f"{metrics.median_top1_distance:.3f} / "
          f"{metrics.p90_top1_distance:.3f}")
    print(f"mean chunks returned (/q)  : {metrics.mean_chunks_returned:.2f}")
    print(f"mean chunk length (chars)  : {metrics.mean_chunk_length:.0f}")
    print(f"mean retrieval latency     : {metrics.mean_latency_seconds*1000:.0f} ms")
    if metrics.source_mix:
        print("source mix:")
        for src, frac in sorted(metrics.source_mix.items(),
                                key=lambda x: x[1], reverse=True):
            print(f"  {src:15s} {frac:6.1%}")


def run_eval(config, questions, output_dir: Path) -> None:
    retriever = _build_retriever(config)
    print(f"retriever: collection='{retriever.collection.name}' "
          f"chunks={retriever.collection.count()} "
          f"top_k={retriever.top_k} max_distance={retriever.max_distance}")

    diagnostics = []
    for q in questions:
        diagnostics.append(evaluate_query(retriever, q))
        if len(diagnostics) % 25 == 0:
            print(f"  evaluated {len(diagnostics)}/{len(questions)}")

    _write_per_query(output_dir / "per_query.jsonl", diagnostics)
    metrics = aggregate(diagnostics)
    _write_metrics(output_dir / "metrics.json", metrics, config.rag)

    all_top1 = [d.distances[0] for d in diagnostics if d.distances]
    plot_distance_histogram(all_top1, output_dir / "distance_top1_hist.png")
    all_distances = [x for d in diagnostics for x in d.distances]
    plot_distance_histogram(all_distances, output_dir / "distance_topk_hist.png")
    plot_source_distribution(metrics.source_mix, output_dir / "source_distribution.png")

    _print_aggregate(metrics)
    print(f"\nwrote diagnostics to {output_dir}")


def run_sweep(config, questions, output_dir: Path,
              top_k_grid: list[int], max_distance_grid: list[float]) -> None:
    print(f"sweep grid: top_k={top_k_grid}  max_distance={max_distance_grid}")
    rows = []
    for max_dist in max_distance_grid:
        for k in top_k_grid:
            retriever = _build_retriever(config, max_distance=max_dist, top_k=k)
            diagnostics = [evaluate_query(retriever, q, top_k=k)
                           for q in questions]
            m = aggregate(diagnostics)
            rows.append({
                "top_k": k,
                "max_distance": max_dist,
                "coverage": m.coverage,
                "retrieval_only_acc": m.retrieval_only_accuracy,
                "mean_correct_recall": m.mean_correct_term_recall,
                "mean_discrimination": m.mean_discrimination,
                "mean_top1_dist": m.mean_top1_distance,
                "mean_latency_ms": m.mean_latency_seconds * 1000,
            })
            print(f"  k={k:>2} maxd={max_dist:.2f}  "
                  f"cov={m.coverage:.2f}  acc={m.retrieval_only_accuracy:.3f}  "
                  f"disc={m.mean_discrimination:+.3f}")

    rows.sort(key=lambda r: (r["retrieval_only_acc"],
                             r["mean_discrimination"]), reverse=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep.json").write_text(json.dumps(rows, indent=2))

    md_lines = [
        "# Retrieval sweep",
        "",
        f"Collection: `{config.rag.collection_name}`  "
        f"(persist_dir `{config.rag.persist_dir}`)",
        f"Questions: {len(questions)}  "
        f"Embedding: `{config.rag.embedding_model}`",
        "",
        "| top_k | max_distance | coverage | retrieval-only acc | "
        "discrimination | mean top-1 dist | latency (ms) |",
        "|------:|-------------:|---------:|-------------------:|"
        "---------------:|----------------:|-------------:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['top_k']} | {r['max_distance']:.2f} | {r['coverage']:.3f} | "
            f"{r['retrieval_only_acc']:.3f} | {r['mean_discrimination']:+.3f} | "
            f"{r['mean_top1_dist']:.3f} | {r['mean_latency_ms']:.0f} |"
        )
    md_lines += ["", "Rows sorted by (retrieval-only accuracy, discrimination) desc."]
    (output_dir / "sweep_report.md").write_text("\n".join(md_lines))
    print(f"\nsweep report written to {output_dir/'sweep_report.md'}")


def run_single_query(config, text: str) -> None:
    retriever = _build_retriever(config)
    print(f"query: {text!r}")
    chunks = retriever.query(text)
    if not chunks:
        print("no chunks retrieved above distance threshold.")
        return
    for i, chunk in enumerate(chunks, 1):
        src = infer_source(chunk)
        preview = chunk[:400].replace("\n", " ") + ("…" if len(chunk) > 400 else "")
        print(f"\n[{i}] source={src} len={len(chunk)}")
        print(f"    {preview}")


def run_question_debug(config, questions, qid: str) -> None:
    match = next((q for q in questions if q.id == qid), None)
    if match is None:
        print(f"question id {qid!r} not found in dataset.")
        return
    retriever = _build_retriever(config)
    diag = evaluate_query(retriever, match)
    print(f"question: {match.question_text}")
    print(f"options:")
    for letter, text in sorted(match.options.items()):
        marker = "*" if letter == match.correct_answer else " "
        overlap = diag.option_overlap.get(letter, 0.0)
        print(f"  {marker} {letter}) {text}    [ctx overlap={overlap:.2f}]")
    print(f"gold={match.correct_answer}  heuristic_pick={diag.heuristic_pick}  "
          f"correct={diag.heuristic_correct}")
    print(f"correct-term recall={diag.correct_term_recall:.2f}  "
          f"discrimination={diag.discrimination:+.2f}  "
          f"latency={diag.latency_seconds*1000:.0f} ms")
    for i, (chunk, dist, src) in enumerate(
        zip(diag.chunks_preview, diag.distances, diag.sources), 1
    ):
        print(f"\n[{i}] dist={dist:.3f} source={src}\n    {chunk}")


def _parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Debug / calibrate / tune the RAG pipeline independently "
                    "of the LLM generation step."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--questions", default=None,
                        help="Override questions JSONL path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (fast iteration).")
    parser.add_argument("--output", default=None,
                        help="Output dir (default results/retrieval/<experiment>).")
    parser.add_argument("--sweep", action="store_true",
                        help="Run grid sweep over top_k and max_distance.")
    parser.add_argument("--top-k-grid", default="1,3,5,8,10")
    parser.add_argument("--max-distance-grid", default="0.20,0.25,0.30,0.35,0.40,1.0")
    parser.add_argument("--query", default=None,
                        help="Debug a single arbitrary query string.")
    parser.add_argument("--question-id", default=None,
                        help="Debug a single dataset question by id.")
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(args.output) if args.output else Path(
        f"results/retrieval/{config.name}"
    )

    if args.query is not None:
        run_single_query(config, args.query)
        return

    questions = _load_questions(config, args.questions)
    if args.limit:
        questions = questions[: args.limit]
    print(f"loaded {len(questions)} questions for {config.dataset.source}")

    if args.question_id:
        run_question_debug(config, questions, args.question_id)
        return

    if args.sweep:
        run_sweep(
            config,
            questions,
            output_dir,
            top_k_grid=_parse_int_list(args.top_k_grid),
            max_distance_grid=_parse_float_list(args.max_distance_grid),
        )
        return

    run_eval(config, questions, output_dir)


if __name__ == "__main__":
    main()
