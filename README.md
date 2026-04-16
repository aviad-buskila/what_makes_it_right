# What Makes It Right

Experiment framework for medical multiple-choice QA to compare:
- model size
- domain specialization
- retrieved context (RAG)

The project runs each model in two modes (`base` and `+RAG`), repeats questions multiple times, stores all runs in JSONL, and generates a markdown report with plots and significance tests.

## Requirements

- Python `>=3.11`
- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- Models configured in `config.yaml` available in Ollama

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

1. Start Ollama:

```bash
ollama serve
```

2. Pull required generation models (from `config.yaml`):

```bash
ollama pull gpt-oss:20b
ollama pull gemma3:4b
ollama pull edwardlo12/medgemma-4b-it-q4_k_m:latest
```

3. Build RAG index (optional but needed for `+RAG` setups):

```bash
python scripts/build_rag_index.py --config config.yaml
```

4. Run experiment:

```bash
python scripts/run_experiment.py --config config.yaml
```

5. Analyze results:

```bash
python scripts/analyze_results.py --experiment <experiment_name>
```

Use the same `<experiment_name>` as `experiment.name` in `config.yaml`.

## Main Config

Edit `config.yaml`:

- `experiment.name`: output file name under `results/`
- `experiment.repetitions`: repeated trials per question
- `experiment.max_questions`: subsample size (`null` for full split)
- `experiment.temperature`: model temperature
- `experiment.timeout_per_query`: per-call timeout
- `experiment.record_failures`: whether failed calls are written to JSONL
- `models`: list of Ollama models to compare
- `rag`: retrieval settings (`top_k`, chunking, embedding model)

## Data and Outputs

- Questions: `data/medqa/questions.jsonl`
- Chroma index: `data/chroma_db/`
- Run results: `results/<experiment_name>.jsonl`
- Analysis report: `results/report.md`
- Figures: `results/*.png`

## Useful Commands

Download and save questions:

```bash
python scripts/download_dataset.py --split test --max-questions 100 --output data/medqa/questions.jsonl
```

Smoke config run:

```bash
python scripts/run_experiment.py --config config_smoke.yaml
```

Analyze a non-default experiment:

```bash
python scripts/analyze_results.py --experiment medical_mcq_comparison_try
```

## Reliability Features

- Retry with exponential backoff for generation and embedding calls
- Per-call timeout support
- One-time embedding-model ensure before RAG startup
- Per-sample exception handling so one failed call does not stop the experiment
- Optional explicit failure rows in results (`succeeded=false`, `error_message`)
- Correct-answer normalization to `A/B/C/D` with fail-fast validation

## Common Pitfalls

- `analyze_results.py` does **not** read `config.yaml`; it uses `--experiment`.
- Results are append-only by experiment file. If you want a fresh run:
  - delete `results/<experiment_name>.jsonl`, or
  - change `experiment.name` in `config.yaml`.
- If you see `model "nomic-embed-text" not found`, run:

```bash
ollama pull nomic-embed-text
```

## Project Entry Points

Also available after `pip install -e .`:

- `download-dataset`
- `build-rag-index`
- `run-experiment`
- `analyze-results`
