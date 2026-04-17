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

2. Pull required models — generation models and the embedding model used for RAG:

```bash
ollama pull gpt-oss:20b
ollama pull gemma3:4b
ollama pull edwardlo12/medgemma-4b-it-q4_k_m:latest
ollama pull nomic-embed-text
```

3. Build the RAG index (required for `+RAG` setups; takes ~25 min on first run, downloads ~86 MB of data):

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

## Datasets

| Role                 | Dataset                                          | Split used |
| -------------------- | ------------------------------------------------ | ---------- |
| Experiment questions | `GBaker/MedQA-USMLE-4-options`                   | `test`     |
| RAG knowledge corpus | `openlifescienceai/medmcqa` (explanations only)  | `train`    |

**Experiment questions** are USMLE Step 1/2/3 style 4-option MCQs. The LLMs are evaluated on these.

**RAG corpus** is built from the `exp` (explanation) field of MedMCQA — Indian PG medical exam explanations covering anatomy, pharmacology, pathology, etc. Only the plain explanation text is indexed; questions and answer options from MedMCQA are never stored. This serves as a proxy for medical textbook knowledge since the original MedQA textbook source (`bigbio/med_qa`) is no longer loadable with `datasets >= 4`.

The two datasets are completely disjoint: no MedQA test questions or answers appear in the RAG index.

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
- The RAG index (`data/chroma_db/`) is not committed to git — run `build_rag_index.py` after cloning.

## Project Entry Points

Also available after `pip install -e .`:

- `download-dataset`
- `build-rag-index`
- `run-experiment`
- `analyze-results`
