# What Makes It Right

Experiment framework for multiple-choice QA that compares:
- model size
- domain specialization (finetuned / continually-pretrained open models)
- retrieved context (RAG over authoritative, structured knowledge bases)

The framework is **domain-agnostic**. Two domains are shipped out of the box:

| Domain         | Questions                                        | Knowledge base                                             |
| -------------- | ------------------------------------------------ | ---------------------------------------------------------- |
| Medical        | MedQA-USMLE-4-options                            | MedMCQA explanations (proxy for textbook knowledge)        |
| Cybersecurity  | CyberMetric (primary) / SecQA (textbook-grounded) | MITRE ATT&CK + CWE + NIST SP 800-53 r5 + OWASP Top 10 (2021) |

Each model runs in two modes (`base` and `+RAG`), questions are repeated, all runs are stored as JSONL, and a markdown report with plots and significance tests is generated.

All prompts now enforce a strict machine-readable answer contract:
`{"answer":"A|B|C|D"}` (no explanation text).

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

2. Pull required models for the domain you target.

   **Medical** (original baselines):

   ```bash
   ollama pull gpt-oss:20b
   ollama pull gemma3:4b
   ollama pull edwardlo12/medgemma-4b-it-q4_k_m:latest
   ollama pull nomic-embed-text
   ```

   **Cybersecurity** (baselines used in `config_cyber.yaml`):

   ```bash
   ollama pull llama3.1:8b
   ollama pull qwen2.5:7b
   ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Q4_K_M-GGUF:Q4_K_M   # security-finetuned
   ollama pull nomic-embed-text
   ```

3. Build the RAG index (required for `+RAG` setups):

   ```bash
   # medical
   python scripts/build_rag_index.py --config config.yaml
   # cybersecurity
   python scripts/build_rag_index.py --config config_cyber.yaml
   ```

4. Run experiment:

   ```bash
   python scripts/run_experiment.py --config config_cyber.yaml
   ```

5. Analyze results:

   ```bash
   python scripts/analyze_results.py --experiment <experiment_name>
   ```

Use the same `<experiment_name>` as `experiment.name` in the config file.

## Main Config

Edit the YAML config you want to run:

- `experiment.name`: output file name under `results/`
- `experiment.repetitions`: repeated trials per question
- `experiment.max_questions`: subsample size (`null` for full split)
- `experiment.temperature`: model temperature
- `experiment.timeout_per_query`: per-call timeout
- `experiment.answer_retry_attempts`: retries when output is not a valid answer letter
- `experiment.record_failures`: whether failed calls are written to JSONL
- `experiment.domain`: `"medical"` or `"cybersecurity"` — drives the prompt persona
- `dataset.source`: `"medqa"` | `"cybermetric"` | `"secqa"`
- `dataset.variant`: CyberMetric size (80/500/2000/10000) or SecQA version (v1/v2)
- `models`: list of Ollama models to compare
- `rag.corpus_source`: `"medqa"` or `"cybersecurity"`
- `rag.corpus_include`: (cybersecurity only) any subset of `[attack, cwe, nist, owasp]` — enables per-source ablations
- `rag.persist_dir`: ChromaDB directory (keep separate per domain, e.g. `data/chroma_cyber`)
- `rag.retrieval_mode`: `fast` (dense only), `balanced` (hybrid rerank), `best` (hybrid + lexical candidate expansion)
- `rag`: retrieval settings (`top_k`, chunking, embedding model, `max_distance`, multipliers, rerank weights)

## Datasets

### Medical

| Role                 | Dataset                                          | Split used |
| -------------------- | ------------------------------------------------ | ---------- |
| Experiment questions | `GBaker/MedQA-USMLE-4-options`                   | `test`     |
| RAG knowledge corpus | `openlifescienceai/medmcqa` (explanations only)  | `train`    |

USMLE Step 1/2/3 style 4-option MCQs. The RAG corpus uses the `exp` (explanation) field of MedMCQA — questions and answer options are never stored. The two datasets are disjoint.

### Cybersecurity

| Role                 | Source                                                                    |
| -------------------- | ------------------------------------------------------------------------- |
| Experiment questions | `CyberMetric-{80,500,2000,10000}` (Tihanyi et al., 2024) — primary benchmark |
|                      | `zefang-liu/secqa` — textbook-grounded secondary benchmark                |
| RAG knowledge corpus | MITRE ATT&CK Enterprise (STIX 2.x JSON, `mitre/cti`)                      |
|                      | MITRE CWE (official CSV export from cwe.mitre.org)                        |
|                      | NIST SP 800-53 r5 (OSCAL JSON from `usnistgov/oscal-content`)             |
|                      | OWASP Top 10 (2021) markdown from the official `OWASP/Top10` repo         |

All cybersecurity KB sources are public, authoritative, and machine-readable. Each entity becomes a self-contained document (ID + title + body) before chunking, which lets the analysis pipeline trace retrieved context back to its source ID (e.g. `CWE-79`, `T1059.003`, `AC-6`). Set `rag.corpus_include` to a subset of `[attack, cwe, nist, owasp]` for per-source ablations.

## Retrieval-only Evaluation (debug / calibrate / tune)

`scripts/evaluate_retrieval.py` measures RAG quality **without running any LLM**, so you can iterate on chunking, indexing, and retrieval parameters in minutes instead of hours. It computes:

- **Intrinsic**: coverage rate, distance distribution (mean / p10 / median / p90), chunk-length stats, per-query latency, source-provenance mix (ATT&CK / CWE / NIST / OWASP).
- **Silver-label (no LLM)**: correct-option term recall vs. retrieved context, discrimination = overlap(correct) − mean overlap(distractors), and a retrieval-only MCQ accuracy upper bound (pick the option with maximum overlap to the retrieved chunks — this is what RAG alone could give a lookup-table "model").

```bash
# full eval on the configured question set
python scripts/evaluate_retrieval.py --config config_cyber.yaml

# fast iteration — first 50 questions
python scripts/evaluate_retrieval.py --config config_cyber.yaml --limit 50

# grid sweep over top_k and max_distance (no re-embedding)
python scripts/evaluate_retrieval.py --config config_cyber.yaml --sweep \
    --top-k-grid 1,3,5,8,10 --max-distance-grid 0.20,0.25,0.30,0.35,0.40,1.0

# single-query debug (prints chunks, distances, inferred source)
python scripts/evaluate_retrieval.py --config config_cyber.yaml \
    --query "Which mitigation maps to CWE-79?"

# per-question deep-dive (overlap per option, gold vs heuristic pick)
python scripts/evaluate_retrieval.py --config config_cyber.yaml \
    --question-id cybermetric_2000_42
```

Outputs are written to `results/retrieval/<experiment_name>/`:
- `per_query.jsonl` — full diagnostics per question
- `metrics.json` — aggregates + the exact `rag` config used
- `distance_top1_hist.png`, `distance_topk_hist.png` — for tuning `max_distance`
- `source_distribution.png` — coverage mix across KB sources
- `sweep.json` + `sweep_report.md` — ranked sweep results

### How to use the output to tune the pipeline
- **`max_distance`**: read it off the top-1 distance histogram. Pick the knee.
- **`top_k`**: look at the sweep row where retrieval-only accuracy plateaus — adding more chunks past that just adds noise.
- **Chunk size / overlap**: rebuild the index with different `rag.chunk_size` / `rag.chunk_overlap`, rerun the eval, compare `mean_discrimination` and retrieval-only accuracy.
- **Corpus ablation**: toggle `rag.corpus_include` in the config, rebuild, and compare source-mix vs. accuracy to measure each KB's marginal contribution.

## Data and Outputs

- Medical questions: `data/medqa/questions.jsonl`
- Cybersecurity questions: `data/cybermetric/questions.jsonl` (or `data/secqa/`)
- Chroma index (medical): `data/chroma_db/`
- Chroma index (cyber): `data/chroma_cyber/`
- Raw KB cache (cyber): `data/cyber_kb/{attack,cwe,nist,owasp}/`
- Run results: `results/<experiment_name>.jsonl`
- Analysis report: `results/report.md`
- Figures: `results/*.png`

## Useful Commands

Download and save questions:

```bash
# medical
python scripts/download_dataset.py --source medqa --split test --max-questions 100
# cybersecurity
python scripts/download_dataset.py --source cybermetric --variant 2000 --max-questions 100
python scripts/download_dataset.py --source secqa --variant v2
```

Smoke config run:

```bash
# medical
python scripts/run_experiment.py --config config_smoke.yaml
# cybersecurity
python scripts/build_rag_index.py --config config_cyber_smoke.yaml
python scripts/run_experiment.py --config config_cyber_smoke.yaml
```

Analyze a non-default experiment:

```bash
# medical
python scripts/analyze_results.py --experiment medical_mcq_comparison_try
# cybersecurity
python scripts/analyze_results.py --experiment cyber_mcq_comparison_3_100_exp
```

Note: report wording is now domain-aware and inferred from `question_id` prefixes
(`medqa_*`, `cybermetric_*`, `secqa_*`), so generated `results/report.md` is no
longer hardcoded to medical phrasing.

## Reliability Features

- Retry with exponential backoff for generation and embedding calls
- Per-call timeout support
- Invalid-answer retry loop when model output is not `A/B/C/D`
- One-time embedding-model ensure before RAG startup
- Retrieval is performed once per question and reused across all RAG model variants
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
