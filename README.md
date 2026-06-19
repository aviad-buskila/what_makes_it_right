# What Makes It Right

Companion code for the paper **"Domain Fine-Tuning vs. Retrieval-Augmented Generation for Medical Multiple-Choice Question Answering: A Controlled Comparison at the 4B-Parameter Scale"** (Buskila, 2026; see [`paper/paper.tex`](paper/paper.tex)).

The repository contains a reproducible experiment framework for multiple-choice QA that holds the prompt, decoding settings, evaluation set, and retrieval pipeline fixed while varying only:

- **Domain adaptation** — general backbone vs. a domain-fine-tuned backbone of the same size.
- **Retrieval** — base prompt vs. the same prompt augmented with retrieved passages from a domain knowledge corpus (RAG).

The framework is domain-agnostic; the paper instantiates and analyzes the **medical** domain. A **cybersecurity** instantiation is also shipped for follow-up work (see *Other domains* below).

## Headline result (medical, 4B parameters)

A controlled $2\times2$ design — `{Gemma 3 4B, MedGemma 4B} × {no RAG, +RAG}` — evaluated on the full **MedQA-USMLE 4-option** test split (1,273 questions, 3 repetitions, 15,276 LLM calls). All models are 4-bit quantized and served locally via Ollama at temperature 0.1. Significance is McNemar's exact paired test on majority-vote correctness.

| Setup                  | Backbone            | Context                  | Accuracy   | 95% CI            | Correct |
| ---------------------- | ------------------- | ------------------------ | ---------- | ----------------- | ------- |
| `gemma3-4b`            | Gemma 3 4B (general)| question only            | 0.4643     | [0.4370, 0.4917]  | 591     |
| `gemma3-4b+RAG`        | Gemma 3 4B (general)| question + retrieved     | 0.4729     | [0.4456, 0.5004]  | 602     |
| `medgemma-4b+RAG`      | MedGemma 4B (domain)| question + retrieved     | 0.5137     | [0.4863, 0.5411]  | 654     |
| `medgemma-4b`          | MedGemma 4B (domain)| question only            | **0.5326** | [0.5051, 0.5599]  | **678** |

Pairwise McNemar's test:

- **Domain fine-tuning is decisive.** Every comparison that crosses the general/domain backbone boundary is significant at $p<0.005$; the headline gap (`medgemma-4b` vs `gemma3-4b`) is $+6.8$ pp at $p<10^{-4}$.
- **RAG does not move the needle at this scale.** Neither `+RAG` toggle is significant within a backbone. For MedGemma, the RAG point estimate is in fact slightly negative ($-1.9$ pp, $p=0.16$).
- **High decoding consistency** ($\geq 0.99$ across the 3 repetitions in every cell), so aggregation rule has negligible impact on the conclusions.

**Practical takeaway.** For a 4B-parameter, locally-deployable medical QA system, adopting a domain-adapted backbone is a more reliable lever than building a retrieval pipeline. RAG is not actively harmful here, but it is not a substitute for in-weights domain knowledge at this scale. See the paper's Discussion for the four hypotheses we cannot rule out (USMLE items are reasoning-driven; the corpus is broad but not authoritative; 4B has limited capacity to ground three new chunks; MedGemma may already encode much of the surfaceable textbook knowledge).

Pre-rendered figures live in `paper/figures/` and are regenerated from the analyzer (see *Run the medical experiment* below):

- `accuracy_comparison.png` — the four-cell accuracy bar chart with 95% CIs.
- `pairwise_significance.png` — the McNemar $p$-value matrix.
- `consistency_comparison.png` — within-setup agreement across repetitions.

## Requirements

- Python `>= 3.11`
- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- Models configured in `config.yaml` available in Ollama

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the medical experiment (reproduce the paper)

1. Start Ollama:

   ```bash
   ollama serve
   ```

2. Pull the models used in the paper:

   ```bash
   ollama pull gemma3:4b
   ollama pull edwardlo12/medgemma-4b-it-q4_k_m:latest
   ollama pull nomic-embed-text
   ```

3. Build the RAG index (MedMCQA explanations, embedded with `nomic-embed-text`, persisted to ChromaDB):

   ```bash
   python scripts/build_rag_index.py --config config.yaml
   ```

4. Run the 2×2 experiment (all four cells, 3 repetitions, full MedQA-USMLE 4-option test split):

   ```bash
   python scripts/run_experiment.py --config config.yaml
   ```

5. Generate the report (accuracy table, McNemar comparisons, consistency, figures):

   ```bash
   python scripts/analyze_results.py --experiment <experiment.name from config.yaml>
   ```

The analyzer writes `results/report.md` and the figures to `results/`.

## Main config

Edit the YAML config you want to run:

- `experiment.name` — output file name under `results/`
- `experiment.repetitions` — repeated trials per question (paper uses 3)
- `experiment.max_questions` — subsample size (`null` for the full 1,273-item test split)
- `experiment.temperature` — decoding temperature (paper uses 0.1)
- `experiment.timeout_per_query` — per-call timeout
- `experiment.answer_retry_attempts` — retries when output is not a valid `A/B/C/D`
- `experiment.record_failures` — whether failed calls are written to JSONL
- `experiment.domain` — `"medical"` or `"cybersecurity"` — drives the prompt persona
- `dataset.source` — `"medqa"` | `"cybermetric"` | `"secqa"`
- `dataset.variant` — CyberMetric size (80/500/2000/10000) or SecQA version (v1/v2)
- `models` — list of Ollama models to compare
- `rag.corpus_source` — `"medqa"` or `"cybersecurity"`
- `rag.corpus_include` — (cybersecurity only) any subset of `[attack, cwe, nist, owasp]` for per-source ablations
- `rag.persist_dir` — ChromaDB directory (keep separate per domain, e.g. `data/chroma_cyber`)
- `rag.retrieval_mode` — `fast` (dense only), `balanced` (hybrid rerank), `best` (hybrid + lexical candidate expansion)
- `rag` — retrieval settings (`top_k`, chunking, embedding model, `max_distance`, multipliers, rerank weights)

All prompts enforce a strict machine-readable answer contract: `{"answer":"A|B|C|D"}` (no explanation text).

## Datasets

### Medical (paper)

| Role                 | Dataset                                          | Split used |
| -------------------- | ------------------------------------------------ | ---------- |
| Experiment questions | `GBaker/MedQA-USMLE-4-options`                   | `test`     |
| RAG knowledge corpus | `openlifescienceai/medmcqa` (explanations only)  | `train`    |

USMLE Step 1/2/3 style 4-option MCQs. The RAG corpus uses the `exp` (explanation) field of MedMCQA — questions and answer options are never indexed. The two datasets are disjoint, so the corpus serves as a textbook-style proxy rather than a leakage path.

### Cybersecurity (additional domain, not in the paper)

| Role                 | Source                                                                    |
| -------------------- | ------------------------------------------------------------------------- |
| Experiment questions | `CyberMetric-{80,500,2000,10000}` (Tihanyi et al., 2024) — primary benchmark |
|                      | `zefang-liu/secqa` — textbook-grounded secondary benchmark                |
| RAG knowledge corpus | MITRE ATT&CK Enterprise (STIX 2.x JSON, `mitre/cti`)                      |
|                      | MITRE CWE (official CSV export from cwe.mitre.org)                        |
|                      | NIST SP 800-53 r5 (OSCAL JSON from `usnistgov/oscal-content`)             |
|                      | OWASP Top 10 (2021) markdown from the official `OWASP/Top10` repo         |

All cybersecurity KB sources are public, authoritative, and machine-readable. Each entity becomes a self-contained document (ID + title + body) before chunking, which lets the analysis pipeline trace retrieved context back to its source ID (e.g. `CWE-79`, `T1059.003`, `AC-6`). Set `rag.corpus_include` to a subset of `[attack, cwe, nist, owasp]` for per-source ablations.

## Other domains (cybersecurity)

A complete cybersecurity instantiation ships in `config_cyber.yaml` for follow-up work that is **not part of the paper**. It uses the same 2×(±RAG) framework with security-domain backbones and an authoritative KB (MITRE ATT&CK + CWE + NIST SP 800-53 r5 + OWASP Top 10).

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull hf.co/fdtn-ai/Foundation-Sec-8B-Q4_K_M-GGUF:Q4_K_M   # security-finetuned
ollama pull nomic-embed-text

python scripts/build_rag_index.py --config config_cyber.yaml
python scripts/run_experiment.py  --config config_cyber.yaml
python scripts/analyze_results.py --experiment <experiment.name from config_cyber.yaml>
```

Report wording is domain-aware and inferred from `question_id` prefixes (`medqa_*`, `cybermetric_*`, `secqa_*`), so generated `results/report.md` is not hardcoded to medical phrasing.

## Retrieval-only evaluation (debug / calibrate / tune)

`scripts/evaluate_retrieval.py` measures RAG quality **without running any LLM**, so you can iterate on chunking, indexing, and retrieval parameters in minutes instead of hours. It computes:

- **Intrinsic**: coverage rate, distance distribution (mean / p10 / median / p90), chunk-length stats, per-query latency, source-provenance mix (ATT&CK / CWE / NIST / OWASP).
- **Silver-label (no LLM)**: correct-option term recall vs. retrieved context, discrimination = overlap(correct) − mean overlap(distractors), and a retrieval-only MCQ accuracy upper bound (pick the option with maximum overlap to the retrieved chunks — what RAG alone could give a lookup-table "model").

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
- **`max_distance`** — read it off the top-1 distance histogram; pick the knee.
- **`top_k`** — pick the sweep row where retrieval-only accuracy plateaus; more chunks past that just add noise.
- **Chunk size / overlap** — rebuild the index with different `rag.chunk_size` / `rag.chunk_overlap`, rerun, compare `mean_discrimination` and retrieval-only accuracy.
- **Corpus ablation** — toggle `rag.corpus_include`, rebuild, and compare source-mix vs. accuracy to measure each KB's marginal contribution.

## Journal-revision experiments

These extend the core 2×2 study to defend the RAG null and establish generality
(see `paper/paper.tex`). New code is config-driven and reuses the same pipeline.

**Stronger statistics (no new runs needed)** — exact McNemar, Holm/BH
correction, odds ratios, TOST equivalence, and power, for one or more runs:

```bash
python scripts/statistical_analysis.py --experiment medical_mcq_comparison_full_0.1_3r --tost-margin 0.05
# merge two runs head-to-head (e.g. different embedders / scales / quantization):
python scripts/statistical_analysis.py --experiment run_nomic --experiment run_medcpt
```

**Cache retrieval once, reuse everywhere (faster, reproducible):**

```bash
# 1. Run retrieval once over all questions -> cache (exact chunks + candidate pool)
python scripts/precompute_retrieval.py --config config.yaml --pool-size 50

# 2. Run experiments from the cache (no re-embedding / re-querying ChromaDB)
python scripts/run_experiment.py --config config.yaml \
    --retrieval-cache data/medqa/retrieval_cache/medqa_textbooks.jsonl

# 3. Tune top_k / max_distance offline from the cache (instant, no LLM)
python scripts/tune_retrieval.py --config config.yaml --sweep \
    --top-k-grid 1,3,5,8 --max-distance-grid 0.2,0.25,0.3,0.35,0.4
# evaluate retrieval correctness on a sample, or inspect one question:
python scripts/tune_retrieval.py --config config.yaml --sample 200
python scripts/tune_retrieval.py --config config.yaml --question-id medqa_test_0
```

The cache stores, per question, the exact chunks the live retriever returned
(for faithful experiment reproduction, `--cache-mode final`) and a dense
candidate pool with distances (so the tuner can replay any `top_k`/`max_distance`
offline, and `run_experiment --cache-mode ranked` can re-rank at a new config).
The legacy online evaluator (`scripts/evaluate_retrieval.py`, re-embeds each run)
remains available.

**Defend the RAG null (Workstream A):**

```bash
# Oracle / upper-bound retrieval (adds answer / answer_soft / privileged_query setups)
python scripts/run_experiment.py --config config_oracle.yaml
python scripts/statistical_analysis.py --experiment medical_mcq_oracle_0.1_3r

# Stronger biomedical embedder (MedCPT) — separate index
pip install -e ".[hf]"
python scripts/build_rag_index.py --config config_medcpt.yaml
python scripts/run_experiment.py  --config config_medcpt.yaml

# Retrieval hyperparameter robustness sweep (no re-indexing; query-time only)
python scripts/rag_sweep.py --config config.yaml \
    --top-k 1,3,5 --max-distance 0.2,0.3,0.4 --query-mode question_plus_options
```

**Generalize across benchmarks, scale, domain (Workstream B):**

```bash
python scripts/run_experiment.py --config config_pubmedqa.yaml
python scripts/run_experiment.py --config config_medmcqa.yaml
python scripts/run_experiment.py --config config_mmlu.yaml
python scripts/run_experiment.py --config config_scale_27b.yaml   # set the 27B model ids first
python scripts/run_experiment.py --config config_cyber.yaml        # cross-domain replication
```

**De-confound (Workstream C):**

```bash
# n-gram leakage / contamination probe (corpus + optional external reference)
python scripts/contamination_probe.py --config config.yaml --n 13
# quantization spot-check on a subset, then compare to the 4-bit run
python scripts/run_experiment.py --config config_quant_check.yaml
python scripts/statistical_analysis.py --experiment medical_mcq_quantcheck_fp16 \
    --experiment medical_mcq_comparison_full_0.1_3r
```

**Error / conflict analysis (Workstream E):**

```bash
python scripts/error_analysis.py --experiment medical_mcq_comparison_full_0.1_3r \
    --questions data/medqa/questions.jsonl
```

New config knobs: `rag.embedding_backend` (`ollama`|`hf`|`medcpt`) and
`rag.oracle_modes` (e.g. `["answer","answer_soft","privileged_query"]`). New
dataset sources: `pubmedqa`, `medmcqa`, `mmlu`. Run the test suite with
`python tests/test_statistics.py` and `python tests/test_revision_modules.py`.

## Data and outputs

- Medical questions: `data/medqa/questions.jsonl`
- Cybersecurity questions: `data/cybermetric/questions.jsonl` (or `data/secqa/`)
- Chroma index (medical): `data/chroma_db/`
- Chroma index (cyber): `data/chroma_cyber/`
- Raw KB cache (cyber): `data/cyber_kb/{attack,cwe,nist,owasp}/`
- Run results: `results/<experiment_name>.jsonl`
- Analysis report: `results/report.md`
- Figures: `results/*.png`
- Paper sources: `paper/paper.tex`, `paper/references.bib`, `paper/figures/`

## Useful commands

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

## Reliability features

- Retry with exponential backoff for generation and embedding calls
- Per-call timeout support
- Invalid-answer retry loop when model output is not `A/B/C/D`
- One-time embedding-model ensure before RAG startup
- Retrieval is performed once per question and reused across all RAG model variants
- Per-sample exception handling so one failed call does not stop the experiment
- Optional explicit failure rows in results (`succeeded=false`, `error_message`)
- Correct-answer normalization to `A/B/C/D` with fail-fast validation

## Common pitfalls

- `analyze_results.py` does **not** read `config.yaml`; it uses `--experiment`.
- Results are append-only by experiment file. For a fresh run:
  - delete `results/<experiment_name>.jsonl`, or
  - change `experiment.name` in the config.
- The RAG index (`data/chroma_db/`) is not committed to git — run `build_rag_index.py` after cloning.

## Project entry points

Available after `pip install -e .`:

- `download-dataset`
- `build-rag-index`
- `run-experiment`
- `analyze-results`

## Citing

If you use this code or build on the paper, please cite:

```bibtex
@misc{buskila2026whatmakesitright,
  title  = {Domain Fine-Tuning vs. Retrieval-Augmented Generation for Medical
            Multiple-Choice Question Answering: A Controlled Comparison at the
            4B-Parameter Scale},
  author = {Buskila, Avi-ad Avraam},
  year   = {2026},
  note   = {Bar-Ilan University, Department of Information Science and Applied AI}
}
```
