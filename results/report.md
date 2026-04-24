# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical multiple-choice QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Domain (inferred)**: medical
- **Questions**: 1273
- **Repetitions per question**: 3
- **Total LLM calls**: 15276

## Accuracy Results (Majority Vote)

| setup_name           |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:---------------------|-----------:|-----------:|-----------:|----------:|--------:|
| gemma3-4b+RAG@t0.1   |     0.4729 |     0.4456 |     0.5004 |       602 |    1273 |
| gemma3-4b@t0.1       |     0.4643 |     0.4370 |     0.4917 |       591 |    1273 |
| medgemma-4b+RAG@t0.1 |     0.5137 |     0.4863 |     0.5411 |       654 |    1273 |
| medgemma-4b@t0.1     |     0.5326 |     0.5051 |     0.5599 |       678 |    1273 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name           |   mean_accuracy |
|:---------------------|----------------:|
| gemma3-4b+RAG@t0.1   |          0.4729 |
| gemma3-4b@t0.1       |          0.4637 |
| medgemma-4b+RAG@t0.1 |          0.5109 |
| medgemma-4b@t0.1     |          0.5323 |

## Consistency (Agreement Across Repetitions)

| setup_name           |   consistency |
|:---------------------|--------------:|
| gemma3-4b+RAG@t0.1   |        0.9992 |
| gemma3-4b@t0.1       |        0.9974 |
| medgemma-4b+RAG@t0.1 |        0.9919 |
| medgemma-4b@t0.1     |        0.9921 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name           |   parse_failure_rate |
|:---------------------|---------------------:|
| gemma3-4b+RAG@t0.1   |               0.0000 |
| gemma3-4b@t0.1       |               0.0000 |
| medgemma-4b+RAG@t0.1 |               0.0039 |
| medgemma-4b@t0.1     |               0.0047 |

## Retrieval Diagnostics (RAG Rows)

- **RAG calls with non-empty context**: 96.35%
- **Avg retrieved chunks per RAG call**: 2.89

## Pairwise Statistical Tests (McNemar's)

| setup_a              | setup_b              |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:---------------------|:---------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b+RAG@t0.1   | gemma3-4b@t0.1       |              156 |              145 |      0.3322 |    0.5644 | False         |
| gemma3-4b+RAG@t0.1   | medgemma-4b+RAG@t0.1 |              139 |              191 |      7.8818 |    0.0050 | True          |
| gemma3-4b+RAG@t0.1   | medgemma-4b@t0.1     |              179 |              255 |     12.9608 |    0.0003 | True          |
| gemma3-4b@t0.1       | medgemma-4b+RAG@t0.1 |              176 |              239 |      9.2627 |    0.0023 | True          |
| gemma3-4b@t0.1       | medgemma-4b@t0.1     |              144 |              231 |     19.7227 |    0.0000 | True          |
| medgemma-4b+RAG@t0.1 | medgemma-4b@t0.1     |              124 |              148 |      1.9449 |    0.1631 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a          | setup_b              |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:-----------------|:---------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b@t0.1   | gemma3-4b+RAG@t0.1   |              145 |              156 |      0.3322 |    0.5644 | False         |
| medgemma-4b@t0.1 | medgemma-4b+RAG@t0.1 |              148 |              124 |      1.9449 |    0.1631 | False         |