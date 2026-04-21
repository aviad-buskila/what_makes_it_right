# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical multiple-choice QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Domain (inferred)**: medical
- **Questions**: 300
- **Repetitions per question**: 1
- **Total LLM calls**: 1200

## Accuracy Results (Majority Vote)

| setup_name      |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------|-----------:|-----------:|-----------:|----------:|--------:|
| gemma3-4b       |     0.4967 |     0.4405 |     0.5529 |       149 |     300 |
| gemma3-4b+RAG   |     0.5067 |     0.4504 |     0.5628 |       152 |     300 |
| medgemma-4b     |     0.5500 |     0.4934 |     0.6053 |       165 |     300 |
| medgemma-4b+RAG |     0.5400 |     0.4835 |     0.5955 |       162 |     300 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name      |   mean_accuracy |
|:----------------|----------------:|
| gemma3-4b       |          0.4967 |
| gemma3-4b+RAG   |          0.5067 |
| medgemma-4b     |          0.5500 |
| medgemma-4b+RAG |          0.5400 |

## Consistency (Agreement Across Repetitions)

| setup_name      |   consistency |
|:----------------|--------------:|
| gemma3-4b       |        1.0000 |
| gemma3-4b+RAG   |        1.0000 |
| medgemma-4b     |        1.0000 |
| medgemma-4b+RAG |        1.0000 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name      |   parse_failure_rate |
|:----------------|---------------------:|
| gemma3-4b       |               0.0000 |
| gemma3-4b+RAG   |               0.0000 |
| medgemma-4b     |               0.0033 |
| medgemma-4b+RAG |               0.0033 |

## Pairwise Statistical Tests (McNemar's)

| setup_a       | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:--------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b     | gemma3-4b+RAG   |               10 |               13 |      0.1739 |    0.6767 | False         |
| gemma3-4b     | medgemma-4b     |               34 |               50 |      2.6786 |    0.1017 | False         |
| gemma3-4b     | medgemma-4b+RAG |               36 |               49 |      1.6941 |    0.1931 | False         |
| gemma3-4b+RAG | medgemma-4b     |               36 |               49 |      1.6941 |    0.1931 | False         |
| gemma3-4b+RAG | medgemma-4b+RAG |               33 |               43 |      1.0658 |    0.3019 | False         |
| medgemma-4b   | medgemma-4b+RAG |               15 |               12 |      0.1481 |    0.7003 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b   | gemma3-4b+RAG   |               10 |               13 |      0.1739 |    0.6767 | False         |
| medgemma-4b | medgemma-4b+RAG |               15 |               12 |      0.1481 |    0.7003 | False         |