# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical multiple-choice QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Domain (inferred)**: medical
- **Questions**: 20
- **Repetitions per question**: 1
- **Total LLM calls**: 120

## Accuracy Results (Majority Vote)

| setup_name      |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------|-----------:|-----------:|-----------:|----------:|--------:|
| gemma3-4b       |     0.5000 |     0.2993 |     0.7007 |        10 |      20 |
| gemma3-4b+RAG   |     0.5000 |     0.2993 |     0.7007 |        10 |      20 |
| gpt-oss-20b     |     0.9000 |     0.6990 |     0.9721 |        18 |      20 |
| gpt-oss-20b+RAG |     0.5000 |     0.2993 |     0.7007 |        10 |      20 |
| medgemma-4b     |     0.5500 |     0.3421 |     0.7418 |        11 |      20 |
| medgemma-4b+RAG |     0.4500 |     0.2582 |     0.6579 |         9 |      20 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name      |   mean_accuracy |
|:----------------|----------------:|
| gemma3-4b       |          0.5000 |
| gemma3-4b+RAG   |          0.5000 |
| gpt-oss-20b     |          0.9000 |
| gpt-oss-20b+RAG |          0.5000 |
| medgemma-4b     |          0.5500 |
| medgemma-4b+RAG |          0.4500 |

## Consistency (Agreement Across Repetitions)

| setup_name      |   consistency |
|:----------------|--------------:|
| gemma3-4b       |        1.0000 |
| gemma3-4b+RAG   |        1.0000 |
| gpt-oss-20b     |        1.0000 |
| gpt-oss-20b+RAG |        1.0000 |
| medgemma-4b     |        1.0000 |
| medgemma-4b+RAG |        1.0000 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name      |   parse_failure_rate |
|:----------------|---------------------:|
| gemma3-4b       |               0.0000 |
| gemma3-4b+RAG   |               0.0000 |
| gpt-oss-20b     |               0.0500 |
| gpt-oss-20b+RAG |               0.3500 |
| medgemma-4b     |               0.0000 |
| medgemma-4b+RAG |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a         | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b       | gemma3-4b+RAG   |                2 |                2 |      0.2500 |    0.6171 | False         |
| gemma3-4b       | gpt-oss-20b     |                0 |                8 |      6.1250 |    0.0133 | True          |
| gemma3-4b       | gpt-oss-20b+RAG |                2 |                2 |      0.2500 |    0.6171 | False         |
| gemma3-4b       | medgemma-4b     |                2 |                3 |      0.0000 |    1.0000 | False         |
| gemma3-4b       | medgemma-4b+RAG |                3 |                2 |      0.0000 |    1.0000 | False         |
| gemma3-4b+RAG   | gpt-oss-20b     |                0 |                8 |      6.1250 |    0.0133 | True          |
| gemma3-4b+RAG   | gpt-oss-20b+RAG |                3 |                3 |      0.1667 |    0.6831 | False         |
| gemma3-4b+RAG   | medgemma-4b     |                3 |                4 |      0.0000 |    1.0000 | False         |
| gemma3-4b+RAG   | medgemma-4b+RAG |                3 |                2 |      0.0000 |    1.0000 | False         |
| gpt-oss-20b     | gpt-oss-20b+RAG |                8 |                0 |      6.1250 |    0.0133 | True          |
| gpt-oss-20b     | medgemma-4b     |                7 |                0 |      5.1429 |    0.0233 | True          |
| gpt-oss-20b     | medgemma-4b+RAG |                9 |                0 |      7.1111 |    0.0077 | True          |
| gpt-oss-20b+RAG | medgemma-4b     |                2 |                3 |      0.0000 |    1.0000 | False         |
| gpt-oss-20b+RAG | medgemma-4b+RAG |                2 |                1 |      0.0000 |    1.0000 | False         |
| medgemma-4b     | medgemma-4b+RAG |                3 |                1 |      0.2500 |    0.6171 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b   | gemma3-4b+RAG   |                2 |                2 |      0.2500 |    0.6171 | False         |
| gpt-oss-20b | gpt-oss-20b+RAG |                8 |                0 |      6.1250 |    0.0133 | True          |
| medgemma-4b | medgemma-4b+RAG |                3 |                1 |      0.2500 |    0.6171 | False         |