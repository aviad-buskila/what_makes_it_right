# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 20
- **Repetitions per question**: 1
- **Total LLM calls**: 40

## Accuracy Results (Majority Vote)

| setup_name      |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------|-----------:|-----------:|-----------:|----------:|--------:|
| medgemma-4b     |     0.5500 |     0.3421 |     0.7418 |        11 |      20 |
| medgemma-4b+RAG |     0.4500 |     0.2582 |     0.6579 |         9 |      20 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name      |   mean_accuracy |
|:----------------|----------------:|
| medgemma-4b     |          0.5500 |
| medgemma-4b+RAG |          0.4500 |

## Consistency (Agreement Across Repetitions)

| setup_name      |   consistency |
|:----------------|--------------:|
| medgemma-4b     |        1.0000 |
| medgemma-4b+RAG |        1.0000 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name      |   parse_failure_rate |
|:----------------|---------------------:|
| medgemma-4b     |               0.0000 |
| medgemma-4b+RAG |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| medgemma-4b | medgemma-4b+RAG |                4 |                2 |      0.1667 |    0.6831 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| medgemma-4b | medgemma-4b+RAG |                4 |                2 |      0.1667 |    0.6831 | False         |