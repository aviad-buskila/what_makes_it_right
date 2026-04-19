# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 1
- **Repetitions per question**: 2
- **Total LLM calls**: 5

## Accuracy Results (Majority Vote)

| setup_name      |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------|-----------:|-----------:|-----------:|----------:|--------:|
| llama3.1-8b     |     1.0000 |     0.2065 |     1.0000 |         1 |       1 |
| llama3.1-8b+RAG |     1.0000 |     0.2065 |     1.0000 |         1 |       1 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name      |   mean_accuracy |
|:----------------|----------------:|
| llama3.1-8b     |          1.0000 |
| llama3.1-8b+RAG |          1.0000 |

## Consistency (Agreement Across Repetitions)

| setup_name      |   consistency |
|:----------------|--------------:|
| llama3.1-8b     |        1.0000 |
| llama3.1-8b+RAG |        1.0000 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name      |   parse_failure_rate |
|:----------------|---------------------:|
| llama3.1-8b     |               0.0000 |
| llama3.1-8b+RAG |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| llama3.1-8b | llama3.1-8b+RAG |                0 |                0 |      0.0000 |    1.0000 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| llama3.1-8b | llama3.1-8b+RAG |                0 |                0 |      0.0000 |    1.0000 | False         |