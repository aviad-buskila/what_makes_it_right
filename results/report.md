# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 30
- **Repetitions per question**: 3
- **Total LLM calls**: 180

## Accuracy Results (Majority Vote)

| setup_name            |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------------|-----------:|-----------:|-----------:|----------:|--------:|
| foundation-sec-8b     |     0.7667 |     0.5907 |     0.8821 |        23 |      30 |
| foundation-sec-8b+RAG |     0.7000 |     0.5212 |     0.8334 |        21 |      30 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name            |   mean_accuracy |
|:----------------------|----------------:|
| foundation-sec-8b     |          0.7667 |
| foundation-sec-8b+RAG |          0.6556 |

## Consistency (Agreement Across Repetitions)

| setup_name            |   consistency |
|:----------------------|--------------:|
| foundation-sec-8b     |        0.9556 |
| foundation-sec-8b+RAG |        0.9111 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name            |   parse_failure_rate |
|:----------------------|---------------------:|
| foundation-sec-8b     |               0.0000 |
| foundation-sec-8b+RAG |               0.1778 |

## Pairwise Statistical Tests (McNemar's)

| setup_a           | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b | foundation-sec-8b+RAG |                3 |                1 |      0.2500 |    0.6171 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a           | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b | foundation-sec-8b+RAG |                3 |                1 |      0.2500 |    0.6171 | False         |