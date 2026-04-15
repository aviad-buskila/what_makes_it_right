# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 3
- **Repetitions per question**: 2
- **Total LLM calls**: 36

## Accuracy Results (Majority Vote)

| setup_name      |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------|-----------:|-----------:|-----------:|----------:|--------:|
| gemma3-4b       |     0.0000 |     0.0000 |     0.5615 |         0 |       3 |
| gemma3-4b+RAG   |     0.0000 |     0.0000 |     0.5615 |         0 |       3 |
| gpt-oss-20b     |     1.0000 |     0.4385 |     1.0000 |         3 |       3 |
| gpt-oss-20b+RAG |     1.0000 |     0.4385 |     1.0000 |         3 |       3 |
| medgemma-4b     |     0.6667 |     0.2077 |     0.9385 |         2 |       3 |
| medgemma-4b+RAG |     0.3333 |     0.0615 |     0.7923 |         1 |       3 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name      |   mean_accuracy |
|:----------------|----------------:|
| gemma3-4b       |          0.0000 |
| gemma3-4b+RAG   |          0.0000 |
| gpt-oss-20b     |          0.8333 |
| gpt-oss-20b+RAG |          0.8333 |
| medgemma-4b     |          0.6667 |
| medgemma-4b+RAG |          0.3333 |

## Consistency (Agreement Across Repetitions)

| setup_name      |   consistency |
|:----------------|--------------:|
| gemma3-4b       |        1.0000 |
| gemma3-4b+RAG   |        1.0000 |
| gpt-oss-20b     |        0.8333 |
| gpt-oss-20b+RAG |        0.8333 |
| medgemma-4b     |        1.0000 |
| medgemma-4b+RAG |        1.0000 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name      |   parse_failure_rate |
|:----------------|---------------------:|
| gemma3-4b       |               0.0000 |
| gemma3-4b+RAG   |               0.0000 |
| gpt-oss-20b     |               0.0000 |
| gpt-oss-20b+RAG |               0.0000 |
| medgemma-4b     |               0.0000 |
| medgemma-4b+RAG |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a         | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b       | gemma3-4b+RAG   |                0 |                0 |      0.0000 |    1.0000 | False         |
| gemma3-4b       | gpt-oss-20b     |                0 |                3 |      1.3333 |    0.2482 | False         |
| gemma3-4b       | gpt-oss-20b+RAG |                0 |                3 |      1.3333 |    0.2482 | False         |
| gemma3-4b       | medgemma-4b     |                0 |                2 |      0.5000 |    0.4795 | False         |
| gemma3-4b       | medgemma-4b+RAG |                0 |                1 |      0.0000 |    1.0000 | False         |
| gemma3-4b+RAG   | gpt-oss-20b     |                0 |                3 |      1.3333 |    0.2482 | False         |
| gemma3-4b+RAG   | gpt-oss-20b+RAG |                0 |                3 |      1.3333 |    0.2482 | False         |
| gemma3-4b+RAG   | medgemma-4b     |                0 |                2 |      0.5000 |    0.4795 | False         |
| gemma3-4b+RAG   | medgemma-4b+RAG |                0 |                1 |      0.0000 |    1.0000 | False         |
| gpt-oss-20b     | gpt-oss-20b+RAG |                0 |                0 |      0.0000 |    1.0000 | False         |
| gpt-oss-20b     | medgemma-4b     |                1 |                0 |      0.0000 |    1.0000 | False         |
| gpt-oss-20b     | medgemma-4b+RAG |                2 |                0 |      0.5000 |    0.4795 | False         |
| gpt-oss-20b+RAG | medgemma-4b     |                1 |                0 |      0.0000 |    1.0000 | False         |
| gpt-oss-20b+RAG | medgemma-4b+RAG |                2 |                0 |      0.5000 |    0.4795 | False         |
| medgemma-4b     | medgemma-4b+RAG |                1 |                0 |      0.0000 |    1.0000 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b   | gemma3-4b+RAG   |                0 |                0 |      0.0000 |    1.0000 | False         |
| gpt-oss-20b | gpt-oss-20b+RAG |                0 |                0 |      0.0000 |    1.0000 | False         |
| medgemma-4b | medgemma-4b+RAG |                1 |                0 |      0.0000 |    1.0000 | False         |