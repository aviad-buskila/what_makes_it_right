# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 10
- **Repetitions per question**: 1
- **Total LLM calls**: 60

## Accuracy Results (Majority Vote)

| setup_name      |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------|-----------:|-----------:|-----------:|----------:|--------:|
| gemma3-4b       |     0.3000 |     0.1078 |     0.6032 |         3 |      10 |
| gemma3-4b+RAG   |     0.1000 |     0.0179 |     0.4042 |         1 |      10 |
| gpt-oss-20b     |     0.9000 |     0.5958 |     0.9821 |         9 |      10 |
| gpt-oss-20b+RAG |     0.8000 |     0.4902 |     0.9433 |         8 |      10 |
| medgemma-4b     |     0.6000 |     0.3127 |     0.8318 |         6 |      10 |
| medgemma-4b+RAG |     0.3000 |     0.1078 |     0.6032 |         3 |      10 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name      |   mean_accuracy |
|:----------------|----------------:|
| gemma3-4b       |          0.3000 |
| gemma3-4b+RAG   |          0.1000 |
| gpt-oss-20b     |          0.9000 |
| gpt-oss-20b+RAG |          0.8000 |
| medgemma-4b     |          0.6000 |
| medgemma-4b+RAG |          0.3000 |

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
| gpt-oss-20b     |               0.0000 |
| gpt-oss-20b+RAG |               0.0000 |
| medgemma-4b     |               0.0000 |
| medgemma-4b+RAG |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a         | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b       | gemma3-4b+RAG   |                2 |                0 |      0.5000 |    0.4795 | False         |
| gemma3-4b       | gpt-oss-20b     |                0 |                6 |      4.1667 |    0.0412 | True          |
| gemma3-4b       | gpt-oss-20b+RAG |                0 |                5 |      3.2000 |    0.0736 | False         |
| gemma3-4b       | medgemma-4b     |                0 |                3 |      1.3333 |    0.2482 | False         |
| gemma3-4b       | medgemma-4b+RAG |                2 |                2 |      0.2500 |    0.6171 | False         |
| gemma3-4b+RAG   | gpt-oss-20b     |                0 |                8 |      6.1250 |    0.0133 | True          |
| gemma3-4b+RAG   | gpt-oss-20b+RAG |                0 |                7 |      5.1429 |    0.0233 | True          |
| gemma3-4b+RAG   | medgemma-4b     |                0 |                5 |      3.2000 |    0.0736 | False         |
| gemma3-4b+RAG   | medgemma-4b+RAG |                0 |                2 |      0.5000 |    0.4795 | False         |
| gpt-oss-20b     | gpt-oss-20b+RAG |                1 |                0 |      0.0000 |    1.0000 | False         |
| gpt-oss-20b     | medgemma-4b     |                3 |                0 |      1.3333 |    0.2482 | False         |
| gpt-oss-20b     | medgemma-4b+RAG |                6 |                0 |      4.1667 |    0.0412 | True          |
| gpt-oss-20b+RAG | medgemma-4b     |                2 |                0 |      0.5000 |    0.4795 | False         |
| gpt-oss-20b+RAG | medgemma-4b+RAG |                5 |                0 |      3.2000 |    0.0736 | False         |
| medgemma-4b     | medgemma-4b+RAG |                3 |                0 |      1.3333 |    0.2482 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b   | gemma3-4b+RAG   |                2 |                0 |      0.5000 |    0.4795 | False         |
| gpt-oss-20b | gpt-oss-20b+RAG |                1 |                0 |      0.0000 |    1.0000 | False         |
| medgemma-4b | medgemma-4b+RAG |                3 |                0 |      1.3333 |    0.2482 | False         |