# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 100
- **Repetitions per question**: 3
- **Total LLM calls**: 1200

## Accuracy Results (Majority Vote)

| setup_name            |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------------|-----------:|-----------:|-----------:|----------:|--------:|
| foundation-sec-8b     |     0.7800 |     0.6893 |     0.8500 |        78 |     100 |
| foundation-sec-8b+RAG |     0.7900 |     0.7002 |     0.8583 |        79 |     100 |
| qwen2.5-7b            |     0.9200 |     0.8500 |     0.9589 |        92 |     100 |
| qwen2.5-7b+RAG        |     0.8600 |     0.7786 |     0.9147 |        86 |     100 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name            |   mean_accuracy |
|:----------------------|----------------:|
| foundation-sec-8b     |          0.7867 |
| foundation-sec-8b+RAG |          0.7233 |
| qwen2.5-7b            |          0.9167 |
| qwen2.5-7b+RAG        |          0.8600 |

## Consistency (Agreement Across Repetitions)

| setup_name            |   consistency |
|:----------------------|--------------:|
| foundation-sec-8b     |        0.9533 |
| foundation-sec-8b+RAG |        0.9200 |
| qwen2.5-7b            |        0.9967 |
| qwen2.5-7b+RAG        |        0.9933 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name            |   parse_failure_rate |
|:----------------------|---------------------:|
| foundation-sec-8b     |               0.0000 |
| foundation-sec-8b+RAG |               0.0600 |
| qwen2.5-7b            |               0.0000 |
| qwen2.5-7b+RAG        |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a               | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b     | foundation-sec-8b+RAG |                5 |                6 |      0.0000 |    1.0000 | False         |
| foundation-sec-8b     | qwen2.5-7b            |                4 |               18 |      7.6818 |    0.0056 | True          |
| foundation-sec-8b     | qwen2.5-7b+RAG        |                8 |               16 |      2.0417 |    0.1530 | False         |
| foundation-sec-8b+RAG | qwen2.5-7b            |                4 |               17 |      6.8571 |    0.0088 | True          |
| foundation-sec-8b+RAG | qwen2.5-7b+RAG        |                8 |               15 |      1.5652 |    0.2109 | False         |
| qwen2.5-7b            | qwen2.5-7b+RAG        |                7 |                1 |      3.1250 |    0.0771 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a           | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b | foundation-sec-8b+RAG |                5 |                6 |      0.0000 |    1.0000 | False         |
| qwen2.5-7b        | qwen2.5-7b+RAG        |                7 |                1 |      3.1250 |    0.0771 | False         |