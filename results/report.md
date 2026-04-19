# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 30
- **Repetitions per question**: 3
- **Total LLM calls**: 540

## Accuracy Results (Majority Vote)

| setup_name            |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------------|-----------:|-----------:|-----------:|----------:|--------:|
| foundation-sec-8b     |     0.8667 |     0.7032 |     0.9469 |        26 |      30 |
| foundation-sec-8b+RAG |     0.7000 |     0.5212 |     0.8334 |        21 |      30 |
| llama3.1-8b           |     0.7000 |     0.5212 |     0.8334 |        21 |      30 |
| llama3.1-8b+RAG       |     0.5333 |     0.3614 |     0.6977 |        16 |      30 |
| qwen2.5-7b            |     0.9333 |     0.7868 |     0.9815 |        28 |      30 |
| qwen2.5-7b+RAG        |     0.9000 |     0.7438 |     0.9654 |        27 |      30 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name            |   mean_accuracy |
|:----------------------|----------------:|
| foundation-sec-8b     |          0.8111 |
| foundation-sec-8b+RAG |          0.6778 |
| llama3.1-8b           |          0.7000 |
| llama3.1-8b+RAG       |          0.5667 |
| qwen2.5-7b            |          0.9444 |
| qwen2.5-7b+RAG        |          0.9000 |

## Consistency (Agreement Across Repetitions)

| setup_name            |   consistency |
|:----------------------|--------------:|
| foundation-sec-8b     |        0.9000 |
| foundation-sec-8b+RAG |        0.8111 |
| llama3.1-8b           |        0.8667 |
| llama3.1-8b+RAG       |        0.8778 |
| qwen2.5-7b            |        0.9889 |
| qwen2.5-7b+RAG        |        0.9778 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name            |   parse_failure_rate |
|:----------------------|---------------------:|
| foundation-sec-8b     |               0.0111 |
| foundation-sec-8b+RAG |               0.0556 |
| llama3.1-8b           |               0.0000 |
| llama3.1-8b+RAG       |               0.0000 |
| qwen2.5-7b            |               0.0000 |
| qwen2.5-7b+RAG        |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a               | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b     | foundation-sec-8b+RAG |                5 |                0 |      3.2000 |    0.0736 | False         |
| foundation-sec-8b     | llama3.1-8b           |                6 |                1 |      2.2857 |    0.1306 | False         |
| foundation-sec-8b     | llama3.1-8b+RAG       |               11 |                1 |      6.7500 |    0.0094 | True          |
| foundation-sec-8b     | qwen2.5-7b            |                1 |                3 |      0.2500 |    0.6171 | False         |
| foundation-sec-8b     | qwen2.5-7b+RAG        |                3 |                4 |      0.0000 |    1.0000 | False         |
| foundation-sec-8b+RAG | llama3.1-8b           |                3 |                3 |      0.1667 |    0.6831 | False         |
| foundation-sec-8b+RAG | llama3.1-8b+RAG       |                7 |                2 |      1.7778 |    0.1824 | False         |
| foundation-sec-8b+RAG | qwen2.5-7b            |                0 |                7 |      5.1429 |    0.0233 | True          |
| foundation-sec-8b+RAG | qwen2.5-7b+RAG        |                1 |                7 |      3.1250 |    0.0771 | False         |
| llama3.1-8b           | llama3.1-8b+RAG       |                6 |                1 |      2.2857 |    0.1306 | False         |
| llama3.1-8b           | qwen2.5-7b            |                0 |                7 |      5.1429 |    0.0233 | True          |
| llama3.1-8b           | qwen2.5-7b+RAG        |                2 |                8 |      2.5000 |    0.1138 | False         |
| llama3.1-8b+RAG       | qwen2.5-7b            |                0 |               12 |     10.0833 |    0.0015 | True          |
| llama3.1-8b+RAG       | qwen2.5-7b+RAG        |                0 |               11 |      9.0909 |    0.0026 | True          |
| qwen2.5-7b            | qwen2.5-7b+RAG        |                2 |                1 |      0.0000 |    1.0000 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a           | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b | foundation-sec-8b+RAG |                5 |                0 |      3.2000 |    0.0736 | False         |
| llama3.1-8b       | llama3.1-8b+RAG       |                6 |                1 |      2.2857 |    0.1306 | False         |
| qwen2.5-7b        | qwen2.5-7b+RAG        |                2 |                1 |      0.0000 |    1.0000 | False         |