# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 10
- **Repetitions per question**: 1
- **Total LLM calls**: 60

## Accuracy Results (Majority Vote)

| setup_name            |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------------|-----------:|-----------:|-----------:|----------:|--------:|
| foundation-sec-8b     |     0.5000 |     0.2366 |     0.7634 |         5 |      10 |
| foundation-sec-8b+RAG |     0.4000 |     0.1682 |     0.6873 |         4 |      10 |
| llama3.1-8b           |     0.8000 |     0.4902 |     0.9433 |         8 |      10 |
| llama3.1-8b+RAG       |     0.9000 |     0.5958 |     0.9821 |         9 |      10 |
| qwen2.5-7b            |     0.9000 |     0.5958 |     0.9821 |         9 |      10 |
| qwen2.5-7b+RAG        |     0.9000 |     0.5958 |     0.9821 |         9 |      10 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name            |   mean_accuracy |
|:----------------------|----------------:|
| foundation-sec-8b     |          0.5000 |
| foundation-sec-8b+RAG |          0.4000 |
| llama3.1-8b           |          0.8000 |
| llama3.1-8b+RAG       |          0.9000 |
| qwen2.5-7b            |          0.9000 |
| qwen2.5-7b+RAG        |          0.9000 |

## Consistency (Agreement Across Repetitions)

| setup_name            |   consistency |
|:----------------------|--------------:|
| foundation-sec-8b     |        1.0000 |
| foundation-sec-8b+RAG |        1.0000 |
| llama3.1-8b           |        1.0000 |
| llama3.1-8b+RAG       |        1.0000 |
| qwen2.5-7b            |        1.0000 |
| qwen2.5-7b+RAG        |        1.0000 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name            |   parse_failure_rate |
|:----------------------|---------------------:|
| foundation-sec-8b     |               0.2000 |
| foundation-sec-8b+RAG |               0.1000 |
| llama3.1-8b           |               0.0000 |
| llama3.1-8b+RAG       |               0.0000 |
| qwen2.5-7b            |               0.0000 |
| qwen2.5-7b+RAG        |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a               | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b     | foundation-sec-8b+RAG |                2 |                1 |      0.0000 |    1.0000 | False         |
| foundation-sec-8b     | llama3.1-8b           |                0 |                3 |      1.3333 |    0.2482 | False         |
| foundation-sec-8b     | llama3.1-8b+RAG       |                0 |                4 |      2.2500 |    0.1336 | False         |
| foundation-sec-8b     | qwen2.5-7b            |                0 |                4 |      2.2500 |    0.1336 | False         |
| foundation-sec-8b     | qwen2.5-7b+RAG        |                0 |                4 |      2.2500 |    0.1336 | False         |
| foundation-sec-8b+RAG | llama3.1-8b           |                0 |                4 |      2.2500 |    0.1336 | False         |
| foundation-sec-8b+RAG | llama3.1-8b+RAG       |                0 |                5 |      3.2000 |    0.0736 | False         |
| foundation-sec-8b+RAG | qwen2.5-7b            |                0 |                5 |      3.2000 |    0.0736 | False         |
| foundation-sec-8b+RAG | qwen2.5-7b+RAG        |                0 |                5 |      3.2000 |    0.0736 | False         |
| llama3.1-8b           | llama3.1-8b+RAG       |                0 |                1 |      0.0000 |    1.0000 | False         |
| llama3.1-8b           | qwen2.5-7b            |                1 |                2 |      0.0000 |    1.0000 | False         |
| llama3.1-8b           | qwen2.5-7b+RAG        |                1 |                2 |      0.0000 |    1.0000 | False         |
| llama3.1-8b+RAG       | qwen2.5-7b            |                1 |                1 |      0.5000 |    0.4795 | False         |
| llama3.1-8b+RAG       | qwen2.5-7b+RAG        |                1 |                1 |      0.5000 |    0.4795 | False         |
| qwen2.5-7b            | qwen2.5-7b+RAG        |                0 |                0 |      0.0000 |    1.0000 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a           | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b | foundation-sec-8b+RAG |                2 |                1 |      0.0000 |    1.0000 | False         |
| llama3.1-8b       | llama3.1-8b+RAG       |                0 |                1 |      0.0000 |    1.0000 | False         |
| qwen2.5-7b        | qwen2.5-7b+RAG        |                0 |                0 |      0.0000 |    1.0000 | False         |