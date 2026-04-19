# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 5
- **Repetitions per question**: 3
- **Total LLM calls**: 90

## Accuracy Results (Majority Vote)

| setup_name            |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------------|-----------:|-----------:|-----------:|----------:|--------:|
| foundation-sec-8b     |     0.8000 |     0.3755 |     0.9638 |         4 |       5 |
| foundation-sec-8b+RAG |     0.6000 |     0.2307 |     0.8824 |         3 |       5 |
| llama3.1-8b           |     1.0000 |     0.5655 |     1.0000 |         5 |       5 |
| llama3.1-8b+RAG       |     0.8000 |     0.3755 |     0.9638 |         4 |       5 |
| qwen2.5-7b            |     1.0000 |     0.5655 |     1.0000 |         5 |       5 |
| qwen2.5-7b+RAG        |     0.8000 |     0.3755 |     0.9638 |         4 |       5 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name            |   mean_accuracy |
|:----------------------|----------------:|
| foundation-sec-8b     |          0.8000 |
| foundation-sec-8b+RAG |          0.6000 |
| llama3.1-8b           |          1.0000 |
| llama3.1-8b+RAG       |          0.6667 |
| qwen2.5-7b            |          1.0000 |
| qwen2.5-7b+RAG        |          0.8000 |

## Consistency (Agreement Across Repetitions)

| setup_name            |   consistency |
|:----------------------|--------------:|
| foundation-sec-8b     |        1.0000 |
| foundation-sec-8b+RAG |        0.9333 |
| llama3.1-8b           |        1.0000 |
| llama3.1-8b+RAG       |        0.8667 |
| qwen2.5-7b            |        1.0000 |
| qwen2.5-7b+RAG        |        1.0000 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name            |   parse_failure_rate |
|:----------------------|---------------------:|
| foundation-sec-8b     |               0.0000 |
| foundation-sec-8b+RAG |               0.0667 |
| llama3.1-8b           |               0.0000 |
| llama3.1-8b+RAG       |               0.0000 |
| qwen2.5-7b            |               0.0000 |
| qwen2.5-7b+RAG        |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a               | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b     | foundation-sec-8b+RAG |                1 |                0 |      0.0000 |    1.0000 | False         |
| foundation-sec-8b     | llama3.1-8b           |                0 |                1 |      0.0000 |    1.0000 | False         |
| foundation-sec-8b     | llama3.1-8b+RAG       |                1 |                1 |      0.5000 |    0.4795 | False         |
| foundation-sec-8b     | qwen2.5-7b            |                0 |                1 |      0.0000 |    1.0000 | False         |
| foundation-sec-8b     | qwen2.5-7b+RAG        |                0 |                0 |      0.0000 |    1.0000 | False         |
| foundation-sec-8b+RAG | llama3.1-8b           |                0 |                2 |      0.5000 |    0.4795 | False         |
| foundation-sec-8b+RAG | llama3.1-8b+RAG       |                1 |                2 |      0.0000 |    1.0000 | False         |
| foundation-sec-8b+RAG | qwen2.5-7b            |                0 |                2 |      0.5000 |    0.4795 | False         |
| foundation-sec-8b+RAG | qwen2.5-7b+RAG        |                0 |                1 |      0.0000 |    1.0000 | False         |
| llama3.1-8b           | llama3.1-8b+RAG       |                1 |                0 |      0.0000 |    1.0000 | False         |
| llama3.1-8b           | qwen2.5-7b            |                0 |                0 |      0.0000 |    1.0000 | False         |
| llama3.1-8b           | qwen2.5-7b+RAG        |                1 |                0 |      0.0000 |    1.0000 | False         |
| llama3.1-8b+RAG       | qwen2.5-7b            |                0 |                1 |      0.0000 |    1.0000 | False         |
| llama3.1-8b+RAG       | qwen2.5-7b+RAG        |                1 |                1 |      0.5000 |    0.4795 | False         |
| qwen2.5-7b            | qwen2.5-7b+RAG        |                1 |                0 |      0.0000 |    1.0000 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a           | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b | foundation-sec-8b+RAG |                1 |                0 |      0.0000 |    1.0000 | False         |
| llama3.1-8b       | llama3.1-8b+RAG       |                1 |                0 |      0.0000 |    1.0000 | False         |
| qwen2.5-7b        | qwen2.5-7b+RAG        |                1 |                0 |      0.0000 |    1.0000 | False         |