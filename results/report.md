# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for cybersecurity multiple-choice QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Domain (inferred)**: cybersecurity
- **Questions**: 100
- **Repetitions per question**: 3
- **Total LLM calls**: 1800

## Accuracy Results (Majority Vote)

| setup_name            |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------------|-----------:|-----------:|-----------:|----------:|--------:|
| foundation-sec-8b     |     0.6000 |     0.5020 |     0.6906 |        60 |     100 |
| foundation-sec-8b+RAG |     0.5800 |     0.4821 |     0.6720 |        58 |     100 |
| llama3.1-8b           |     0.7900 |     0.7002 |     0.8583 |        79 |     100 |
| llama3.1-8b+RAG       |     0.8400 |     0.7558 |     0.8990 |        84 |     100 |
| qwen2.5-7b            |     0.8700 |     0.7902 |     0.9224 |        87 |     100 |
| qwen2.5-7b+RAG        |     0.8500 |     0.7672 |     0.9069 |        85 |     100 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name            |   mean_accuracy |
|:----------------------|----------------:|
| foundation-sec-8b     |          0.5867 |
| foundation-sec-8b+RAG |          0.5433 |
| llama3.1-8b           |          0.8000 |
| llama3.1-8b+RAG       |          0.8333 |
| qwen2.5-7b            |          0.8733 |
| qwen2.5-7b+RAG        |          0.8500 |

## Consistency (Agreement Across Repetitions)

| setup_name            |   consistency |
|:----------------------|--------------:|
| foundation-sec-8b     |        0.9133 |
| foundation-sec-8b+RAG |        0.8467 |
| llama3.1-8b           |        0.9800 |
| llama3.1-8b+RAG       |        0.9833 |
| qwen2.5-7b            |        0.9967 |
| qwen2.5-7b+RAG        |        0.9933 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name            |   parse_failure_rate |
|:----------------------|---------------------:|
| foundation-sec-8b     |               0.0267 |
| foundation-sec-8b+RAG |               0.1067 |
| llama3.1-8b           |               0.0000 |
| llama3.1-8b+RAG       |               0.0000 |
| qwen2.5-7b            |               0.0000 |
| qwen2.5-7b+RAG        |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a               | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b     | foundation-sec-8b+RAG |                6 |                4 |      0.1000 |    0.7518 | False         |
| foundation-sec-8b     | llama3.1-8b           |                4 |               23 |     12.0000 |    0.0005 | True          |
| foundation-sec-8b     | llama3.1-8b+RAG       |                3 |               27 |     17.6333 |    0.0000 | True          |
| foundation-sec-8b     | qwen2.5-7b            |                3 |               30 |     20.4848 |    0.0000 | True          |
| foundation-sec-8b     | qwen2.5-7b+RAG        |                4 |               29 |     17.4545 |    0.0000 | True          |
| foundation-sec-8b+RAG | llama3.1-8b           |                6 |               27 |     12.1212 |    0.0005 | True          |
| foundation-sec-8b+RAG | llama3.1-8b+RAG       |                4 |               30 |     18.3824 |    0.0000 | True          |
| foundation-sec-8b+RAG | qwen2.5-7b            |                5 |               34 |     20.1026 |    0.0000 | True          |
| foundation-sec-8b+RAG | qwen2.5-7b+RAG        |                6 |               33 |     17.3333 |    0.0000 | True          |
| llama3.1-8b           | llama3.1-8b+RAG       |                0 |                5 |      3.2000 |    0.0736 | False         |
| llama3.1-8b           | qwen2.5-7b            |                4 |               12 |      3.0625 |    0.0801 | False         |
| llama3.1-8b           | qwen2.5-7b+RAG        |                7 |               13 |      1.2500 |    0.2636 | False         |
| llama3.1-8b+RAG       | qwen2.5-7b            |                5 |                8 |      0.3077 |    0.5791 | False         |
| llama3.1-8b+RAG       | qwen2.5-7b+RAG        |                7 |                8 |      0.0000 |    1.0000 | False         |
| qwen2.5-7b            | qwen2.5-7b+RAG        |                4 |                2 |      0.1667 |    0.6831 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a           | setup_b               |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------------|:----------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| foundation-sec-8b | foundation-sec-8b+RAG |                6 |                4 |      0.1000 |    0.7518 | False         |
| llama3.1-8b       | llama3.1-8b+RAG       |                0 |                5 |      3.2000 |    0.0736 | False         |
| qwen2.5-7b        | qwen2.5-7b+RAG        |                4 |                2 |      0.1667 |    0.6831 | False         |