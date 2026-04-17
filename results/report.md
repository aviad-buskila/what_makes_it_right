# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 100
- **Repetitions per question**: 3
- **Total LLM calls**: 1800

## Accuracy Results (Majority Vote)

| setup_name      |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------|-----------:|-----------:|-----------:|----------:|--------:|
| gemma3-4b       |     0.4900 |     0.3942 |     0.5865 |        49 |     100 |
| gemma3-4b+RAG   |     0.4200 |     0.3280 |     0.5179 |        42 |     100 |
| gpt-oss-20b     |     0.8900 |     0.8137 |     0.9375 |        89 |     100 |
| gpt-oss-20b+RAG |     0.9000 |     0.8256 |     0.9448 |        90 |     100 |
| medgemma-4b     |     0.5900 |     0.4920 |     0.6813 |        59 |     100 |
| medgemma-4b+RAG |     0.5600 |     0.4623 |     0.6533 |        56 |     100 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name      |   mean_accuracy |
|:----------------|----------------:|
| gemma3-4b       |          0.4933 |
| gemma3-4b+RAG   |          0.4200 |
| gpt-oss-20b     |          0.8833 |
| gpt-oss-20b+RAG |          0.8767 |
| medgemma-4b     |          0.5800 |
| medgemma-4b+RAG |          0.5600 |

## Consistency (Agreement Across Repetitions)

| setup_name      |   consistency |
|:----------------|--------------:|
| gemma3-4b       |        0.9967 |
| gemma3-4b+RAG   |        1.0000 |
| gpt-oss-20b     |        0.9633 |
| gpt-oss-20b+RAG |        0.9500 |
| medgemma-4b     |        0.9733 |
| medgemma-4b+RAG |        0.9800 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name      |   parse_failure_rate |
|:----------------|---------------------:|
| gemma3-4b       |               0.0000 |
| gemma3-4b+RAG   |               0.0000 |
| gpt-oss-20b     |               0.0033 |
| gpt-oss-20b+RAG |               0.0033 |
| medgemma-4b     |               0.0100 |
| medgemma-4b+RAG |               0.0133 |

## Pairwise Statistical Tests (McNemar's)

| setup_a         | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b       | gemma3-4b+RAG   |               12 |                5 |      2.1176 |    0.1456 | False         |
| gemma3-4b       | gpt-oss-20b     |                4 |               44 |     31.6875 |    0.0000 | True          |
| gemma3-4b       | gpt-oss-20b+RAG |                5 |               46 |     31.3725 |    0.0000 | True          |
| gemma3-4b       | medgemma-4b     |               11 |               21 |      2.5312 |    0.1116 | False         |
| gemma3-4b       | medgemma-4b+RAG |               12 |               19 |      1.1613 |    0.2812 | False         |
| gemma3-4b+RAG   | gpt-oss-20b     |                3 |               50 |     39.9245 |    0.0000 | True          |
| gemma3-4b+RAG   | gpt-oss-20b+RAG |                3 |               51 |     40.9074 |    0.0000 | True          |
| gemma3-4b+RAG   | medgemma-4b     |                9 |               26 |      7.3143 |    0.0068 | True          |
| gemma3-4b+RAG   | medgemma-4b+RAG |                8 |               22 |      5.6333 |    0.0176 | True          |
| gpt-oss-20b     | gpt-oss-20b+RAG |                2 |                3 |      0.0000 |    1.0000 | False         |
| gpt-oss-20b     | medgemma-4b     |               32 |                2 |     24.7353 |    0.0000 | True          |
| gpt-oss-20b     | medgemma-4b+RAG |               37 |                4 |     24.9756 |    0.0000 | True          |
| gpt-oss-20b+RAG | medgemma-4b     |               33 |                2 |     25.7143 |    0.0000 | True          |
| gpt-oss-20b+RAG | medgemma-4b+RAG |               38 |                4 |     25.9286 |    0.0000 | True          |
| medgemma-4b     | medgemma-4b+RAG |                6 |                3 |      0.4444 |    0.5050 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b   | gemma3-4b+RAG   |               12 |                5 |      2.1176 |    0.1456 | False         |
| gpt-oss-20b | gpt-oss-20b+RAG |                2 |                3 |      0.0000 |    1.0000 | False         |
| medgemma-4b | medgemma-4b+RAG |                6 |                3 |      0.4444 |    0.5050 | False         |