# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical multiple-choice QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Domain (inferred)**: medical
- **Questions**: 500
- **Repetitions per question**: 3
- **Total LLM calls**: 12000

## Accuracy Results (Majority Vote)

| setup_name           |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:---------------------|-----------:|-----------:|-----------:|----------:|--------:|
| gemma3-4b+RAG@t0.1   |     0.5020 |     0.4583 |     0.5456 |       251 |     500 |
| gemma3-4b+RAG@t0.3   |     0.5000 |     0.4563 |     0.5437 |       250 |     500 |
| gemma3-4b@t0.1       |     0.4820 |     0.4385 |     0.5258 |       241 |     500 |
| gemma3-4b@t0.3       |     0.4840 |     0.4405 |     0.5278 |       242 |     500 |
| medgemma-4b+RAG@t0.1 |     0.5180 |     0.4742 |     0.5615 |       259 |     500 |
| medgemma-4b+RAG@t0.3 |     0.5180 |     0.4742 |     0.5615 |       259 |     500 |
| medgemma-4b@t0.1     |     0.5520 |     0.5082 |     0.5950 |       276 |     500 |
| medgemma-4b@t0.3     |     0.5480 |     0.5042 |     0.5911 |       274 |     500 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name           |   mean_accuracy |
|:---------------------|----------------:|
| gemma3-4b+RAG@t0.1   |          0.5013 |
| gemma3-4b+RAG@t0.3   |          0.4993 |
| gemma3-4b@t0.1       |          0.4820 |
| gemma3-4b@t0.3       |          0.4813 |
| medgemma-4b+RAG@t0.1 |          0.5153 |
| medgemma-4b+RAG@t0.3 |          0.5167 |
| medgemma-4b@t0.1     |          0.5440 |
| medgemma-4b@t0.3     |          0.5473 |

## Consistency (Agreement Across Repetitions)

| setup_name           |   consistency |
|:---------------------|--------------:|
| gemma3-4b+RAG@t0.1   |        0.9993 |
| gemma3-4b+RAG@t0.3   |        0.9993 |
| gemma3-4b@t0.1       |        0.9987 |
| gemma3-4b@t0.3       |        0.9967 |
| medgemma-4b+RAG@t0.1 |        0.9900 |
| medgemma-4b+RAG@t0.3 |        0.9827 |
| medgemma-4b@t0.1     |        0.9913 |
| medgemma-4b@t0.3     |        0.9853 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name           |   parse_failure_rate |
|:---------------------|---------------------:|
| gemma3-4b+RAG@t0.1   |               0.0000 |
| gemma3-4b+RAG@t0.3   |               0.0000 |
| gemma3-4b@t0.1       |               0.0000 |
| gemma3-4b@t0.3       |               0.0000 |
| medgemma-4b+RAG@t0.1 |               0.0053 |
| medgemma-4b+RAG@t0.3 |               0.0047 |
| medgemma-4b@t0.1     |               0.0047 |
| medgemma-4b@t0.3     |               0.0033 |

## Retrieval Diagnostics (RAG Rows)

- **RAG calls with non-empty context**: 96.77%
- **Avg retrieved chunks per RAG call**: 2.90

## Pairwise Statistical Tests (McNemar's)

| setup_a              | setup_b              |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:---------------------|:---------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b+RAG@t0.1   | gemma3-4b+RAG@t0.3   |                1 |                0 |      0.0000 |    1.0000 | False         |
| gemma3-4b+RAG@t0.1   | gemma3-4b@t0.1       |               60 |               50 |      0.7364 |    0.3908 | False         |
| gemma3-4b+RAG@t0.1   | gemma3-4b@t0.3       |               60 |               51 |      0.5766 |    0.4477 | False         |
| gemma3-4b+RAG@t0.1   | medgemma-4b+RAG@t0.1 |               58 |               66 |      0.3952 |    0.5296 | False         |
| gemma3-4b+RAG@t0.1   | medgemma-4b+RAG@t0.3 |               58 |               66 |      0.3952 |    0.5296 | False         |
| gemma3-4b+RAG@t0.1   | medgemma-4b@t0.1     |               76 |              101 |      3.2542 |    0.0712 | False         |
| gemma3-4b+RAG@t0.1   | medgemma-4b@t0.3     |               77 |              100 |      2.7345 |    0.0982 | False         |
| gemma3-4b+RAG@t0.3   | gemma3-4b@t0.1       |               59 |               50 |      0.5872 |    0.4435 | False         |
| gemma3-4b+RAG@t0.3   | gemma3-4b@t0.3       |               59 |               51 |      0.4455 |    0.5045 | False         |
| gemma3-4b+RAG@t0.3   | medgemma-4b+RAG@t0.1 |               57 |               66 |      0.5203 |    0.4707 | False         |
| gemma3-4b+RAG@t0.3   | medgemma-4b+RAG@t0.3 |               57 |               66 |      0.5203 |    0.4707 | False         |
| gemma3-4b+RAG@t0.3   | medgemma-4b@t0.1     |               75 |              101 |      3.5511 |    0.0595 | False         |
| gemma3-4b+RAG@t0.3   | medgemma-4b@t0.3     |               76 |              100 |      3.0057 |    0.0830 | False         |
| gemma3-4b@t0.1       | gemma3-4b@t0.3       |                0 |                1 |      0.0000 |    1.0000 | False         |
| gemma3-4b@t0.1       | medgemma-4b+RAG@t0.1 |               74 |               92 |      1.7410 |    0.1870 | False         |
| gemma3-4b@t0.1       | medgemma-4b+RAG@t0.3 |               74 |               92 |      1.7410 |    0.1870 | False         |
| gemma3-4b@t0.1       | medgemma-4b@t0.1     |               55 |               90 |      7.9724 |    0.0047 | True          |
| gemma3-4b@t0.1       | medgemma-4b@t0.3     |               55 |               88 |      7.1608 |    0.0075 | True          |
| gemma3-4b@t0.3       | medgemma-4b+RAG@t0.1 |               75 |               92 |      1.5329 |    0.2157 | False         |
| gemma3-4b@t0.3       | medgemma-4b+RAG@t0.3 |               75 |               92 |      1.5329 |    0.2157 | False         |
| gemma3-4b@t0.3       | medgemma-4b@t0.1     |               56 |               90 |      7.4589 |    0.0063 | True          |
| gemma3-4b@t0.3       | medgemma-4b@t0.3     |               56 |               88 |      6.6736 |    0.0098 | True          |
| medgemma-4b+RAG@t0.1 | medgemma-4b+RAG@t0.3 |                2 |                2 |      0.2500 |    0.6171 | False         |
| medgemma-4b+RAG@t0.1 | medgemma-4b@t0.1     |               52 |               69 |      2.1157 |    0.1458 | False         |
| medgemma-4b+RAG@t0.1 | medgemma-4b@t0.3     |               52 |               67 |      1.6471 |    0.1994 | False         |
| medgemma-4b+RAG@t0.3 | medgemma-4b@t0.1     |               51 |               68 |      2.1513 |    0.1425 | False         |
| medgemma-4b+RAG@t0.3 | medgemma-4b@t0.3     |               51 |               66 |      1.6752 |    0.1956 | False         |
| medgemma-4b@t0.1     | medgemma-4b@t0.3     |                2 |                0 |      0.5000 |    0.4795 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a          | setup_b              |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:-----------------|:---------------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b@t0.1   | gemma3-4b+RAG@t0.1   |               50 |               60 |      0.7364 |    0.3908 | False         |
| gemma3-4b@t0.3   | gemma3-4b+RAG@t0.3   |               51 |               59 |      0.4455 |    0.5045 | False         |
| medgemma-4b@t0.1 | medgemma-4b+RAG@t0.1 |               69 |               52 |      2.1157 |    0.1458 | False         |
| medgemma-4b@t0.3 | medgemma-4b+RAG@t0.3 |               66 |               51 |      1.6752 |    0.1956 | False         |