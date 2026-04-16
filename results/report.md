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
| gemma3-4b       |     0.5000 |     0.4038 |     0.5962 |        50 |     100 |
| gemma3-4b+RAG   |     0.4400 |     0.3467 |     0.5377 |        44 |     100 |
| gpt-oss-20b     |     0.8600 |     0.7786 |     0.9147 |        86 |     100 |
| gpt-oss-20b+RAG |     0.9000 |     0.8256 |     0.9448 |        90 |     100 |
| medgemma-4b     |     0.5800 |     0.4821 |     0.6720 |        58 |     100 |
| medgemma-4b+RAG |     0.5100 |     0.4135 |     0.6058 |        51 |     100 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name      |   mean_accuracy |
|:----------------|----------------:|
| gemma3-4b       |          0.5000 |
| gemma3-4b+RAG   |          0.4400 |
| gpt-oss-20b     |          0.8767 |
| gpt-oss-20b+RAG |          0.8833 |
| medgemma-4b     |          0.5733 |
| medgemma-4b+RAG |          0.5133 |

## Consistency (Agreement Across Repetitions)

| setup_name      |   consistency |
|:----------------|--------------:|
| gemma3-4b       |        1.0000 |
| gemma3-4b+RAG   |        0.9967 |
| gpt-oss-20b     |        0.9567 |
| gpt-oss-20b+RAG |        0.9633 |
| medgemma-4b     |        0.9800 |
| medgemma-4b+RAG |        0.9867 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name      |   parse_failure_rate |
|:----------------|---------------------:|
| gemma3-4b       |               0.0000 |
| gemma3-4b+RAG   |               0.0000 |
| gpt-oss-20b     |               0.0067 |
| gpt-oss-20b+RAG |               0.0000 |
| medgemma-4b     |               0.0033 |
| medgemma-4b+RAG |               0.0067 |

## Pairwise Statistical Tests (McNemar's)

| setup_a         | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b       | gemma3-4b+RAG   |               10 |                4 |      1.7857 |    0.1814 | False         |
| gemma3-4b       | gpt-oss-20b     |                6 |               42 |     25.5208 |    0.0000 | True          |
| gemma3-4b       | gpt-oss-20b+RAG |                5 |               45 |     30.4200 |    0.0000 | True          |
| gemma3-4b       | medgemma-4b     |               11 |               19 |      1.6333 |    0.2012 | False         |
| gemma3-4b       | medgemma-4b+RAG |               14 |               15 |      0.0000 |    1.0000 | False         |
| gemma3-4b+RAG   | gpt-oss-20b     |                5 |               47 |     32.3269 |    0.0000 | True          |
| gemma3-4b+RAG   | gpt-oss-20b+RAG |                4 |               50 |     37.5000 |    0.0000 | True          |
| gemma3-4b+RAG   | medgemma-4b     |                8 |               22 |      5.6333 |    0.0176 | True          |
| gemma3-4b+RAG   | medgemma-4b+RAG |               10 |               17 |      1.3333 |    0.2482 | False         |
| gpt-oss-20b     | gpt-oss-20b+RAG |                1 |                5 |      1.5000 |    0.2207 | False         |
| gpt-oss-20b     | medgemma-4b     |               32 |                4 |     20.2500 |    0.0000 | True          |
| gpt-oss-20b     | medgemma-4b+RAG |               39 |                4 |     26.8837 |    0.0000 | True          |
| gpt-oss-20b+RAG | medgemma-4b     |               34 |                2 |     26.6944 |    0.0000 | True          |
| gpt-oss-20b+RAG | medgemma-4b+RAG |               42 |                3 |     32.0889 |    0.0000 | True          |
| medgemma-4b     | medgemma-4b+RAG |                9 |                2 |      3.2727 |    0.0704 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b   | gemma3-4b+RAG   |               10 |                4 |      1.7857 |    0.1814 | False         |
| gpt-oss-20b | gpt-oss-20b+RAG |                1 |                5 |      1.5000 |    0.2207 | False         |
| medgemma-4b | medgemma-4b+RAG |                9 |                2 |      3.2727 |    0.0704 | False         |