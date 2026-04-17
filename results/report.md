# Experiment Report: What Makes It Right?

## Research Question
What is more crucial for medical QA accuracy: model size, domain expertise, or retrieved knowledge?

## Setup Summary
- **Questions**: 500
- **Repetitions per question**: 3
- **Total LLM calls**: 9000

## Accuracy Results (Majority Vote)

| setup_name      |   accuracy |   ci_lower |   ci_upper |   correct |   total |
|:----------------|-----------:|-----------:|-----------:|----------:|--------:|
| gemma3-4b       |     0.4580 |     0.4148 |     0.5018 |       229 |     500 |
| gemma3-4b+RAG   |     0.4320 |     0.3893 |     0.4758 |       216 |     500 |
| gpt-oss-20b     |     0.8740 |     0.8420 |     0.9003 |       437 |     500 |
| gpt-oss-20b+RAG |     0.8700 |     0.8377 |     0.8967 |       435 |     500 |
| medgemma-4b     |     0.5500 |     0.5062 |     0.5931 |       275 |     500 |
| medgemma-4b+RAG |     0.5280 |     0.4842 |     0.5714 |       264 |     500 |

![Accuracy Comparison](accuracy_comparison.png)

## Mean Per-Question Accuracy

| setup_name      |   mean_accuracy |
|:----------------|----------------:|
| gemma3-4b       |          0.4580 |
| gemma3-4b+RAG   |          0.4320 |
| gpt-oss-20b     |          0.8560 |
| gpt-oss-20b+RAG |          0.8560 |
| medgemma-4b     |          0.5507 |
| medgemma-4b+RAG |          0.5300 |

## Consistency (Agreement Across Repetitions)

| setup_name      |   consistency |
|:----------------|--------------:|
| gemma3-4b       |        0.9960 |
| gemma3-4b+RAG   |        0.9967 |
| gpt-oss-20b     |        0.9480 |
| gpt-oss-20b+RAG |        0.9427 |
| medgemma-4b     |        0.9927 |
| medgemma-4b+RAG |        0.9900 |

![Consistency Comparison](consistency_comparison.png)

## Parse Failure Rate

| setup_name      |   parse_failure_rate |
|:----------------|---------------------:|
| gemma3-4b       |               0.0000 |
| gemma3-4b+RAG   |               0.0000 |
| gpt-oss-20b     |               0.0020 |
| gpt-oss-20b+RAG |               0.0013 |
| medgemma-4b     |               0.0007 |
| medgemma-4b+RAG |               0.0000 |

## Pairwise Statistical Tests (McNemar's)

| setup_a         | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:----------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b       | gemma3-4b+RAG   |               56 |               43 |      1.4545 |    0.2278 | False         |
| gemma3-4b       | gpt-oss-20b     |               18 |              226 |    175.6107 |    0.0000 | True          |
| gemma3-4b       | gpt-oss-20b+RAG |               19 |              225 |    172.2336 |    0.0000 | True          |
| gemma3-4b       | medgemma-4b     |               56 |              102 |     12.8165 |    0.0003 | True          |
| gemma3-4b       | medgemma-4b+RAG |               66 |              101 |      6.9222 |    0.0085 | True          |
| gemma3-4b+RAG   | gpt-oss-20b     |               23 |              244 |    181.2734 |    0.0000 | True          |
| gemma3-4b+RAG   | gpt-oss-20b+RAG |               21 |              240 |    182.0843 |    0.0000 | True          |
| gemma3-4b+RAG   | medgemma-4b     |               59 |              118 |     19.0056 |    0.0000 | True          |
| gemma3-4b+RAG   | medgemma-4b+RAG |               53 |              101 |     14.3442 |    0.0002 | True          |
| gpt-oss-20b     | gpt-oss-20b+RAG |               22 |               20 |      0.0238 |    0.8774 | False         |
| gpt-oss-20b     | medgemma-4b     |              182 |               20 |    128.3218 |    0.0000 | True          |
| gpt-oss-20b     | medgemma-4b+RAG |              193 |               20 |    138.8920 |    0.0000 | True          |
| gpt-oss-20b+RAG | medgemma-4b     |              179 |               19 |    127.6818 |    0.0000 | True          |
| gpt-oss-20b+RAG | medgemma-4b+RAG |              188 |               17 |    140.9756 |    0.0000 | True          |
| medgemma-4b     | medgemma-4b+RAG |               55 |               44 |      1.0101 |    0.3149 | False         |

![Pairwise Significance](pairwise_significance.png)

## RAG Effect Tests

| setup_a     | setup_b         |   a_only_correct |   b_only_correct |   statistic |   p_value | significant   |
|:------------|:----------------|-----------------:|-----------------:|------------:|----------:|:--------------|
| gemma3-4b   | gemma3-4b+RAG   |               56 |               43 |      1.4545 |    0.2278 | False         |
| gpt-oss-20b | gpt-oss-20b+RAG |               22 |               20 |      0.0238 |    0.8774 | False         |
| medgemma-4b | medgemma-4b+RAG |               55 |               44 |      1.0101 |    0.3149 | False         |