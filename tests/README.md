# Tests

## Overview

Unit tests for the `classification` package: data validation, metrics checked against scikit-learn, parameter validation and forwarding, processor behavior for all methods, splitters, and end-to-end evaluation.

## Components

| File | Description |
| --- | --- |
| [`helpers.py`](helpers.py) | Shared synthetic datasets and score builders. |
| [`fixed_splitter.py`](fixed_splitter.py) | `FixedSplitter`, a splitter with predetermined splits. |
| [`test_dataset.py`](test_dataset.py) | `SupervisedDataset` validation and subsetting. |
| [`test_class_scores.py`](test_class_scores.py) | `ClassScores` lookups and immutability. |
| [`test_confusion_matrix.py`](test_confusion_matrix.py) | `ConfusionMatrix` counting, combining, lookups, equality, and hashing. |
| [`test_metric_values.py`](test_metric_values.py) | `MetricValues` checked against scikit-learn, including undefined metrics. |
| [`test_parameter_validation.py`](test_parameter_validation.py) | Parameter range, NaN, and integer validation. |
| [`test_parameter_forwarding.py`](test_parameter_forwarding.py) | Forwarding of parameters to estimators. |
| [`test_feature_scaling.py`](test_feature_scaling.py) | `FeatureScaling` pipelines. |
| [`test_all_processors.py`](test_all_processors.py) | Fit, predict, evaluate, and fitted state across all methods. |
| [`test_scores.py`](test_scores.py) | Probabilities and margins. |
| [`test_baseline.py`](test_baseline.py) | `BaselineParameters` strategies. |
| [`test_stratified_holdout_splitter.py`](test_stratified_holdout_splitter.py) | `StratifiedHoldoutSplitter` partitions. |
| [`test_stratified_k_fold_splitter.py`](test_stratified_k_fold_splitter.py) | `StratifiedKFoldSplitter` partitions. |
| [`test_stratified_group_k_fold_splitter.py`](test_stratified_group_k_fold_splitter.py) | `StratifiedGroupKFoldSplitter` partitions. |
| [`test_time_series_splitter.py`](test_time_series_splitter.py) | `TimeSeriesSplitter` partitions. |
| [`test_evaluate.py`](test_evaluate.py) | `ClassificationEvaluator.evaluate`, reports, and absence of test-to-train leakage. |
| [`test_compare.py`](test_compare.py) | `ClassificationEvaluator.compare` and comparison of folds. |
| [`test_metric_summary.py`](test_metric_summary.py) | `MetricSummary` aggregation and formatting. |

## Examples

```bash
python -m pytest tests/ -v
```
