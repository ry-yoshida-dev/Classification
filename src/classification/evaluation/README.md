# evaluation

## Overview

Estimates generalization performance. `ClassificationEvaluator` builds a fresh processor per fold from a parameter class, measures it on training and held-out samples, and evaluates a baseline on the same splits, so results are free of preprocessing leakage and directly comparable.

## Components

| Component | Description |
|-----------|-------------|
| [evaluator.py](./evaluator.py) | `ClassificationEvaluator`, with `evaluate` and `compare`. |
| [report.py](./report.py) | `EvaluationReport`, a model next to its baseline. |
| [comparison.py](./comparison.py) | `ModelComparison`, candidates ranked by one metric. |
| [ranked_model.py](./ranked_model.py) | `RankedModel`, one candidate in a comparison. |
| [model_evaluation.py](./model_evaluation.py) | `ModelEvaluation`, one configuration across all folds. |
| [fold.py](./fold.py) | `FoldEvaluation`, train and test metrics of one split. |
| [metric_summary.py](./metric_summary.py) | `MetricSummary`, the aggregate of a metric over folds. |
| [partition.py](./partition.py) | `EvaluationPartition` enum: train or test. |
| [splitter.py](./splitter.py) | `DatasetSplitter`, the abstract splitting strategy. |
| [data_split.py](./data_split.py) | `DataSplit`, the indices of one partition. |
| [text_table.py](./text_table.py) | `TextTable`, plain text rendering of reports. |
| [splitters/](./splitters/README.md) | Concrete splitting strategies. |
