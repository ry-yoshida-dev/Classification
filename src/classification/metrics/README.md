# metrics

## Overview

Performance metrics computed from true labels, predicted labels, and per-class scores. Metrics that cannot be computed for an input are `None` rather than a placeholder number.

## Components

| Component | Description |
|-----------|-------------|
| [values.py](./values.py) | `MetricValues`, all metrics for one set of samples. |
| [metric.py](./metric.py) | `ClassificationMetric` enum of scalar metrics, with their direction of improvement. |
| [per_class.py](./per_class.py) | `PerClassMetrics`, the metrics of a single class. |
| [confusion_matrix.py](./confusion_matrix.py) | `ConfusionMatrix`, which can be normalized and combined across folds. |
| [class_scores.py](./class_scores.py) | `ClassScores`, per-class scores tied to their class labels. |
| [class_score_kind.py](./class_score_kind.py) | `ClassScoreKind` enum: probability or margin. |
