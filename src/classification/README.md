# classification

## Overview

This module provides a unified interface for classification. Layers depend only on those listed before them: shared types, defaults, validators, and `dataset.py`; `metrics/`, which computes performance from predictions; `processor.py` and `processors/`, which wrap estimators; `methods/`, whose parameter classes build processors; and `evaluation/`, which trains models from parameter classes on dataset splits.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Shared type aliases. |
| [method.py](./method.py) | `ClassificationMethod` enum of available methods. |
| [dataset.py](./dataset.py) | `SupervisedDataset`, the validated pair of features and labels. |
| [feature_scaling.py](./feature_scaling.py) | `FeatureScaling` enum that builds the scaler and estimator pipeline. |
| [class_weighting.py](./class_weighting.py) | `ClassWeighting` enum for imbalanced data. |
| [tree_criterion.py](./tree_criterion.py) | `TreeCriterion` enum, shared by decision trees and random forests. |
| [defaults.py](./defaults.py) | Default values shared across the package, such as the random seed. |
| [parameter.py](./parameter.py) | Abstract base class for classification parameters. |
| [parameter_validator.py](./parameter_validator.py) | `ParameterValidator`, runtime checks for NaN and non-integer parameters. |
| [processor.py](./processor.py) | Abstract base class for processors; implements validation, fitted state, `predict`, and `evaluate`. |
| [processors/](./processors/README.md) | Concrete processors, split by the type of per-class scores. |
| [methods/](./methods/README.md) | Per-method parameter classes and enums. |
| [metrics/](./metrics/README.md) | Performance metrics computed from labels and scores. |
| [evaluation/](./evaluation/README.md) | Splitters, cross-validated evaluation, reports, and model comparison. |
