# Classification

## Overview

Classification (`classification`) is a Python package that wraps scikit-learn classifiers behind a small, consistent API and adds a performance evaluation layer. Parameter dataclasses validate their fields and build `ClassificationProcessor` instances with `fit`, `predict`, and `evaluate`; `ClassificationEvaluator` estimates generalization performance by cross-validation and compares configurations.

Supported methods are logistic regression, support vector classification, k-nearest neighbors, decision trees, random forests, histogram gradient boosting, and Gaussian naive Bayes. Everything is exported from the top-level `classification` package.

For module-level design notes, see [src/classification/README.md](src/classification/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Quick example

Train a model and measure it on separate test data:

```python
from classification import RandomForestParameters

processor = RandomForestParameters(n_estimators=200).build_processor()
processor.fit(X_train, y_train)
labels = processor.predict(X_test)
probabilities = processor.predict_proba(X_test)

metrics = processor.evaluate(X_test, y_test)
print(metrics.macro_f1, metrics.roc_auc)
print(metrics.confusion_matrix.counts)
```

Cross-validate a configuration against a feature-independent baseline:

```python
from classification import (
    ClassificationEvaluator,
    ClassificationMetric,
    LogisticRegressionParameters,
)

report = ClassificationEvaluator().evaluate(LogisticRegressionParameters(), X, y)
print(report.format_table())
print(report.model.summarize(ClassificationMetric.MACRO_F1).mean)
print(report.get_baseline_improvement(ClassificationMetric.MACRO_F1))
```

```text
metric                train            test             baseline test
--------------------  ---------------  ---------------  ---------------
accuracy              0.9881 ± 0.0046  0.9789 ± 0.0100  0.6274 ± 0.0044
macro_f1              0.9873 ± 0.0049  0.9773 ± 0.0108  0.3855 ± 0.0017
...
```

Rank configurations on identical splits:

```python
from classification import (
    ClassificationEvaluator,
    ClassificationMetric,
    LogisticRegressionParameters,
    RandomForestParameters,
    SVCParameters,
)

comparison = ClassificationEvaluator().compare(
    {
        "svc": SVCParameters(),
        "logistic": LogisticRegressionParameters(),
        "forest": RandomForestParameters(),
    },
    X,
    y,
    metric=ClassificationMetric.ROC_AUC,
)
print(comparison.best.name)
print(comparison.format_table())
```

Keep correlated samples together, or respect temporal order, by choosing a splitter:

```python
from classification import ClassificationEvaluator, StratifiedGroupKFoldSplitter

evaluator = ClassificationEvaluator(
    splitter=StratifiedGroupKFoldSplitter(groups=subject_ids, n_splits=5)
)
```

## Notes

- Scaling is part of the estimator pipeline, and every fold builds a fresh processor, so test samples never influence preprocessing. Scale-sensitive methods default to `FeatureScaling.STANDARD`.
- Labels must be integers, booleans, or strings with shape `(n_samples,)`. Float and object labels raise `TypeError`, and multi-label or multi-output targets are not supported.
- Macro-averaged ROC AUC and average precision are one-vs-rest, also for binary problems; for the score of the positive class, read `metrics.per_class`.
- Metrics that cannot be computed are `None`, not a misleading number: `log_loss` for `SVCParameters`, whose processor produces margins rather than probabilities, and `roc_auc` when no class has both positive and negative samples.
- Evaluating on a label the classifier was not fitted on raises `ValueError`, as does a split that trains on a single class.
- `MetricSummary.std` is `None` for a single holdout split: the variability is unknown, not zero.
- Means are only compared on the same folds. MCC and Cohen's kappa are undefined on a fold where only one class occurs in the true and predicted labels, so `compare` and `get_baseline_improvement` raise `ValueError` when they are defined on different folds for two models, and the comparison table omits such a baseline.
- Parameter and splitter objects reject NaN and non-integer values at construction, with `ValueError` and `TypeError` respectively.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run basedpyright src tests
uv run pytest
```

The same checks run in CI on every push to `main` and every pull request.

## License

Released under the [MIT License](LICENSE).
