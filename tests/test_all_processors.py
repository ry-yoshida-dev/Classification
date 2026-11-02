from __future__ import annotations

import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from classification import (
    ClassificationMethod,
    ClassificationParameters,
    DecisionTreeParameters,
    GaussianNaiveBayesParameters,
    HistGradientBoostingParameters,
    KNearestNeighborsParameters,
    LogisticRegressionParameters,
    MarginClassificationProcessor,
    MetricValues,
    ProbabilisticClassificationProcessor,
    RandomForestParameters,
    SVCParameters,
)
from classification.processor import ClassificationProcessor
from classification.types import LabelArray
from tests.helpers import (
    separable_blobs,
)

ALL_PARAMETERS: tuple[ClassificationParameters, ...] = (
    LogisticRegressionParameters(),
    SVCParameters(),
    KNearestNeighborsParameters(),
    DecisionTreeParameters(),
    RandomForestParameters(n_estimators=10),
    HistGradientBoostingParameters(max_iter=10, min_samples_leaf=2),
    GaussianNaiveBayesParameters(),
)


class TestAllProcessors(unittest.TestCase):
    def test_learns_separable_data(self) -> None:
        X, y = separable_blobs()
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor: ClassificationProcessor = parameters.build_processor()
                processor.fit(X, y)
                self.assertEqual(processor.classes.tolist(), [0, 1, 2])
                self.assertEqual(processor.predict(X).tolist(), y.tolist())
                metrics: MetricValues = processor.evaluate(X, y)
                self.assertEqual(metrics.accuracy, 1.0)
                self.assertEqual(metrics.roc_auc, 1.0)

    def test_predict_before_fit_raises(self) -> None:
        X, _ = separable_blobs()
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                with self.assertRaises(NotFittedError):
                    _ = parameters.build_processor().predict(X)

    def test_string_labels(self) -> None:
        X, y = separable_blobs(num_classes=2)
        labels: LabelArray = np.array(["cat", "dog"])[y]
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor: ClassificationProcessor = parameters.build_processor()
                processor.fit(X, labels)
                self.assertEqual(processor.evaluate(X, labels).accuracy, 1.0)

    def test_single_class_is_rejected(self) -> None:
        X, y = separable_blobs()
        processor: ProbabilisticClassificationProcessor = (
            LogisticRegressionParameters().build_processor()
        )
        with self.assertRaises(ValueError):
            processor.fit(X, np.zeros_like(y))

    def test_failed_refit_clears_fitted_state(self) -> None:
        X, y = separable_blobs()
        processor: ProbabilisticClassificationProcessor = (
            LogisticRegressionParameters().build_processor()
        )
        processor.fit(X, y)
        with self.assertRaises(ValueError):
            processor.fit(X, np.zeros_like(y))
        with self.assertRaises(NotFittedError):
            _ = processor.classes

    def test_unseen_label_in_evaluate_raises(self) -> None:
        X, y = separable_blobs(num_classes=2)
        processor: ProbabilisticClassificationProcessor = (
            LogisticRegressionParameters().build_processor()
        )
        processor.fit(X, y)
        with self.assertRaises(ValueError):
            _ = processor.evaluate(X, y + 5)

    def test_method_is_reported(self) -> None:
        processor: MarginClassificationProcessor = SVCParameters().build_processor()
        self.assertEqual(processor.method, ClassificationMethod.SVC)

    def test_processors_compare_by_identity(self) -> None:
        X, y = separable_blobs()
        first: ClassificationProcessor = (
            GaussianNaiveBayesParameters().build_processor()
        )
        second: ClassificationProcessor = (
            GaussianNaiveBayesParameters().build_processor()
        )
        first.fit(X, y)
        second.fit(X, y)
        self.assertNotEqual(first, second)
        self.assertEqual(first, first)

    def test_fitted_classes_are_read_only(self) -> None:
        X, y = separable_blobs()
        processor: ClassificationProcessor = (
            GaussianNaiveBayesParameters().build_processor()
        )
        processor.fit(X, y)
        self.assertFalse(processor.classes.flags.writeable)

    def test_score_kind_matches_scores(self) -> None:
        X, y = separable_blobs(num_classes=2)
        svc: MarginClassificationProcessor = SVCParameters().build_processor()
        tree: ProbabilisticClassificationProcessor = (
            DecisionTreeParameters().build_processor()
        )
        svc.fit(X, y)
        tree.fit(X, y)
        self.assertEqual(svc.score_kind, svc.decision_function(X).kind)
        self.assertEqual(tree.score_kind, tree.predict_proba(X).kind)


if __name__ == "__main__":
    _ = unittest.main()
