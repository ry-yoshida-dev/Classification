from __future__ import annotations

import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from classification import (
    ClassWeighting,
    KNearestNeighborsMetric,
    KNearestNeighborsParameters,
    KNearestNeighborsWeighting,
    LogisticRegressionParameters,
    MarginClassificationProcessor,
    ProbabilisticClassificationProcessor,
    RandomForestMaxFeatures,
    RandomForestParameters,
    SVCGamma,
    SVCKernel,
    SVCParameters,
)


class TestParameterForwarding(unittest.TestCase):
    def test_logistic_regression(self) -> None:
        processor: ProbabilisticClassificationProcessor = LogisticRegressionParameters(
            C=0.5,
            class_weighting=ClassWeighting.BALANCED,
        ).build_processor()
        assert isinstance(processor, ProbabilisticClassificationProcessor)
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, LogisticRegression)
        self.assertEqual(estimator.C, 0.5)
        self.assertEqual(estimator.class_weight, "balanced")

    def test_svc_gamma_rule_and_value(self) -> None:
        for gamma, expected in ((SVCGamma.AUTO, "auto"), (0.25, 0.25)):
            with self.subTest(gamma=gamma):
                processor: MarginClassificationProcessor = SVCParameters(
                    gamma=gamma, kernel=SVCKernel.POLYNOMIAL
                ).build_processor()
                assert isinstance(processor, MarginClassificationProcessor)
                assert isinstance(processor.estimator, Pipeline)
                estimator: object = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, SVC)
                self.assertEqual(estimator.gamma, expected)
                self.assertEqual(estimator.kernel, "poly")

    def test_svc_unlimited_iterations_map_to_negative_one(self) -> None:
        processor: MarginClassificationProcessor = SVCParameters().build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, SVC)
        self.assertEqual(estimator.max_iter, -1)

    def test_k_nearest_neighbors(self) -> None:
        processor: ProbabilisticClassificationProcessor = KNearestNeighborsParameters(
            n_neighbors=3,
            weighting=KNearestNeighborsWeighting.DISTANCE,
            metric=KNearestNeighborsMetric.MANHATTAN,
        ).build_processor()
        assert isinstance(processor, ProbabilisticClassificationProcessor)
        assert isinstance(processor.estimator, Pipeline)
        estimator: object = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, KNeighborsClassifier)
        self.assertEqual(estimator.n_neighbors, 3)
        self.assertEqual(estimator.weights, "distance")
        self.assertEqual(estimator.metric, "manhattan")

    def test_random_forest_all_features(self) -> None:
        for max_features, expected in (
            (None, None),
            (RandomForestMaxFeatures.LOG2, "log2"),
        ):
            with self.subTest(max_features=max_features):
                processor: ProbabilisticClassificationProcessor = (
                    RandomForestParameters(max_features=max_features).build_processor()
                )
                assert isinstance(processor, ProbabilisticClassificationProcessor)
                assert isinstance(processor.estimator, Pipeline)
                estimator: object = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, RandomForestClassifier)
                self.assertEqual(estimator.max_features, expected)


if __name__ == "__main__":
    _ = unittest.main()
