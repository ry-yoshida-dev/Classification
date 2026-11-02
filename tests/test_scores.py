from __future__ import annotations

import unittest

import numpy as np

from classification import (
    ClassScoreKind,
    ClassScores,
    HistGradientBoostingParameters,
    MarginClassificationProcessor,
    ProbabilisticClassificationProcessor,
    RandomForestParameters,
    SVCParameters,
)
from classification.types import LabelArray
from tests.helpers import (
    separable_blobs,
)


class TestScores(unittest.TestCase):
    def test_probabilities_sum_to_one(self) -> None:
        X, y = separable_blobs()
        processor: ProbabilisticClassificationProcessor = RandomForestParameters(
            n_estimators=5
        ).build_processor()
        assert isinstance(processor, ProbabilisticClassificationProcessor)
        processor.fit(X, y)
        scores: ClassScores = processor.predict_proba(X)
        self.assertEqual(scores.kind, ClassScoreKind.PROBABILITY)
        np.testing.assert_allclose(scores.values.sum(axis=1), 1.0)

    def test_binary_margins_are_expanded_per_class(self) -> None:
        X, y = separable_blobs(num_classes=2)
        processor: MarginClassificationProcessor = SVCParameters().build_processor()
        processor.fit(X, y)
        scores: ClassScores = processor.decision_function(X)
        self.assertEqual(scores.kind, ClassScoreKind.MARGIN)
        self.assertEqual(scores.values.shape, (X.shape[0], 2))
        predicted: LabelArray = scores.classes[np.argmax(scores.values, axis=1)]
        self.assertEqual(predicted.tolist(), processor.predict(X).tolist())

    def test_multiclass_margins_have_a_column_per_class(self) -> None:
        X, y = separable_blobs()
        processor: MarginClassificationProcessor = SVCParameters().build_processor()
        assert isinstance(processor, MarginClassificationProcessor)
        processor.fit(X, y)
        self.assertEqual(processor.decision_function(X).values.shape, (X.shape[0], 3))
        self.assertIsNone(processor.evaluate(X, y).log_loss)

    def test_nan_features_are_supported_by_hist_gradient_boosting(self) -> None:
        X, y = separable_blobs(num_classes=2)
        X[0, 0] = np.nan
        processor: ProbabilisticClassificationProcessor = (
            HistGradientBoostingParameters(
                max_iter=5, min_samples_leaf=2
            ).build_processor()
        )
        processor.fit(X, y)
        self.assertEqual(processor.predict(X).shape, y.shape)


if __name__ == "__main__":
    _ = unittest.main()
