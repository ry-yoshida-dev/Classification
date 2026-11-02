from __future__ import annotations

import unittest

import numpy as np

from classification import (
    BaselineParameters,
    BaselineStrategy,
    ProbabilisticClassificationProcessor,
)
from classification.types import LabelArray, NumericArray


class TestBaseline(unittest.TestCase):
    def test_prior_predicts_majority_with_prior_probabilities(self) -> None:
        X: NumericArray = np.zeros((4, 1))
        y: LabelArray = np.array([0, 0, 0, 1])
        processor: ProbabilisticClassificationProcessor = (
            BaselineParameters().build_processor()
        )
        processor.fit(X, y)
        self.assertEqual(processor.predict(X).tolist(), [0, 0, 0, 0])
        assert isinstance(processor, ProbabilisticClassificationProcessor)
        self.assertEqual(processor.predict_proba(X).values.tolist()[0], [0.75, 0.25])

    def test_uniform_strategy_is_reproducible(self) -> None:
        X: NumericArray = np.zeros((20, 1))
        y: LabelArray = np.array([0, 1] * 10)
        parameters: BaselineParameters = BaselineParameters(
            strategy=BaselineStrategy.UNIFORM
        )
        first: ProbabilisticClassificationProcessor = parameters.build_processor()
        second: ProbabilisticClassificationProcessor = parameters.build_processor()
        first.fit(X, y)
        second.fit(X, y)
        self.assertEqual(first.predict(X).tolist(), second.predict(X).tolist())


if __name__ == "__main__":
    _ = unittest.main()
