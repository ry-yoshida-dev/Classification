from __future__ import annotations

import unittest

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from classification import (
    FeatureScaling,
)


class TestFeatureScaling(unittest.TestCase):
    def test_scaler_types(self) -> None:
        estimator: object = object()
        self.assertIsInstance(
            FeatureScaling.STANDARD.build_pipeline(estimator).named_steps["scaler"],
            StandardScaler,
        )
        self.assertIsInstance(
            FeatureScaling.MIN_MAX.build_pipeline(estimator).named_steps["scaler"],
            MinMaxScaler,
        )
        self.assertIsInstance(
            FeatureScaling.ROBUST.build_pipeline(estimator).named_steps["scaler"],
            RobustScaler,
        )
        self.assertEqual(
            FeatureScaling.NONE.build_pipeline(estimator).steps[0],
            ("scaler", "passthrough"),
        )


if __name__ == "__main__":
    _ = unittest.main()
