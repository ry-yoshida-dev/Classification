from __future__ import annotations

import unittest
from typing import cast

import numpy as np

from classification import (
    BaselineParameters,
    DecisionTreeParameters,
    GaussianNaiveBayesParameters,
    HistGradientBoostingParameters,
    KNearestNeighborsAlgorithm,
    KNearestNeighborsMetric,
    KNearestNeighborsParameters,
    LogisticRegressionParameters,
    LogisticRegressionSolver,
    RandomForestParameters,
    SVCParameters,
)
from classification.parameter_validator import ParameterValidator


class TestParameterValidation(unittest.TestCase):
    def test_logistic_regression(self) -> None:
        with self.assertRaises(ValueError):
            _ = LogisticRegressionParameters(C=0.0)
        with self.assertRaises(ValueError):
            _ = LogisticRegressionParameters(l1_ratio=1.5)
        with self.assertRaises(ValueError):
            _ = LogisticRegressionParameters(max_iter=0)
        with self.assertRaises(ValueError):
            _ = LogisticRegressionParameters(tol=-1.0)

    def test_logistic_regression_solver_penalty_compatibility(self) -> None:
        with self.assertRaises(ValueError):
            _ = LogisticRegressionParameters(l1_ratio=0.5)
        with self.assertRaises(ValueError):
            _ = LogisticRegressionParameters(
                l1_ratio=0.5, solver=LogisticRegressionSolver.LIBLINEAR
            )
        _ = LogisticRegressionParameters(
            l1_ratio=1.0, solver=LogisticRegressionSolver.LIBLINEAR
        )
        _ = LogisticRegressionParameters(
            l1_ratio=0.5, solver=LogisticRegressionSolver.SAGA
        )

    def test_svc(self) -> None:
        with self.assertRaises(ValueError):
            _ = SVCParameters(C=-1.0)
        with self.assertRaises(ValueError):
            _ = SVCParameters(gamma=0.0)
        with self.assertRaises(ValueError):
            _ = SVCParameters(max_iter=0)

    def test_k_nearest_neighbors(self) -> None:
        with self.assertRaises(ValueError):
            _ = KNearestNeighborsParameters(n_neighbors=0)
        with self.assertRaises(ValueError):
            _ = KNearestNeighborsParameters(
                metric=KNearestNeighborsMetric.COSINE,
                algorithm=KNearestNeighborsAlgorithm.KD_TREE,
            )

    def test_decision_tree(self) -> None:
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(max_depth=0)
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(min_samples_split=1)
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(ccp_alpha=-0.1)

    def test_random_forest(self) -> None:
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(n_estimators=0)
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(min_samples_leaf=0)

    def test_hist_gradient_boosting(self) -> None:
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(learning_rate=0.0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(max_bins=256)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(validation_fraction=1.0)

    def test_gaussian_naive_bayes(self) -> None:
        with self.assertRaises(ValueError):
            _ = GaussianNaiveBayesParameters(var_smoothing=-1.0)

    def test_hist_gradient_boosting_remaining_ranges(self) -> None:
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(max_iter=0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(max_leaf_nodes=1)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(max_depth=0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(min_samples_leaf=0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(n_iter_no_change=0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(l2_regularization=-1.0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(tol=-1.0)

    def test_decision_tree_remaining_ranges(self) -> None:
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(min_samples_leaf=0)
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(max_leaf_nodes=1)

    def test_random_forest_remaining_ranges(self) -> None:
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(max_depth=0)
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(min_samples_split=1)

    def test_svc_remaining_ranges(self) -> None:
        with self.assertRaises(ValueError):
            _ = SVCParameters(degree=-1)
        with self.assertRaises(ValueError):
            _ = SVCParameters(tol=0.0)

    def test_k_nearest_neighbors_leaf_size(self) -> None:
        with self.assertRaises(ValueError):
            _ = KNearestNeighborsParameters(leaf_size=0)

    def test_nan_is_rejected_at_construction(self) -> None:
        nan: float = float("nan")
        with self.assertRaises(ValueError):
            _ = LogisticRegressionParameters(C=nan)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(learning_rate=nan)
        with self.assertRaises(ValueError):
            _ = SVCParameters(gamma=nan)
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(ccp_alpha=nan)
        with self.assertRaises(ValueError):
            _ = GaussianNaiveBayesParameters(var_smoothing=nan)

    def test_non_integer_is_rejected_at_construction(self) -> None:
        not_integer: int = cast(int, 1.5)
        with self.assertRaises(TypeError):
            _ = RandomForestParameters(n_estimators=not_integer)
        with self.assertRaises(TypeError):
            _ = KNearestNeighborsParameters(n_neighbors=not_integer)
        with self.assertRaises(TypeError):
            _ = BaselineParameters(random_state=not_integer)

    def test_integer_validator_accepts_integers_and_none(self) -> None:
        ParameterValidator.require_integers(
            plain=1, numpy_integer=np.int64(1), optional=None
        )

    def test_integer_validator_rejects_booleans(self) -> None:
        with self.assertRaises(TypeError):
            ParameterValidator.require_integers(flag=True)


if __name__ == "__main__":
    _ = unittest.main()
