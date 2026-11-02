from __future__ import annotations

import unittest

from classification import (
    BaselineParameters,
    ClassificationEvaluator,
    ClassificationMetric,
    DecisionTreeParameters,
    KNearestNeighborsParameters,
    LogisticRegressionParameters,
    ModelComparison,
    StratifiedKFoldSplitter,
    SVCParameters,
)
from tests.helpers import (
    noise_features,
    separable_blobs,
    single_class_test_folds,
)


class TestCompare(unittest.TestCase):
    def test_ranks_best_first(self) -> None:
        X, y = separable_blobs()
        comparison: ModelComparison = ClassificationEvaluator().compare(
            {
                "underfit": DecisionTreeParameters(max_depth=1),
                "fit": LogisticRegressionParameters(),
            },
            X,
            y,
        )
        self.assertEqual(comparison.best.name, "fit")
        self.assertEqual(
            [entry.name for entry in comparison.ranking], ["fit", "underfit"]
        )
        self.assertIn("(baseline)", comparison.format_table())

    def test_lower_better_metric_ranks_ascending(self) -> None:
        X, y = separable_blobs()
        comparison: ModelComparison = ClassificationEvaluator().compare(
            {
                "underfit": DecisionTreeParameters(max_depth=1),
                "fit": LogisticRegressionParameters(),
            },
            X,
            y,
            metric=ClassificationMetric.LOG_LOSS,
        )
        self.assertEqual(comparison.best.name, "fit")
        self.assertLess(
            comparison.ranking[0].score.mean, comparison.ranking[1].score.mean
        )

    def test_candidates_share_splits(self) -> None:
        X, y = noise_features()
        evaluator: ClassificationEvaluator = ClassificationEvaluator(
            splitter=StratifiedKFoldSplitter(n_splits=3)
        )
        comparison: ModelComparison = evaluator.compare(
            {"first": DecisionTreeParameters(), "second": DecisionTreeParameters()},
            X,
            y,
        )
        self.assertEqual(
            comparison.ranking[0].evaluation, comparison.ranking[1].evaluation
        )

    def test_empty_candidates_raise(self) -> None:
        X, y = separable_blobs()
        with self.assertRaises(ValueError):
            _ = ClassificationEvaluator().compare({}, X, y)

    def test_metric_undefined_for_candidate_raises(self) -> None:
        X, y = separable_blobs()
        with self.assertRaises(ValueError):
            _ = ClassificationEvaluator().compare(
                {"svc": SVCParameters()}, X, y, metric=ClassificationMetric.LOG_LOSS
            )

    def test_metric_unsupported_by_candidate_raises_before_training(self) -> None:
        X, y = separable_blobs()
        with self.assertRaisesRegex(ValueError, "produces margin scores"):
            _ = ClassificationEvaluator().compare(
                {"svc": SVCParameters()}, X, y, metric=ClassificationMetric.LOG_LOSS
            )

    def test_metric_defined_on_different_folds_raises(self) -> None:
        X, y, splitter = single_class_test_folds()
        with self.assertRaisesRegex(ValueError, "not comparable"):
            _ = ClassificationEvaluator(splitter=splitter).compare(
                {
                    "majority": BaselineParameters(),
                    "neighbors": KNearestNeighborsParameters(n_neighbors=1),
                },
                X,
                y,
                metric=ClassificationMetric.MATTHEWS_CORRELATION,
            )

    def test_metric_defined_on_all_folds_ranks_despite_single_class_fold(self) -> None:
        X, y, splitter = single_class_test_folds()
        comparison: ModelComparison = ClassificationEvaluator(
            splitter=splitter
        ).compare(
            {
                "majority": BaselineParameters(),
                "neighbors": KNearestNeighborsParameters(n_neighbors=1),
            },
            X,
            y,
        )
        self.assertEqual(len(comparison.ranking), 2)

    def test_table_omits_baseline_defined_on_other_folds(self) -> None:
        X, y, splitter = single_class_test_folds()
        comparison: ModelComparison = ClassificationEvaluator(
            splitter=splitter
        ).compare(
            {"neighbors": KNearestNeighborsParameters(n_neighbors=1)},
            X,
            y,
            metric=ClassificationMetric.MATTHEWS_CORRELATION,
        )
        self.assertNotIn("(baseline)", comparison.format_table())


if __name__ == "__main__":
    _ = unittest.main()
