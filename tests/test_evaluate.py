from __future__ import annotations

import unittest

import numpy as np

from classification import (
    ClassificationEvaluator,
    ClassificationMetric,
    ConfusionMatrix,
    DecisionTreeParameters,
    EvaluationPartition,
    EvaluationReport,
    KNearestNeighborsParameters,
    LogisticRegressionParameters,
    MetricSummary,
    ModelEvaluation,
    RandomForestParameters,
    StratifiedGroupKFoldSplitter,
    StratifiedHoldoutSplitter,
    SupervisedDataset,
    SVCParameters,
    TimeSeriesSplitter,
)
from classification.types import (
    GroupArray,
    IndexArray,
    LabelArray,
    NumericArray,
)
from tests.fixed_splitter import FixedSplitter
from tests.helpers import (
    noise_features,
    separable_blobs,
    single_class_test_folds,
)


class TestEvaluate(unittest.TestCase):
    def test_separable_data_beats_baseline(self) -> None:
        X, y = separable_blobs()
        report: EvaluationReport = ClassificationEvaluator().evaluate(
            LogisticRegressionParameters(), X, y
        )
        self.assertEqual(len(report.model.folds), 5)
        self.assertEqual(
            report.model.summarize(ClassificationMetric.ACCURACY).mean, 1.0
        )
        self.assertGreater(
            report.get_baseline_improvement(ClassificationMetric.ACCURACY), 0.5
        )
        self.assertGreater(
            report.get_baseline_improvement(ClassificationMetric.LOG_LOSS), 0.5
        )

    def test_noise_does_not_beat_baseline_accuracy(self) -> None:
        X, y = noise_features()
        report: EvaluationReport = ClassificationEvaluator().evaluate(
            RandomForestParameters(n_estimators=20), X, y
        )
        self.assertLessEqual(
            report.get_baseline_improvement(ClassificationMetric.ACCURACY), 0.0
        )

    def test_overfitting_shows_as_train_test_gap(self) -> None:
        X, y = noise_features()
        model: ModelEvaluation = (
            ClassificationEvaluator(baseline=None)
            .evaluate(DecisionTreeParameters(), X, y)
            .model
        )
        train: MetricSummary = model.summarize(
            ClassificationMetric.ACCURACY, EvaluationPartition.TRAIN
        )
        test: MetricSummary = model.summarize(ClassificationMetric.ACCURACY)
        self.assertEqual(train.mean, 1.0)
        self.assertLess(test.mean, 0.9)

    def test_scaling_is_fitted_on_training_samples_only(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        X: NumericArray = rng.standard_normal((40, 2))
        y: LabelArray = np.array([0, 1] * 20)
        splitter: StratifiedHoldoutSplitter = StratifiedHoldoutSplitter(
            test_fraction=0.25
        )
        test_indices: IndexArray = splitter.split(SupervisedDataset(X, y))[
            0
        ].test_indices
        X_with_test_outlier: NumericArray = X.copy()
        X_with_test_outlier[test_indices[:1], 0] = 1e9
        evaluator: ClassificationEvaluator = ClassificationEvaluator(
            splitter=splitter, baseline=None
        )
        clean: EvaluationReport = evaluator.evaluate(
            KNearestNeighborsParameters(), X, y
        )
        perturbed: EvaluationReport = evaluator.evaluate(
            KNearestNeighborsParameters(), X_with_test_outlier, y
        )
        self.assertEqual(
            clean.model.folds[0].train_metrics,
            perturbed.model.folds[0].train_metrics,
        )

    def test_is_reproducible(self) -> None:
        X, y = noise_features()
        evaluator: ClassificationEvaluator = ClassificationEvaluator()
        first: EvaluationReport = evaluator.evaluate(
            RandomForestParameters(n_estimators=5), X, y
        )
        second: EvaluationReport = evaluator.evaluate(
            RandomForestParameters(n_estimators=5), X, y
        )
        self.assertEqual(first, second)

    def test_margin_classifier_has_no_log_loss(self) -> None:
        X, y = separable_blobs()
        report: EvaluationReport = ClassificationEvaluator().evaluate(
            SVCParameters(), X, y
        )
        self.assertFalse(report.model.is_defined(ClassificationMetric.LOG_LOSS))
        self.assertTrue(report.model.is_defined(ClassificationMetric.ROC_AUC))
        with self.assertRaises(ValueError):
            _ = report.model.summarize(ClassificationMetric.LOG_LOSS)
        with self.assertRaises(ValueError):
            _ = report.get_baseline_improvement(ClassificationMetric.LOG_LOSS)

    def test_no_baseline(self) -> None:
        X, y = separable_blobs()
        report: EvaluationReport = ClassificationEvaluator(baseline=None).evaluate(
            LogisticRegressionParameters(), X, y
        )
        self.assertIsNone(report.baseline)
        with self.assertRaises(ValueError):
            _ = report.get_baseline_improvement(ClassificationMetric.ACCURACY)

    def test_combined_confusion_matrix_counts_all_test_samples(self) -> None:
        X, y = separable_blobs()
        report: EvaluationReport = ClassificationEvaluator().evaluate(
            LogisticRegressionParameters(), X, y
        )
        matrix: ConfusionMatrix = report.model.confusion_matrix
        self.assertEqual(int(matrix.counts.sum()), y.size)
        self.assertEqual(matrix.counts.sum(axis=1).tolist(), [30, 30, 30])

    def test_holdout_has_unknown_variability(self) -> None:
        X, y = separable_blobs()
        report: EvaluationReport = ClassificationEvaluator(
            splitter=StratifiedHoldoutSplitter()
        ).evaluate(LogisticRegressionParameters(), X, y)
        summary: MetricSummary = report.model.summarize(ClassificationMetric.ACCURACY)
        self.assertIsNone(summary.std)
        self.assertEqual(summary.num_folds, 1)

    def test_group_splitter(self) -> None:
        X, y = separable_blobs(num_samples_per_class=20)
        groups: GroupArray = np.arange(y.size) % 12
        report: EvaluationReport = ClassificationEvaluator(
            splitter=StratifiedGroupKFoldSplitter(groups=groups, n_splits=3)
        ).evaluate(LogisticRegressionParameters(), X, y)
        self.assertEqual(len(report.model.folds), 3)

    def test_time_series_with_class_first_seen_in_test_raises(self) -> None:
        X: NumericArray = np.arange(60.0).reshape(-1, 1)
        y: LabelArray = np.array([0, 1] * 25 + [2] * 10)
        evaluator: ClassificationEvaluator = ClassificationEvaluator(
            splitter=TimeSeriesSplitter(n_splits=3)
        )
        with self.assertRaises(ValueError):
            _ = evaluator.evaluate(LogisticRegressionParameters(), X, y)

    def test_format_table_lists_every_metric(self) -> None:
        X, y = separable_blobs()
        table: str = (
            ClassificationEvaluator().evaluate(SVCParameters(), X, y).format_table()
        )
        for metric in ClassificationMetric:
            self.assertIn(metric.value, table)
        log_loss_row: str = next(
            line for line in table.splitlines() if line.startswith("log_loss")
        )
        self.assertIn("-", log_loss_row)

    def test_baseline_improvement_needs_shared_folds(self) -> None:
        X, y, splitter = single_class_test_folds()
        report: EvaluationReport = ClassificationEvaluator(splitter=splitter).evaluate(
            KNearestNeighborsParameters(n_neighbors=1), X, y
        )
        self.assertEqual(
            report.model.get_defined_fold_indices(
                ClassificationMetric.MATTHEWS_CORRELATION
            ),
            (0, 1),
        )
        with self.assertRaisesRegex(ValueError, "different folds"):
            _ = report.get_baseline_improvement(
                ClassificationMetric.MATTHEWS_CORRELATION
            )
        _ = report.get_baseline_improvement(ClassificationMetric.ACCURACY)

    def test_splitter_without_splits_raises(self) -> None:
        X, y = separable_blobs()
        evaluator: ClassificationEvaluator = ClassificationEvaluator(
            splitter=FixedSplitter(splits=())
        )
        with self.assertRaises(ValueError):
            _ = evaluator.evaluate(LogisticRegressionParameters(), X, y)


if __name__ == "__main__":
    _ = unittest.main()
