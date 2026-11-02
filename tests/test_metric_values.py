from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from classification import (
    ClassificationMetric,
    ClassScoreKind,
    ClassScores,
    MetricValues,
    PerClassMetrics,
)
from classification.types import FloatArray, LabelArray
from tests.helpers import (
    probability_scores,
)


class TestMetricValues(unittest.TestCase):
    def test_multiclass_agrees_with_sklearn(self) -> None:
        y_true: LabelArray = np.array([0, 0, 1, 1, 2, 2])
        y_pred: LabelArray = np.array([0, 1, 1, 1, 2, 0])
        scores: ClassScores = probability_scores(
            np.array([0, 1, 2]),
            [
                [0.7, 0.2, 0.1],
                [0.3, 0.6, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.5, 0.1, 0.4],
            ],
        )
        metrics: MetricValues = MetricValues.compute(y_true, y_pred, scores)
        self.assertAlmostEqual(
            metrics.macro_f1,
            f1_score(y_true, y_pred, average="macro", zero_division=0),
        )
        self.assertAlmostEqual(
            metrics.weighted_f1,
            f1_score(y_true, y_pred, average="weighted", zero_division=0),
        )
        self.assertAlmostEqual(
            metrics.balanced_accuracy, balanced_accuracy_score(y_true, y_pred)
        )
        assert metrics.roc_auc is not None
        self.assertAlmostEqual(
            metrics.roc_auc,
            float(
                np.mean(
                    [
                        roc_auc_score(np.equal(y_true, label), scores.values[:, label])
                        for label in range(3)
                    ]
                )
            ),
        )
        self.assertEqual(metrics.num_samples, 6)

    def test_per_class_binary_scores_match_sklearn(self) -> None:
        y_true: LabelArray = np.array(["neg", "neg", "pos", "pos"])
        y_pred: LabelArray = np.array(["neg", "pos", "pos", "pos"])
        scores: ClassScores = probability_scores(
            np.array(["neg", "pos"]),
            [[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]],
        )
        metrics: MetricValues = MetricValues.compute(y_true, y_pred, scores)
        positive: PerClassMetrics = metrics.per_class[1]
        self.assertEqual(positive.label, "pos")
        self.assertEqual(positive.support, 2)
        self.assertAlmostEqual(positive.recall, 1.0)
        self.assertAlmostEqual(
            positive.roc_auc or 0.0,
            roc_auc_score(np.equal(y_true, "pos"), scores.values[:, 1]),
        )
        self.assertAlmostEqual(
            positive.average_precision or 0.0,
            average_precision_score(np.equal(y_true, "pos"), scores.values[:, 1]),
        )

    def test_class_absent_from_labels_does_not_distort_macro_average(self) -> None:
        y_true: LabelArray = np.array([0, 1])
        scores: ClassScores = probability_scores(
            np.array([0, 1, 2]), [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]
        )
        metrics: MetricValues = MetricValues.compute(y_true, y_true, scores)
        self.assertEqual(metrics.macro_f1, 1.0)
        self.assertEqual(metrics.roc_auc, 1.0)
        self.assertIsNone(metrics.per_class[2].roc_auc)
        self.assertEqual(metrics.per_class[2].support, 0)

    def test_single_class_undefined_metrics_are_none(self) -> None:
        y_true: LabelArray = np.array([1, 1])
        scores: ClassScores = probability_scores(
            np.array([0, 1]), [[0.1, 0.9], [0.2, 0.8]]
        )
        metrics: MetricValues = MetricValues.compute(y_true, y_true, scores)
        self.assertIsNone(metrics.cohen_kappa)
        self.assertIsNone(metrics.matthews_correlation)
        self.assertIsNone(metrics.roc_auc)
        self.assertEqual(metrics.get(ClassificationMetric.ACCURACY), 1.0)

    def test_margin_scores_have_no_log_loss(self) -> None:
        y_true: LabelArray = np.array([0, 1])
        values: FloatArray = np.array([[1.0, -1.0], [-1.0, 1.0]])
        scores: ClassScores = ClassScores(
            np.array([0, 1]), values, ClassScoreKind.MARGIN
        )
        metrics: MetricValues = MetricValues.compute(y_true, y_true, scores)
        self.assertIsNone(metrics.log_loss)
        self.assertEqual(metrics.roc_auc, 1.0)

    def test_unknown_label_raises(self) -> None:
        scores: ClassScores = probability_scores(np.array([0, 1]), [[0.5, 0.5]])
        with self.assertRaises(ValueError):
            _ = MetricValues.compute(np.array([7]), np.array([0]), scores)

    def test_misaligned_inputs_raise(self) -> None:
        scores: ClassScores = probability_scores(np.array([0, 1]), [[0.5, 0.5]])
        with self.assertRaises(ValueError):
            _ = MetricValues.compute(np.array([0, 1]), np.array([0]), scores)
        with self.assertRaises(ValueError):
            _ = MetricValues.compute(np.array([0, 1]), np.array([0, 1]), scores)

    def test_every_metric_has_a_getter(self) -> None:
        y_true: LabelArray = np.array([0, 1])
        scores: ClassScores = probability_scores(
            np.array([0, 1]), [[0.9, 0.1], [0.1, 0.9]]
        )
        metrics: MetricValues = MetricValues.compute(y_true, y_true, scores)
        for metric in ClassificationMetric:
            self.assertIsNotNone(metrics.get(metric))

    def test_only_log_loss_is_lower_better(self) -> None:
        lower_better: list[ClassificationMetric] = [
            metric for metric in ClassificationMetric if not metric.is_higher_better
        ]
        self.assertEqual(lower_better, [ClassificationMetric.LOG_LOSS])

    def test_empty_true_labels_raise(self) -> None:
        scores: ClassScores = probability_scores(np.array([0, 1]), [[0.5, 0.5]])
        with self.assertRaises(ValueError):
            _ = MetricValues.compute(
                np.array([], dtype=np.int64), np.array([], dtype=np.int64), scores
            )

    def test_only_log_loss_depends_on_score_kind(self) -> None:
        for metric in ClassificationMetric:
            with self.subTest(metric=metric):
                self.assertTrue(metric.is_defined_for(ClassScoreKind.PROBABILITY))
                self.assertEqual(
                    metric.is_defined_for(ClassScoreKind.MARGIN),
                    metric != ClassificationMetric.LOG_LOSS,
                )


if __name__ == "__main__":
    _ = unittest.main()
