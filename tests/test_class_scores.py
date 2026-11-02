from __future__ import annotations

import unittest

import numpy as np

from classification import (
    ClassScores,
)
from classification.types import LabelArray
from tests.helpers import (
    probability_scores,
)


class TestClassScores(unittest.TestCase):
    def test_get_column(self) -> None:
        scores: ClassScores = probability_scores(
            np.array(["a", "b"]), [[0.2, 0.8], [0.6, 0.4]]
        )
        self.assertEqual(scores.get_column("b").tolist(), [0.8, 0.4])
        with self.assertRaises(KeyError):
            _ = scores.get_column("c")

    def test_column_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = probability_scores(np.array([0, 1, 2]), [[0.2, 0.8]])

    def test_values_are_read_only(self) -> None:
        scores: ClassScores = probability_scores(np.array([0, 1]), [[0.2, 0.8]])
        self.assertFalse(scores.values.flags.writeable)

    def test_label_of_another_type_raises_key_error(self) -> None:
        scores: ClassScores = probability_scores(np.array(["a", "b"]), [[0.2, 0.8]])
        with self.assertRaises(KeyError):
            _ = scores.get_column(1)

    def test_classes_are_read_only_copies(self) -> None:
        classes: LabelArray = np.array([0, 1])
        scores: ClassScores = probability_scores(classes, [[0.2, 0.8]])
        self.assertFalse(scores.classes.flags.writeable)
        classes[0] = 9
        self.assertEqual(scores.classes.tolist(), [0, 1])


if __name__ == "__main__":
    _ = unittest.main()
