from __future__ import annotations

import unittest

import numpy as np

from classification import (
    ConfusionMatrix,
)
from classification.types import CountArray


class TestConfusionMatrix(unittest.TestCase):
    def test_from_predictions(self) -> None:
        matrix: ConfusionMatrix = ConfusionMatrix.from_predictions(
            np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([0, 1])
        )
        self.assertEqual(matrix.counts.tolist(), [[1, 1], [0, 1]])
        self.assertEqual(matrix.get_count(0, 1), 1)
        self.assertEqual(matrix.row_normalized.tolist(), [[0.5, 0.5], [0.0, 1.0]])

    def test_row_normalized_empty_row_is_zero(self) -> None:
        matrix: ConfusionMatrix = ConfusionMatrix(
            np.array([0, 1]), np.array([[0, 0], [1, 1]])
        )
        self.assertEqual(matrix.row_normalized.tolist(), [[0.0, 0.0], [0.5, 0.5]])

    def test_unknown_label_raises(self) -> None:
        matrix: ConfusionMatrix = ConfusionMatrix(
            np.array([0, 1]), np.array([[1, 0], [0, 1]])
        )
        with self.assertRaises(KeyError):
            _ = matrix.get_count(0, 9)

    def test_invalid_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = ConfusionMatrix(np.array([0, 1]), np.array([[1, 0]]))

    def test_combine_aligns_different_classes(self) -> None:
        first: ConfusionMatrix = ConfusionMatrix(
            np.array([0, 2]), np.array([[1, 2], [3, 4]])
        )
        second: ConfusionMatrix = ConfusionMatrix(
            np.array([1, 2]), np.array([[5, 6], [7, 8]])
        )
        combined: ConfusionMatrix = first.combine(second)
        self.assertEqual(combined.classes.tolist(), [0, 1, 2])
        self.assertEqual(combined.counts.tolist(), [[1, 0, 2], [0, 5, 6], [3, 7, 12]])

    def test_equality(self) -> None:
        first: ConfusionMatrix = ConfusionMatrix(
            np.array([0, 1]), np.array([[1, 0], [0, 1]])
        )
        second: ConfusionMatrix = ConfusionMatrix(
            np.array([0, 1]), np.array([[1, 0], [0, 1]])
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

    def test_label_of_another_type_raises_key_error(self) -> None:
        strings: ConfusionMatrix = ConfusionMatrix(
            np.array(["a", "b"]), np.array([[1, 0], [0, 1]])
        )
        integers: ConfusionMatrix = ConfusionMatrix(
            np.array([0, 1]), np.array([[1, 0], [0, 1]])
        )
        with self.assertRaises(KeyError):
            _ = strings.get_count(1, "a")
        with self.assertRaises(KeyError):
            _ = integers.get_count("a", 0)

    def test_equal_matrices_of_different_dtypes_have_equal_hashes(self) -> None:
        counts: CountArray = np.eye(2, dtype=np.int64)
        for first_dtype, second_dtype, labels in (
            (np.int32, np.int64, [0, 1]),
            ("<U1", "<U5", ["a", "b"]),
        ):
            with self.subTest(first_dtype=first_dtype):
                first: ConfusionMatrix = ConfusionMatrix(
                    np.array(labels, dtype=first_dtype), counts
                )
                second: ConfusionMatrix = ConfusionMatrix(
                    np.array(labels, dtype=second_dtype), counts
                )
                self.assertEqual(first, second)
                self.assertEqual(hash(first), hash(second))

    def test_not_equal_to_other_types(self) -> None:
        matrix: ConfusionMatrix = ConfusionMatrix(
            np.array([0, 1]), np.array([[1, 0], [0, 1]])
        )
        self.assertNotEqual(matrix, "matrix")

    def test_classes_are_read_only(self) -> None:
        matrix: ConfusionMatrix = ConfusionMatrix(
            np.array([0, 1]), np.array([[1, 0], [0, 1]])
        )
        self.assertFalse(matrix.classes.flags.writeable)


if __name__ == "__main__":
    _ = unittest.main()
