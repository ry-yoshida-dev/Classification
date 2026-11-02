from __future__ import annotations

import unittest

import numpy as np

from classification import SupervisedDataset


class TestSupervisedDataset(unittest.TestCase):
    def test_valid_dataset(self) -> None:
        dataset: SupervisedDataset = SupervisedDataset(
            np.zeros((3, 2)), np.array(["b", "a", "b"])
        )
        self.assertEqual(dataset.num_samples, 3)
        self.assertEqual(list(dataset.classes), ["a", "b"])

    def test_subset(self) -> None:
        dataset: SupervisedDataset = SupervisedDataset(
            np.arange(6.0).reshape(3, 2), np.array([0, 1, 2])
        )
        subset: SupervisedDataset = dataset.subset(np.array([2, 0], dtype=np.intp))
        self.assertEqual(subset.y.tolist(), [2, 0])
        np.testing.assert_array_equal(subset.X, [[4.0, 5.0], [0.0, 1.0]])

    def test_invalid_feature_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = SupervisedDataset(np.zeros(3), np.zeros(3, dtype=np.int64))
        with self.assertRaises(ValueError):
            _ = SupervisedDataset(np.zeros((0, 2)), np.zeros(0, dtype=np.int64))

    def test_non_numeric_features_raise(self) -> None:
        with self.assertRaises(TypeError):
            _ = SupervisedDataset(np.array([["a"]]), np.array([0]))

    def test_float_labels_raise(self) -> None:
        with self.assertRaises(TypeError):
            _ = SupervisedDataset(np.zeros((2, 1)), np.array([0.0, 1.0]))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = SupervisedDataset(np.zeros((2, 1)), np.array([0, 1, 1]))

    def test_multi_dimensional_labels_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = SupervisedDataset(np.zeros((2, 1)), np.array([[0], [1]]))


if __name__ == "__main__":
    _ = unittest.main()
