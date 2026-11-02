from __future__ import annotations

import unittest
from typing import cast

import numpy as np

from classification import (
    DataSplit,
    StratifiedKFoldSplitter,
    SupervisedDataset,
)
from classification.types import IndexArray, LabelArray, NumericArray
from tests.helpers import (
    separable_blobs,
)


class TestStratifiedKFoldSplitter(unittest.TestCase):
    def test_test_partitions_cover_all_samples_once(self) -> None:
        X, y = separable_blobs()
        splits: tuple[DataSplit, ...] = StratifiedKFoldSplitter(n_splits=3).split(
            SupervisedDataset(X, y)
        )
        test_indices: IndexArray = np.concatenate(
            [split.test_indices for split in splits]
        )
        self.assertEqual(np.sort(test_indices).tolist(), list(range(y.size)))
        for split in splits:
            self.assertEqual(
                np.intersect1d(split.train_indices, split.test_indices).size, 0
            )

    def test_is_reproducible(self) -> None:
        X, y = separable_blobs()
        dataset: SupervisedDataset = SupervisedDataset(X, y)
        first: tuple[DataSplit, ...] = StratifiedKFoldSplitter(random_state=1).split(
            dataset
        )
        second: tuple[DataSplit, ...] = StratifiedKFoldSplitter(random_state=1).split(
            dataset
        )
        np.testing.assert_array_equal(first[0].test_indices, second[0].test_indices)

    def test_unshuffled_ignores_random_state(self) -> None:
        X, y = separable_blobs()
        splits: tuple[DataSplit, ...] = StratifiedKFoldSplitter(
            is_shuffled=False, random_state=1
        ).split(SupervisedDataset(X, y))
        self.assertEqual(len(splits), 5)

    def test_rare_class_raises(self) -> None:
        X: NumericArray = np.zeros((7, 1))
        y: LabelArray = np.array([0, 0, 0, 0, 0, 1, 1])
        with self.assertRaises(ValueError):
            _ = StratifiedKFoldSplitter(n_splits=3).split(SupervisedDataset(X, y))

    def test_invalid_n_splits_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = StratifiedKFoldSplitter(n_splits=1)

    def test_non_integer_n_splits_raises(self) -> None:
        with self.assertRaises(TypeError):
            _ = StratifiedKFoldSplitter(n_splits=cast(int, 2.5))


if __name__ == "__main__":
    _ = unittest.main()
