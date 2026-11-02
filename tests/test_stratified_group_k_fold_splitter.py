from __future__ import annotations

import unittest

import numpy as np

from classification import (
    DataSplit,
    StratifiedGroupKFoldSplitter,
    SupervisedDataset,
)
from classification.types import GroupArray
from tests.helpers import (
    separable_blobs,
)


class TestStratifiedGroupKFoldSplitter(unittest.TestCase):
    def test_groups_never_cross_partitions(self) -> None:
        X, y = separable_blobs(num_samples_per_class=20)
        groups: GroupArray = np.arange(y.size) % 12
        splits: tuple[DataSplit, ...] = StratifiedGroupKFoldSplitter(
            groups=groups, n_splits=3
        ).split(SupervisedDataset(X, y))
        for split in splits:
            self.assertEqual(
                np.intersect1d(
                    groups[split.train_indices], groups[split.test_indices]
                ).size,
                0,
            )

    def test_group_length_mismatch_raises(self) -> None:
        X, y = separable_blobs()
        splitter: StratifiedGroupKFoldSplitter = StratifiedGroupKFoldSplitter(
            groups=np.array([0, 1])
        )
        with self.assertRaises(ValueError):
            _ = splitter.split(SupervisedDataset(X, y))

    def test_invalid_n_splits_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = StratifiedGroupKFoldSplitter(groups=np.array([0, 1]), n_splits=1)

    def test_float_groups_raise(self) -> None:
        with self.assertRaises(TypeError):
            _ = StratifiedGroupKFoldSplitter(groups=np.array([0.5, 1.5]))


if __name__ == "__main__":
    _ = unittest.main()
