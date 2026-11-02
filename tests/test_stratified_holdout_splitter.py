from __future__ import annotations

import unittest
from typing import cast

import numpy as np

from classification import (
    DataSplit,
    StratifiedHoldoutSplitter,
    SupervisedDataset,
)
from tests.helpers import (
    separable_blobs,
)


class TestStratifiedHoldoutSplitter(unittest.TestCase):
    def test_preserves_class_proportions(self) -> None:
        X, y = separable_blobs(num_samples_per_class=20)
        splits: tuple[DataSplit, ...] = StratifiedHoldoutSplitter(
            test_fraction=0.25
        ).split(SupervisedDataset(X, y))
        self.assertEqual(len(splits), 1)
        self.assertEqual(np.bincount(y[splits[0].test_indices]).tolist(), [5, 5, 5])

    def test_invalid_fraction_raises(self) -> None:
        for test_fraction in (0.0, 1.0):
            with self.subTest(test_fraction=test_fraction):
                with self.assertRaises(ValueError):
                    _ = StratifiedHoldoutSplitter(test_fraction=test_fraction)

    def test_non_integer_seed_raises(self) -> None:
        with self.assertRaises(TypeError):
            _ = StratifiedHoldoutSplitter(random_state=cast(int, 1.5))


if __name__ == "__main__":
    _ = unittest.main()
