from __future__ import annotations

import unittest

from classification import (
    DataSplit,
    SupervisedDataset,
    TimeSeriesSplitter,
)
from tests.helpers import (
    separable_blobs,
)


class TestTimeSeriesSplitter(unittest.TestCase):
    def test_training_precedes_test_with_gap(self) -> None:
        X, y = separable_blobs()
        splits: tuple[DataSplit, ...] = TimeSeriesSplitter(n_splits=3, gap=2).split(
            SupervisedDataset(X, y)
        )
        for split in splits:
            self.assertEqual(
                int(split.train_indices.max()) + 3, int(split.test_indices.min())
            )

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(n_splits=1)
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(gap=-1)
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(test_size=0)

    def test_invalid_max_train_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = TimeSeriesSplitter(max_train_size=0)


if __name__ == "__main__":
    _ = unittest.main()
