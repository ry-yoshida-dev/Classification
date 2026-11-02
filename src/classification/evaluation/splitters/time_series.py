from dataclasses import dataclass
from typing import override

from sklearn.model_selection import TimeSeriesSplit

from ...dataset import SupervisedDataset
from ...parameter_validator import ParameterValidator
from ..data_split import DataSplit
from ..splitter import DatasetSplitter


@dataclass(frozen=True)
class TimeSeriesSplitter(DatasetSplitter):
    """
    Forward-chaining folds that always test on samples later than the training.

    Samples must already be in temporal order.

    Attributes
    ----------
    n_splits : int
        Number of folds.
    max_train_size : int | None
        Maximum number of training samples; unlimited when None.
    test_size : int | None
        Number of test samples per fold; chosen from n_splits when None.
    gap : int
        Number of samples excluded between training and test.
    """

    n_splits: int = 5
    max_train_size: int | None = None
    test_size: int | None = None
    gap: int = 0

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            gap=self.gap,
        )
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.max_train_size is not None and self.max_train_size <= 0:
            raise ValueError("max_train_size must be greater than 0 when set")
        if self.test_size is not None and self.test_size <= 0:
            raise ValueError("test_size must be greater than 0 when set")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")

    @override
    def split(self, dataset: SupervisedDataset) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into forward-chaining folds.

        Parameters
        ----------
        dataset : SupervisedDataset
            The dataset to partition.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per fold, oldest first.
        """
        splitter: TimeSeriesSplit = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            gap=self.gap,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(dataset.X)
        )
