from dataclasses import dataclass
from typing import override

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ...dataset import SupervisedDataset
from ...defaults import DEFAULT_RANDOM_STATE
from ...parameter_validator import ParameterValidator
from ...types import CountArray, LabelArray
from ..data_split import DataSplit
from ..splitter import DatasetSplitter


@dataclass(frozen=True)
class StratifiedKFoldSplitter(DatasetSplitter):
    """
    K-fold cross-validation that preserves class proportions in each fold.

    Attributes
    ----------
    n_splits : int
        Number of folds.
    is_shuffled : bool
        Whether samples are shuffled before folding.
    random_state : int | None
        Seed for the shuffle; ignored when is_shuffled is False.
    """

    n_splits: int = 5
    is_shuffled: bool = True
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(
            n_splits=self.n_splits,
            random_state=self.random_state,
        )
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

    @override
    def split(self, dataset: SupervisedDataset) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into stratified folds.

        Parameters
        ----------
        dataset : SupervisedDataset
            The dataset to partition.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per fold.

        Raises
        ------
        ValueError
            If a class has fewer samples than folds, so that it could not occur
            in every test fold.
        """
        labels: LabelArray
        counts: CountArray
        labels, counts = np.unique(dataset.y, return_counts=True)
        rare_labels: LabelArray = labels[counts < self.n_splits]
        if rare_labels.size > 0:
            raise ValueError(
                f"each class needs at least {self.n_splits} samples, but these have fewer: {rare_labels.tolist()}"
            )
        splitter: StratifiedKFold = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.is_shuffled,
            random_state=self.random_state if self.is_shuffled else None,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(dataset.X, dataset.y)
        )
