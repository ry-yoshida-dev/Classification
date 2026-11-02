from __future__ import annotations

from abc import ABC, abstractmethod

from ..dataset import SupervisedDataset
from .data_split import DataSplit


class DatasetSplitter(ABC):
    """
    Base class for strategies that partition a dataset for evaluation.
    """

    @abstractmethod
    def split(self, dataset: SupervisedDataset) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into train/test splits.

        Parameters
        ----------
        dataset : SupervisedDataset
            The dataset to partition.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per evaluation fold.
        """
