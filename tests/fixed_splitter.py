from dataclasses import dataclass
from typing import override

from classification import DatasetSplitter, DataSplit, SupervisedDataset


@dataclass(frozen=True)
class FixedSplitter(DatasetSplitter):
    """
    Splitter that returns predetermined splits.

    Attributes
    ----------
    splits : tuple[DataSplit, ...]
        The splits to return.
    """

    splits: tuple[DataSplit, ...]

    @override
    def split(self, dataset: SupervisedDataset) -> tuple[DataSplit, ...]:
        """
        Return the predetermined splits.

        Parameters
        ----------
        dataset : SupervisedDataset
            The dataset, which is ignored.

        Returns
        -------
        tuple[DataSplit, ...]
            The predetermined splits.
        """
        return self.splits
