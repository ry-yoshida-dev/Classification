from dataclasses import dataclass

from ..types import IndexArray


@dataclass(frozen=True, eq=False)
class DataSplit:
    """
    Sample indices of one train/test partition.

    Attributes
    ----------
    train_indices : IndexArray
        Indices of the training samples.
    test_indices : IndexArray
        Indices of the test samples.
    """

    train_indices: IndexArray
    test_indices: IndexArray
