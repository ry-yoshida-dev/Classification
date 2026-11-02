from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..dataset import SupervisedDataset
from ..method import ClassificationMethod
from ..methods import BaselineParameters
from ..metrics import ClassificationMetric, ClassScoreKind
from ..parameter import ClassificationParameters
from ..processor import ClassificationProcessor
from ..types import LabelArray, NumericArray
from .comparison import ModelComparison
from .data_split import DataSplit
from .fold import FoldEvaluation
from .model_evaluation import ModelEvaluation
from .ranked_model import RankedModel
from .report import EvaluationReport
from .splitter import DatasetSplitter
from .splitters import StratifiedKFoldSplitter


@dataclass(frozen=True)
class ClassificationEvaluator:
    """
    Estimates generalization performance by training on splits of a dataset.

    Each fold builds a fresh processor from the parameters, so nothing carries
    over between folds and all preprocessing is fitted on training samples only.

    Attributes
    ----------
    splitter : DatasetSplitter
        The strategy that partitions the dataset.
    baseline : ClassificationParameters | None
        The reference model evaluated on the same splits; None to skip it.
    """

    splitter: DatasetSplitter = field(default_factory=StratifiedKFoldSplitter)
    baseline: ClassificationParameters | None = field(
        default_factory=BaselineParameters
    )

    def evaluate(
        self,
        parameters: ClassificationParameters,
        X: NumericArray,
        y: LabelArray,
    ) -> EvaluationReport:
        """
        Evaluate one model configuration.

        Parameters
        ----------
        parameters : ClassificationParameters
            The model configuration.
        X : NumericArray
            The feature matrix.
        y : LabelArray
            The class labels.

        Returns
        -------
        EvaluationReport
            The per-fold results and the baseline.

        Raises
        ------
        ValueError
            If the data is invalid, or if a split trains on fewer than two
            classes or tests on a class missing from its training samples.
        """
        dataset: SupervisedDataset = SupervisedDataset(X=X, y=y)
        splits: tuple[DataSplit, ...] = self.splitter.split(dataset)
        return EvaluationReport(
            model=self._evaluate_on_splits(parameters, dataset, splits),
            baseline=self._evaluate_baseline(dataset, splits),
        )

    def compare(
        self,
        candidates: Mapping[str, ClassificationParameters],
        X: NumericArray,
        y: LabelArray,
        metric: ClassificationMetric = ClassificationMetric.MACRO_F1,
    ) -> ModelComparison:
        """
        Rank model configurations evaluated on identical splits.

        Parameters
        ----------
        candidates : Mapping[str, ClassificationParameters]
            The model configurations by name.
        X : NumericArray
            The feature matrix.
        y : LabelArray
            The class labels.
        metric : ClassificationMetric
            The metric to rank by, using its mean over the test partitions.

        Returns
        -------
        ModelComparison
            The candidates, best first.

        Raises
        ------
        ValueError
            If there are no candidates, the data is invalid, or the metric is
            undefined for a candidate or defined on different folds for two
            candidates, which would make their mean scores incomparable.
        """
        if not candidates:
            raise ValueError("candidates must not be empty")
        self._validate_metric_support(candidates, metric)
        dataset: SupervisedDataset = SupervisedDataset(X=X, y=y)
        splits: tuple[DataSplit, ...] = self.splitter.split(dataset)
        evaluated: list[RankedModel] = [
            self._rank_candidate(name, parameters, dataset, splits, metric)
            for name, parameters in candidates.items()
        ]
        self._validate_shared_folds(evaluated, metric)
        ranking: list[RankedModel] = sorted(
            evaluated,
            key=lambda entry: entry.score.mean,
            reverse=metric.is_higher_better,
        )
        return ModelComparison(
            metric=metric,
            ranking=tuple(ranking),
            baseline=self._evaluate_baseline(dataset, splits),
        )

    @staticmethod
    def _validate_metric_support(
        candidates: Mapping[str, ClassificationParameters],
        metric: ClassificationMetric,
    ) -> None:
        for name, parameters in candidates.items():
            kind: ClassScoreKind = parameters.build_processor().score_kind
            if not metric.is_defined_for(kind):
                raise ValueError(
                    f"{metric.value} is undefined for candidate {name!r}, which produces {kind.value} scores"
                )

    @staticmethod
    def _validate_shared_folds(
        evaluated: list[RankedModel],
        metric: ClassificationMetric,
    ) -> None:
        reference: RankedModel = evaluated[0]
        expected: tuple[int, ...] = reference.evaluation.get_defined_fold_indices(
            metric
        )
        for entry in evaluated[1:]:
            actual: tuple[int, ...] = entry.evaluation.get_defined_fold_indices(metric)
            if actual != expected:
                raise ValueError(
                    f"{metric.value} is defined on folds {expected} for {reference.name!r} "
                    + f"but on folds {actual} for {entry.name!r}, so their means are not comparable"
                )

    def _rank_candidate(
        self,
        name: str,
        parameters: ClassificationParameters,
        dataset: SupervisedDataset,
        splits: tuple[DataSplit, ...],
        metric: ClassificationMetric,
    ) -> RankedModel:
        evaluation: ModelEvaluation = self._evaluate_on_splits(
            parameters, dataset, splits
        )
        return RankedModel(
            name=name,
            evaluation=evaluation,
            score=evaluation.summarize(metric),
        )

    def _evaluate_baseline(
        self,
        dataset: SupervisedDataset,
        splits: tuple[DataSplit, ...],
    ) -> ModelEvaluation | None:
        if self.baseline is None:
            return None
        return self._evaluate_on_splits(self.baseline, dataset, splits)

    @staticmethod
    def _evaluate_on_splits(
        parameters: ClassificationParameters,
        dataset: SupervisedDataset,
        splits: tuple[DataSplit, ...],
    ) -> ModelEvaluation:
        method: ClassificationMethod = parameters.build_processor().method
        folds: tuple[FoldEvaluation, ...] = tuple(
            ClassificationEvaluator._evaluate_fold(
                parameters, dataset, split, fold_index
            )
            for fold_index, split in enumerate(splits)
        )
        return ModelEvaluation(method=method, folds=folds)

    @staticmethod
    def _evaluate_fold(
        parameters: ClassificationParameters,
        dataset: SupervisedDataset,
        split: DataSplit,
        fold_index: int,
    ) -> FoldEvaluation:
        train: SupervisedDataset = dataset.subset(split.train_indices)
        test: SupervisedDataset = dataset.subset(split.test_indices)
        processor: ClassificationProcessor = parameters.build_processor()
        processor.fit(train.X, train.y)
        return FoldEvaluation(
            fold_index=fold_index,
            train_metrics=processor.evaluate(train.X, train.y),
            test_metrics=processor.evaluate(test.X, test.y),
        )
