from __future__ import annotations

from dataclasses import dataclass

from ..metrics import ClassificationMetric
from .model_evaluation import ModelEvaluation
from .ranked_model import RankedModel
from .text_table import TextTable


@dataclass(frozen=True)
class ModelComparison:
    """
    Candidate models ranked by one metric, all evaluated on identical splits.

    Attributes
    ----------
    metric : ClassificationMetric
        The ranking metric.
    ranking : tuple[RankedModel, ...]
        The candidates, best first.
    baseline : ModelEvaluation | None
        The reference model; None when the evaluator has no baseline.
    """

    metric: ClassificationMetric
    ranking: tuple[RankedModel, ...]
    baseline: ModelEvaluation | None

    @property
    def best(self) -> RankedModel:
        """
        Get the candidate with the best mean score.

        Returns
        -------
        RankedModel
            The best candidate.
        """
        return self.ranking[0]

    def format_table(self) -> str:
        """
        Format the ranking as a plain text table.

        Returns
        -------
        str
            The table, best candidate first, followed by the baseline if the
            ranking metric is defined on the same folds for it.
        """
        rows: list[tuple[str, str]] = [
            (entry.name, entry.score.format()) for entry in self.ranking
        ]
        if self.baseline is not None and self.baseline.get_defined_fold_indices(
            self.metric
        ) == self.best.evaluation.get_defined_fold_indices(self.metric):
            rows.append(("(baseline)", self.baseline.summarize(self.metric).format()))
        return TextTable(header=("model", self.metric.value), rows=tuple(rows)).render()
