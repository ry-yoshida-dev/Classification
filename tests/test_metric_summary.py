from __future__ import annotations

import math
import unittest

from classification import (
    MetricSummary,
)


class TestMetricSummary(unittest.TestCase):
    def test_from_values(self) -> None:
        summary: MetricSummary = MetricSummary.from_values([1.0, 3.0])
        self.assertEqual(summary.mean, 2.0)
        self.assertAlmostEqual(summary.std or 0.0, math.sqrt(2.0))
        self.assertEqual((summary.minimum, summary.maximum), (1.0, 3.0))
        self.assertEqual(summary.format(), "2.0000 ± 1.4142")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = MetricSummary.from_values([])

    def test_format_without_variability(self) -> None:
        self.assertEqual(MetricSummary.from_values([0.5]).format(), "0.5000")


if __name__ == "__main__":
    _ = unittest.main()
