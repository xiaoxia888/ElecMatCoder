from __future__ import annotations

import unittest

import pandas as pd

from apps.trainer.routing_confidence_analyzer.analyze import (
    build_recommendations,
    build_strategy_comparison,
    build_threshold_simulation,
    normalize_code,
)


class RoutingConfidenceAnalyzerTest(unittest.TestCase):
    def test_normalize_code(self):
        self.assertEqual(normalize_code(" ab 12\n"), "AB12")

    def test_intersection_improves_safe_acceptance(self):
        detail = pd.DataFrame({
            "分析_编码是否正确": [True, False, True, False],
            "分析_模型置信分": [0.999, 0.8, 0.999, 0.2],
            "分析_分流类别": ["简单", "简单", "困难", "困难"],
        })
        simulation = build_threshold_simulation(detail)
        row = simulation.loc[
            simulation["策略"].eq("分流+置信分交集")
            & simulation["置信分阈值"].eq(0.99)
        ].iloc[0]
        self.assertEqual(row["自动放行数"], 1)
        self.assertEqual(row["自动放行准确率"], 1.0)

        recommendations = build_recommendations(simulation, (1.0,))
        combined = recommendations.loc[recommendations["策略"].eq("分流+置信分交集")].iloc[0]
        self.assertTrue(combined["是否可达到"])

        comparison = build_strategy_comparison(detail, 0.99).set_index("策略")
        self.assertEqual(comparison.loc["仅分流", "自动放行错误数"], 1)
        self.assertEqual(comparison.loc["仅模型分", "自动放行数"], 2)
        self.assertEqual(comparison.loc["分流+模型分交集", "自动放行错误数"], 0)


if __name__ == "__main__":
    unittest.main()
