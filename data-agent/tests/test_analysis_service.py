import unittest
from unittest.mock import AsyncMock, patch

from app.application.services.analysis_service import (
    _deterministic_anomaly_report,
    _deterministic_prediction,
    anomaly,
    predict,
)


MONTHLY_ROWS = [
    {"month": month, "total_amount": value}
    for month, value in enumerate(
        [
            210_000_000.0,
            190_000_000.0,
            211_000_000.0,
            204_000_000.0,
            212_000_000.0,
            203_000_000.0,
            211_000_000.0,
            211_000_000.0,
            204_000_000.0,
            220_000_000.0,
            254_000_000.0,
            262_000_000.0,
        ],
        start=1,
    )
]


class AnalysisServiceDeterministicTests(unittest.IsolatedAsyncioTestCase):
    def test_monthly_prediction_crosses_year_without_month_13(self):
        rows = _deterministic_prediction(
            "2025年每月销售额趋势",
            {"type": "line", "x": "month", "y": "total_amount"},
            MONTHLY_ROWS,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["month"] for row in rows], ["2026-01", "2026-02"])
        self.assertTrue(all(row["total_amount"] > 0 for row in rows))

    def test_anomaly_report_is_non_empty_and_flags_large_spike(self):
        report = _deterministic_anomaly_report([
            {"period": 1, "amount": 100},
            {"period": 2, "amount": 102},
            {"period": 3, "amount": 101},
            {"period": 4, "amount": 1000},
        ])

        self.assertIn("确定性异常检测", report)
        self.assertIn("amount", report)
        self.assertTrue("突变" in report or "阈值" in report)

    @patch("app.application.services.analysis_service.invoke_optional_llm", new_callable=AsyncMock)
    async def test_predict_uses_fast_deterministic_path_before_llm(self, optional_llm):
        chunks = [
            chunk
            async for chunk in predict(
                "2025年每月销售额趋势",
                "SELECT month, SUM(amount) AS total_amount ...",
                {"type": "line", "x": "month", "y": "total_amount"},
                MONTHLY_ROWS,
            )
        ]

        self.assertEqual(chunks[0], {"stage": "数据预测"})
        self.assertEqual(chunks[1]["method"], "deterministic_linear_trend_v1")
        self.assertEqual(len(chunks[1]["predict"]), 2)
        optional_llm.assert_not_awaited()

    @patch("app.application.services.analysis_service.invoke_optional_llm", new_callable=AsyncMock)
    async def test_anomaly_always_returns_deterministic_result_for_numeric_data(
        self, optional_llm
    ):
        chunks = [
            chunk
            async for chunk in anomaly(
                "检查月度销售额异常",
                "SELECT month, SUM(amount) AS total_amount ...",
                {"type": "line"},
                MONTHLY_ROWS,
            )
        ]

        self.assertEqual(chunks[0], {"stage": "数据异常检测"})
        self.assertEqual(chunks[1]["method"], "deterministic_iqr_zscore_v1")
        self.assertIn("确定性异常检测", chunks[1]["anomaly"])
        self.assertNotIn("month：", chunks[1]["anomaly"])
        self.assertIn("total_amount：", chunks[1]["anomaly"])
        optional_llm.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
