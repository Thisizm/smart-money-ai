import math
import unittest

from app.pro_analysis import advanced_analysis
from app.backtesting import run_research_backtest


def sample_candles(count=320):
    candles = []
    previous = 100.0
    for i in range(count):
        close = 100.0 + i * 0.035 + math.sin(i / 8.0) * 1.8
        open_price = previous
        high = max(open_price, close) + 0.45
        low = min(open_price, close) - 0.45
        candles.append({
            "time": 1_700_000_000_000 + i * 86_400_000,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0 + (i % 17) * 25.0
        })
        previous = close
    return candles


class AdvancedAnalysisTests(unittest.TestCase):
    def test_advanced_features_are_returned(self):
        result = advanced_analysis(sample_candles(), "1d")
        self.assertIn("rsi_wilder_14", result)
        self.assertIn("adx", result)
        self.assertIn("volatility", result)
        self.assertIn("volume_profile", result)
        self.assertIn("liquidity_map", result)
        self.assertIn("risk_template", result)
        self.assertTrue(0 <= result["rsi_wilder_14"] <= 100)

    def test_backtest_returns_out_of_sample_report(self):
        result = run_research_backtest(sample_candles(), "1d")
        self.assertTrue(result["available"])
        self.assertIn("in_sample", result)
        self.assertIn("out_of_sample", result)
        self.assertIn("out_of_sample_2x_cost_stress", result)
        self.assertEqual(result["split"]["out_of_sample_percent"], 30)


if __name__ == "__main__":
    unittest.main()
