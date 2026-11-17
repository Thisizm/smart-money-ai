import math
import unittest

from app.pro_analysis import advanced_analysis, _rsi_wilder_series, _rsi_wilder
from app.backtesting import run_research_backtest
from app.main import analyze, pearson_correlation, build_liquidation_context, _build_dominance_payload


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

    def test_rsi_series_matches_latest_wilder_rsi(self):
        closes = [100.0 + i * 0.2 + math.sin(i / 4.0) for i in range(80)]
        series = _rsi_wilder_series(closes, 14)
        self.assertEqual(len(series), len(closes))
        self.assertAlmostEqual(series[-1], _rsi_wilder(closes, 14), places=2)

    def test_market_cap_dominance_uses_market_share_fields(self):
        result = _build_dominance_payload("CoinGecko", {
            "data": {
                "market_cap_percentage": {"btc": 55.0, "eth": 12.0, "usdt": 5.0},
                "updated_at": 1_700_000_000,
                "total_market_cap": {"usd": 2_000_000_000_000},
                "total_volume": {"usd": 80_000_000_000}
            }
        })
        self.assertEqual(result["btc"], 55.0)
        self.assertEqual(result["eth"], 12.0)
        self.assertEqual(result["usdt"], 5.0)
        self.assertEqual(result["others_ex_btc_eth_usdt"], 28.0)
        self.assertIn("market-cap", result["source"])

    def test_coinglass_confluence_is_low_weight_and_directional(self):
        result = {
            "signal": "BUY", "atr": 100.0, "bull_score": 50,
            "bear_score": 40, "setup_score": 50, "analysis": []
        }
        heatmap = {
            "available": True,
            "clusters": [
                {"price": 1020.0, "intensity": 100.0, "relative_intensity": 1.0},
                {"price": 950.0, "intensity": 50.0, "relative_intensity": 0.5}
            ]
        }
        context = build_liquidation_context(result, heatmap, 1000.0)
        self.assertTrue(context["available"])
        self.assertEqual(context["score_adjustment"], 2)
        self.assertEqual(result["bull_score"], 52)
        self.assertEqual(result["setup_score"], 52)

    def test_backtest_returns_out_of_sample_report(self):
        result = run_research_backtest(sample_candles(), "1d")
        self.assertTrue(result["available"])
        self.assertIn("in_sample", result)
        self.assertIn("out_of_sample", result)
        self.assertIn("out_of_sample_2x_cost_stress", result)
        self.assertIn("walk_forward", result)
        self.assertIn("monte_carlo", result)
        self.assertEqual(result["split"]["out_of_sample_percent"], 30)

    def test_integrated_analysis_has_risk_gate(self):
        result = analyze(sample_candles(), "crypto", "BTCUSDT", "1d")
        self.assertIn(result["signal"], ("BUY", "SELL", "WAIT"))
        self.assertIn("risk_gate_passed", result["risk_management"])
        self.assertIn("advanced_analysis", result)

    def test_pearson_correlation(self):
        result = pearson_correlation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        self.assertAlmostEqual(result, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
