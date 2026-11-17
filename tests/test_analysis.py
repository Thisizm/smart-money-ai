import math
import unittest

from app.pro_analysis import advanced_analysis, _rsi_wilder_series, _rsi_wilder
from app.backtesting import run_research_backtest
from app.main import analyze, pearson_correlation, build_liquidation_context, _build_dominance_payload, normalize_marginpad_liquidation_data


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

    def test_marginpad_confluence_is_low_weight_and_directional(self):
        result = {
            "signal": "BUY", "atr": 100.0, "bull_score": 50,
            "bear_score": 40, "setup_score": 50, "analysis": []
        }
        heatmap = {
            "available": True,
            "clusters": [
                {"price": 1020.0, "side": "short", "est_notional": 100.0, "intensity": 100.0, "relative_intensity": 1.0},
                {"price": 950.0, "side": "long", "est_notional": 50.0, "intensity": 50.0, "relative_intensity": 0.5}
            ]
        }
        context = build_liquidation_context(result, heatmap, 1000.0)
        self.assertTrue(context["available"])
        self.assertEqual(context["score_adjustment"], 1)
        self.assertEqual(result["bull_score"], 51)
        self.assertEqual(result["setup_score"], 51)

    def test_marginpad_keeps_model_clusters_separate_from_observed_events(self):
        data = normalize_marginpad_liquidation_data(
            {
                "updatedAt": "2026-11-17T12:00:00Z",
                "clusters": [
                    {"price": "1020", "side": "short", "est_notional": "250000"},
                    {"price": "950", "side": "long", "est_notional": "125000"}
                ]
            },
            {
                "events": [
                    {"ts": 1794916800, "exchange": "Binance", "symbol": "BTCUSDT",
                     "side": "short_liquidated", "price": "1010", "qty": "0.5", "notional": "505"},
                    {"ts": 1794916810, "exchange": "Bybit", "symbol": "BTCUSDT",
                     "side": "long_liquidated", "price": "990", "qty": "0.2", "notional": "198"}
                ]
            },
            "BTCUSDT"
        )
        self.assertTrue(data["available"])
        self.assertEqual(len(data["model_clusters"]), 2)
        self.assertEqual(len(data["events"]), 2)
        self.assertEqual(data["model_clusters"][0]["kind"], "model_estimate")
        self.assertEqual(data["events"][0]["kind"], "observed_event")
        self.assertEqual(data["event_summary"]["short_liquidated_count"], 1)
        self.assertEqual(data["event_summary"]["long_liquidated_count"], 1)

    def test_backtest_returns_out_of_sample_report(self):
        result = run_research_backtest(sample_candles(), "1d")
        self.assertTrue(result["available"])
        self.assertIn("in_sample", result)
        self.assertIn("out_of_sample", result)
        self.assertIn("out_of_sample_2x_cost_stress", result)
        self.assertIn("walk_forward", result)
        self.assertIn("monte_carlo", result)
        self.assertEqual(result["split"]["out_of_sample_percent"], 30)

    def test_production_engine_backtest_is_explicitly_labelled(self):
        result = run_research_backtest(
            sample_candles(), "1d",
            signal_engine=lambda history: {"signal": "WAIT", "trade_plan": {}}
        )
        self.assertTrue(result["available"])
        self.assertIn("Production analyze()", result["strategy"])
        self.assertTrue(result["out_of_sample"]["trades"] >= 0)

    def test_integrated_analysis_has_risk_gate(self):
        result = analyze(sample_candles(), "crypto", "BTCUSDT", "1d")
        self.assertIn(result["signal"], ("BUY", "SELL", "WAIT"))
        self.assertIn("risk_gate_passed", result["risk_management"])
        self.assertIn("advanced_analysis", result)

    def test_smart_entry_plan_is_returned_for_both_directions(self):
        result = analyze(sample_candles(), "crypto", "BTCUSDT", "1d")
        self.assertIn("smart_entry", result)
        self.assertIn("long", result["smart_entry"])
        self.assertIn("short", result["smart_entry"])
        self.assertIn(result["smart_entry"]["preferred_side"], ("long", "short", "neutral"))
        for side in ("long", "short"):
            plan = result["smart_entry"][side]
            self.assertIn("available", plan)
            if plan["available"]:
                self.assertLessEqual(plan["zone"]["low"], plan["zone"]["high"])
                self.assertGreater(plan["stop_loss"], 0)
                self.assertEqual(len(plan["targets"]), 3)
                self.assertIn(plan["status"], (
                    "CONFIRMED_ON_CLOSED_CANDLE", "WAIT_FOR_PULLBACK_CONFIRMATION",
                    "WAIT_FOR_LIVE_PRICE_CONFIRMATION", "BLOCKED_BY_ECONOMIC_CALENDAR"
                ))

    def test_pearson_correlation(self):
        result = pearson_correlation([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        self.assertAlmostEqual(result, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
