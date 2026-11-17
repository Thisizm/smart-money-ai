import unittest

from app.intelligence_engine import assess_hidden_intelligence, apply_intelligence_gate


class HiddenIntelligenceTests(unittest.TestCase):
    def aligned_analysis(self):
        return {
            "signal": "BUY", "market": "crypto", "price": 100.0,
            "market_structure": "bullish", "bos": "صعودی", "choch": "صعودی",
            "liquidity_sweep": "پایین", "fvg": "صعودی",
            "advanced_analysis": {
                "rsi_wilder_14": 58,
                "macd": {"histogram": 0.5},
                "adx": {"value": 28, "plus_di": 30, "minus_di": 12, "regime": "trending"},
                "momentum": {"vwap_utc_day": 99},
                "volume_profile": {"available": True, "poc": 99.5},
                "volatility": {"squeeze": False},
            },
            "rsi": 58, "strategy_plan": {"passed": True},
            "risk_management": {"risk_gate_passed": True, "risk_flags": []},
            "smart_entry": {"preferred_side": "long", "long": {"available": True}, "short": {"available": False}},
            "trade_plan": {"entry": 100, "stop_loss": 98, "targets": [103]},
            "chart_levels": {"entry": 100, "stop_loss": 98, "targets": [103]},
            "analysis": [], "wait_reason": [],
        }

    def aligned_derivatives(self):
        return {
            "funding_rate": -0.0004, "taker_buy_sell_ratio": 1.2,
            "open_interest_contracts": 100000,
            "open_interest_change_pct_5m": 1.0,
            "cvd_delta_percent": 20,
            "spot_orderbook_imbalance_top20": 0.2,
            "spot_order_book_available": True,
            "metric_status": {"open_interest": True},
        }

    def test_aligned_independent_evidence_passes(self):
        result = self.aligned_analysis()
        assessment = assess_hidden_intelligence(
            "crypto", result, self.aligned_derivatives(),
            {"available": True},
            {"news": {"available": True}, "macro": {"available": True}, "cot": {"available": False}},
            {"available": True},
        )
        self.assertTrue(assessment["gate_passed"])
        self.assertEqual(assessment["final_signal"], "BUY")
        self.assertGreaterEqual(len(assessment["aligned_evidence_groups"]), 3)

    def test_strongly_opposed_flow_blocks_entry(self):
        result = self.aligned_analysis()
        derivatives = self.aligned_derivatives()
        derivatives["cvd_delta_percent"] = -20
        derivatives["spot_orderbook_imbalance_top20"] = -0.25
        assessment = assess_hidden_intelligence("crypto", result, derivatives)
        self.assertFalse(assessment["gate_passed"])
        self.assertTrue(any("جریان معاملات" in reason for reason in assessment["blocked_reasons"]))

    def test_range_without_sweep_or_structure_event_blocks_trend_entry(self):
        result = self.aligned_analysis()
        result["bos"] = "ندارد"
        result["choch"] = "تأیید نشده"
        result["liquidity_sweep"] = "ندارد"
        result["advanced_analysis"]["adx"]["regime"] = "ranging"
        assessment = assess_hidden_intelligence("gold", result)
        self.assertFalse(assessment["gate_passed"])
        self.assertTrue(any("بازار رنج" in reason for reason in assessment["blocked_reasons"]))

    def test_missing_data_is_reported_not_fabricated(self):
        result = self.aligned_analysis()
        assessment = assess_hidden_intelligence("gold", result, {}, {}, {}, {})
        self.assertIn("دلتا معاملات/دفتر سفارش", assessment["missing_data"])
        self.assertIsNone(assessment["derivatives_snapshot"]["cvd_delta_percent"])

    def test_blocked_signal_has_no_trade_levels(self):
        result = self.aligned_analysis()
        assessment = assess_hidden_intelligence("crypto", result, {})
        apply_intelligence_gate(result, assessment)
        if not assessment["gate_passed"]:
            self.assertEqual(result["signal"], "WAIT")
            self.assertIsNone(result["trade_plan"]["entry"])
            self.assertFalse(result["risk_management"]["risk_gate_passed"])


if __name__ == "__main__":
    unittest.main()
