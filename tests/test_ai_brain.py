import unittest

from app.ai_brain import build_prompt, validate_decision


class AIBrainValidationTests(unittest.TestCase):
    def test_prompt_requests_json_and_uses_supplied_evidence(self):
        prompt = build_prompt({"symbol": "BTCUSDT", "rsi": 54})
        self.assertIn("JSON", prompt)
        self.assertIn("BTCUSDT", prompt)
        self.assertIn("هیچ قیمت", prompt)

    def test_valid_buy_plan_passes_with_fresh_quote(self):
        result = validate_decision({
            "decision": "BUY", "summary": "سناریوی خرید",
            "entry_zone": {"low": 100, "high": 101},
            "stop_loss": 98, "targets": [104, 106],
            "setup_quality": 78
        }, live_quote={"available": True, "price": 100.5, "is_proxy": False},
           execution_quality={"blocked": False})
        self.assertEqual(result["decision"], "BUY")
        self.assertEqual(result["entry_zone"]["low"], 100)
        self.assertEqual(result["targets"], [104.0, 106.0])

    def test_wrong_direction_plan_is_downgraded_to_wait(self):
        result = validate_decision({
            "decision": "BUY", "summary": "پلن ناسازگار",
            "entry_zone": {"low": 100, "high": 101},
            "stop_loss": 102, "targets": [104, 106],
            "setup_quality": 90
        }, live_quote={"available": True, "price": 100.5, "is_proxy": False},
           execution_quality={"blocked": False})
        self.assertEqual(result["decision"], "WAIT")
        self.assertIsNone(result["entry_zone"])
        self.assertTrue(result["conflicts"])

    def test_stale_quote_blocks_directional_decision(self):
        result = validate_decision({
            "decision": "SELL", "summary": "سناریوی فروش",
            "entry_zone": {"low": 99, "high": 100},
            "stop_loss": 102, "targets": [96, 94],
            "setup_quality": 80
        }, live_quote={"available": False, "is_proxy": False},
           execution_quality={"blocked": True})
        self.assertEqual(result["decision"], "WAIT")
        self.assertEqual(result["targets"], [])

    def test_wait_does_not_expose_trade_levels(self):
        result = validate_decision({
            "decision": "WAIT", "entry_zone": {"low": 1, "high": 2},
            "stop_loss": 0.5, "targets": [3, 4]
        })
        self.assertIsNone(result["entry_zone"])
        self.assertIsNone(result["stop_loss"])
        self.assertEqual(result["targets"], [])


if __name__ == "__main__":
    unittest.main()
