import io
import unittest
from unittest.mock import patch
import urllib.error

from app.ai_brain import build_prompt, validate_decision, _request_with_retry


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


class GeminiRetryTests(unittest.TestCase):
    @patch("app.ai_brain.time.sleep")
    @patch("app.ai_brain._gemini_request")
    def test_temporary_503_is_retried_without_changing_request(self, request, sleep):
        unavailable = urllib.error.HTTPError(
            "https://example.invalid", 503, "Unavailable", {}, io.BytesIO(b"busy")
        )
        request.side_effect = [unavailable, {"ok": True}]
        result = _request_with_retry("gemini-3.8-flash", "key", b"same-body")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(request.call_count, 2)
        request.assert_any_call("gemini-3.8-flash", "key", b"same-body")
        sleep.assert_called_once()

    @patch("app.ai_brain.time.sleep")
    @patch("app.ai_brain._gemini_request")
    def test_503_stops_after_two_attempts(self, request, sleep):
        unavailable = urllib.error.HTTPError(
            "https://example.invalid", 503, "Unavailable", {}, io.BytesIO(b"busy")
        )
        request.side_effect = [unavailable, unavailable]
        with self.assertRaises(urllib.error.HTTPError):
            _request_with_retry("gemini-3.8-flash", "key", b"same-body")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(sleep.call_count, 1)

    @patch("app.ai_brain.time.sleep")
    @patch("app.ai_brain._gemini_request")
    def test_read_timeout_is_retried_once(self, request, sleep):
        import socket
        request.side_effect = [socket.timeout("read timed out"), {"ok": True}]
        result = _request_with_retry("gemini-3.8-flash", "key", b"same-body")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once()

    @patch("app.ai_brain.time.sleep")
    @patch("app.ai_brain._gemini_request")
    def test_persistent_read_timeout_has_clear_message(self, request, sleep):
        import socket
        request.side_effect = [socket.timeout("read timed out"), socket.timeout("read timed out")]
        with self.assertRaisesRegex(RuntimeError, "نشانه تمام‌شدن سهمیه رایگان نیست"):
            _request_with_retry("gemini-3.8-flash", "key", b"same-body")
        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
