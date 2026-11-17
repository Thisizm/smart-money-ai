import os
import tempfile
import unittest
from unittest.mock import patch

from app import paper_trading


class PaperTradingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp.name, "paper.sqlite3")
        self.db_patch = patch.object(paper_trading, "DB_PATH", self.db_path)
        self.db_patch.start()
        paper_trading.init_paper_db()

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    def sample_signal(self):
        return {
            "signal": "BUY", "market": "crypto", "symbol": "BTCUSDT",
            "timeframe": "15m", "last_candle_time": 1700000000000,
            "updated_at": "2026-11-17T12:00:00+00:00", "entry_price": 100.0,
            "atr": 2.0, "live_quote": {
                "available": True, "price": 100.1, "age_seconds": 2,
                "is_proxy": False, "source": "Binance Spot"
            },
            "trade_plan": {"entry": 100.0, "stop_loss": 98.0,
                           "targets": [103.0, 105.0, 107.0]},
            "risk_management": {"risk_gate_passed": True},
            "execution_quality": {"blocked": False},
            "economic_event_risk": {"entry_blocked_by_calendar": False}
        }

    def test_valid_signal_is_recorded_once(self):
        first = paper_trading.record_signal(self.sample_signal())
        second = paper_trading.record_signal(self.sample_signal())
        self.assertTrue(first["recorded"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(paper_trading.list_paper_trades()["stats"]["active"], 1)

    def test_stale_or_proxy_quote_is_not_recorded(self):
        signal = self.sample_signal()
        signal["live_quote"]["age_seconds"] = 999
        self.assertFalse(paper_trading.record_signal(signal)["recorded"])
        signal = self.sample_signal()
        signal["live_quote"]["is_proxy"] = True
        self.assertFalse(paper_trading.record_signal(signal)["recorded"])

    def test_tp1_partial_then_tp2_closes_trade(self):
        paper_trading.record_signal(self.sample_signal())
        paper_trading.monitor_open_trades(
            lambda market, symbol: {"available": True, "price": 103.2,
                                    "age_seconds": 1, "is_proxy": False}
        )
        self.assertEqual(paper_trading.list_paper_trades()["trades"][0]["status"], "TP1_HIT")
        paper_trading.monitor_open_trades(
            lambda market, symbol: {"available": True, "price": 105.2,
                                    "age_seconds": 1, "is_proxy": False}
        )
        trade = paper_trading.list_paper_trades()["trades"][0]
        self.assertEqual(trade["status"], "TP2_HIT")
        self.assertGreater(trade["realized_r"], 0)

    def test_stop_loss_closes_trade_as_loss(self):
        paper_trading.record_signal(self.sample_signal())
        paper_trading.monitor_open_trades(
            lambda market, symbol: {"available": True, "price": 97.9,
                                    "age_seconds": 1, "is_proxy": False}
        )
        trade = paper_trading.list_paper_trades()["trades"][0]
        self.assertEqual(trade["status"], "SL_HIT")
        self.assertAlmostEqual(trade["realized_r"], -1.0)


if __name__ == "__main__":
    unittest.main()
