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
                "available": True, "price": 100.5, "age_seconds": 2,
                "is_proxy": False, "source": "Binance Spot"
            },
            "trade_plan": {"entry": 100.0, "stop_loss": 98.0,
                           "targets": [103.0, 105.0, 107.0]},
            "risk_management": {"risk_gate_passed": True},
            "execution_quality": {"blocked": False},
            "economic_event_risk": {"entry_blocked_by_calendar": False},
            "smart_entry": {
                "preferred_side": "long", "risk_gate_passed": True,
                "long": {
                    "available": True, "status": "WAIT_FOR_PULLBACK_CONFIRMATION",
                    "zone": {"low": 99.8, "high": 100.2},
                    "entry_reference": 100.0, "stop_loss": 98.0,
                    "targets": [103.0, 105.0, 107.0]
                }
            }
        }

    def test_valid_signal_is_recorded_once(self):
        first = paper_trading.record_signal(self.sample_signal())
        second = paper_trading.record_signal(self.sample_signal())
        self.assertTrue(first["recorded"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(paper_trading.list_paper_trades()["stats"]["pending_entry"], 1)

    def test_generic_signal_without_smart_entry_zone_is_not_recorded(self):
        signal = self.sample_signal()
        signal.pop("smart_entry")
        result = paper_trading.record_signal(signal)
        self.assertFalse(result["recorded"])
        self.assertEqual(result["reason"], "smart_entry_not_available")
        self.assertEqual(paper_trading.list_paper_trades()["stats"]["pending_entry"], 0)

    def test_primary_analysis_entry_has_its_own_pending_trade(self):
        result = paper_trading.record_signal(self.sample_signal(), entry_mode="primary")
        self.assertTrue(result["recorded"])
        trade = paper_trading.list_paper_trades()["trades"][0]
        self.assertEqual(trade["source"], "PRIMARY_ANALYSIS_ENTRY")
        self.assertAlmostEqual(trade["entry_low"], 99.9)
        self.assertAlmostEqual(trade["entry_high"], 100.1)

    def test_primary_and_smart_entries_are_independent(self):
        signal = self.sample_signal()
        smart = paper_trading.record_signal(signal, entry_mode="smart")
        primary = paper_trading.record_signal(signal, entry_mode="primary")
        self.assertTrue(smart["recorded"])
        self.assertTrue(primary["recorded"])
        self.assertEqual(paper_trading.list_paper_trades()["stats"]["pending_entry"], 2)

    def test_canceled_entry_can_be_armed_again(self):
        first = paper_trading.record_signal(self.sample_signal(), entry_mode="primary")
        self.assertTrue(first["recorded"])
        self.assertTrue(paper_trading.close_paper_trade(first["trade_id"])["closed"])
        second = paper_trading.record_signal(self.sample_signal(), entry_mode="primary")
        self.assertTrue(second["recorded"])
        self.assertNotEqual(first["trade_id"], second["trade_id"])

    def test_smart_entry_structural_zone_is_the_recorded_entry(self):
        result = paper_trading.record_signal(self.sample_signal())
        self.assertTrue(result["recorded"])
        trade = paper_trading.list_paper_trades()["trades"][0]
        self.assertEqual(trade["source"], "SMART_ENTRY_STRUCTURAL_ZONE")
        self.assertAlmostEqual(trade["entry_low"], 99.8)
        self.assertAlmostEqual(trade["entry_high"], 100.2)

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
            lambda market, symbol: {"available": True, "price": 100.0,
                                    "age_seconds": 1, "is_proxy": False}
        )
        self.assertEqual(paper_trading.list_paper_trades()["trades"][0]["status"], "ACTIVE")
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


    def test_virtual_position_has_auto_leverage_and_live_pnl(self):
        paper_trading.record_signal(self.sample_signal())
        paper_trading.monitor_open_trades(
            lambda market, symbol: {"available": True, "price": 100.0,
                                    "age_seconds": 1, "is_proxy": False}
        )
        trade = paper_trading.list_paper_trades()["trades"][0]
        self.assertEqual(trade["status"], "ACTIVE")
        self.assertGreater(trade["quantity"], 0)
        self.assertGreaterEqual(trade["leverage"], 1)
        self.assertLessEqual(trade["leverage"], 5)
        self.assertGreater(trade["margin_used"], 0)
        paper_trading.monitor_open_trades(
            lambda market, symbol: {"available": True, "price": 101.0,
                                    "age_seconds": 1, "is_proxy": False}
        )
        trade = paper_trading.list_paper_trades()["trades"][0]
        self.assertGreater(trade["unrealized_pnl"], 0)
        self.assertGreater(trade["total_pnl"], 0)

    def test_manual_close_active_trade_records_exit_and_pnl(self):
        paper_trading.record_signal(self.sample_signal())
        paper_trading.monitor_open_trades(
            lambda market, symbol: {"available": True, "price": 100.0,
                                    "age_seconds": 1, "is_proxy": False}
        )
        result = paper_trading.close_paper_trade(1, exit_price=101.0)
        self.assertTrue(result["closed"])
        trade = paper_trading.list_paper_trades()["trades"][0]
        self.assertEqual(trade["status"], "MANUAL_CLOSED")
        self.assertAlmostEqual(trade["exit_price"], 101.0)
        self.assertGreater(trade["realized_r"], 0)
        self.assertEqual(paper_trading.list_paper_trades()["stats"]["closed"], 1)

    def test_manual_cancel_pending_entry_is_not_counted_as_win_or_loss(self):
        paper_trading.record_signal(self.sample_signal())
        result = paper_trading.close_paper_trade(1)
        self.assertTrue(result["closed"])
        data = paper_trading.list_paper_trades()
        self.assertEqual(data["trades"][0]["status"], "MANUAL_CLOSED")
        self.assertEqual(data["stats"]["closed"], 1)
        self.assertIsNone(data["stats"]["win_rate_percent"])
        self.assertEqual(data["stats"]["net_r"], 0)


    def test_active_and_closed_trades_are_separated(self):
        paper_trading.record_signal(self.sample_signal())
        active_data = paper_trading.list_paper_trades()
        self.assertEqual(len(active_data["active_trades"]), 1)
        self.assertEqual(len(active_data["history_trades"]), 0)
        paper_trading.close_paper_trade(1)
        closed_data = paper_trading.list_paper_trades()
        self.assertEqual(len(closed_data["active_trades"]), 0)
        self.assertEqual(len(closed_data["history_trades"]), 1)
        self.assertEqual(closed_data["history_trades"][0]["status"], "MANUAL_CLOSED")

    def test_all_active_trades_are_returned_even_when_history_limit_is_small(self):
        signal = self.sample_signal()
        paper_trading.record_signal(signal)
        signal2 = self.sample_signal()
        signal2["symbol"] = "ETHUSDT"
        signal2["last_candle_time"] += 900000
        paper_trading.record_signal(signal2)
        data = paper_trading.list_paper_trades(limit=1)
        self.assertEqual(len(data["active_trades"]), 2)
        self.assertEqual(data["stats"]["pending_entry"], 2)

    def test_trade_remains_after_database_reinitialization(self):
        paper_trading.record_signal(self.sample_signal())
        paper_trading.init_paper_db()
        data = paper_trading.list_paper_trades()
        self.assertEqual(len(data["active_trades"]), 1)
        self.assertEqual(data["active_trades"][0]["symbol"], "BTCUSDT")

    def test_stop_loss_closes_trade_as_loss(self):
        paper_trading.record_signal(self.sample_signal())
        paper_trading.monitor_open_trades(
            lambda market, symbol: {"available": True, "price": 100.0,
                                    "age_seconds": 1, "is_proxy": False}
        )
        paper_trading.monitor_open_trades(
            lambda market, symbol: {"available": True, "price": 97.9,
                                    "age_seconds": 1, "is_proxy": False}
        )
        trade = paper_trading.list_paper_trades()["trades"][0]
        self.assertEqual(trade["status"], "SL_HIT")
        self.assertAlmostEqual(trade["realized_r"], -1.0)


if __name__ == "__main__":
    unittest.main()
