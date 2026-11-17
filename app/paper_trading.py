"""Persistent paper-trading tracker for signals emitted by the analysis engine.

This module never submits orders to an exchange. It records simulated entries and
tracks exits from fresh public quotes only.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

DB_PATH = os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "alerts.db"))
_LOCK = threading.RLock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.row_factory = sqlite3.Row
    return con


def init_paper_db():
    with _LOCK, _connect() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            side TEXT NOT NULL,
            signal_candle_time INTEGER,
            entry_price REAL NOT NULL,
            signal_entry REAL,
            stop_loss REAL NOT NULL,
            target_1 REAL NOT NULL,
            target_2 REAL,
            target_3 REAL,
            status TEXT NOT NULL,
            tp1_hit_at TEXT,
            closed_at TEXT,
            exit_price REAL,
            exit_reason TEXT,
            realized_r REAL NOT NULL DEFAULT 0,
            realized_return_percent REAL NOT NULL DEFAULT 0,
            source TEXT,
            note TEXT
        )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_paper_status ON paper_trades(status)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_paper_created ON paper_trades(created_at DESC)")


def record_signal(result):
    """Open a simulated position only for a fresh, executable BUY/SELL signal."""
    if not isinstance(result, dict) or result.get("signal") not in ("BUY", "SELL"):
        return {"recorded": False, "reason": "no_directional_signal"}
    if (result.get("risk_management") or {}).get("risk_gate_passed") is False:
        return {"recorded": False, "reason": "risk_gate_blocked"}
    if (result.get("execution_quality") or {}).get("blocked"):
        return {"recorded": False, "reason": "execution_quality_blocked"}
    if (result.get("economic_event_risk") or {}).get("entry_blocked_by_calendar"):
        return {"recorded": False, "reason": "economic_calendar_blocked"}

    quote = result.get("live_quote") or {}
    if not quote.get("available") or quote.get("is_proxy"):
        return {"recorded": False, "reason": "fresh_non_proxy_quote_required"}
    age = quote.get("age_seconds")
    market = result.get("market", "crypto")
    if age is None or age > (60 if market == "crypto" else 900):
        return {"recorded": False, "reason": "stale_quote"}
    try:
        live_entry = float(quote["price"])
        signal_entry = float(result.get("entry_price") or (result.get("trade_plan") or {}).get("entry"))
        stop = float((result.get("trade_plan") or {})["stop_loss"])
        targets = [float(x) for x in (result.get("trade_plan") or {}).get("targets", [])]
        if len(targets) < 2:
            raise ValueError("two_targets_required")
    except (TypeError, ValueError, KeyError):
        return {"recorded": False, "reason": "incomplete_trade_plan"}

    side = "LONG" if result["signal"] == "BUY" else "SHORT"
    risk = live_entry - stop if side == "LONG" else stop - live_entry
    if risk <= 0:
        return {"recorded": False, "reason": "live_price_beyond_stop"}
    atr = float(result.get("atr") or 0)
    max_drift = max(atr * 0.35, abs(signal_entry) * 0.001)
    if abs(live_entry - signal_entry) > max_drift:
        return {"recorded": False, "reason": "price_drifted_from_signal"}
    if side == "LONG" and (targets[0] <= live_entry or live_entry <= stop):
        return {"recorded": False, "reason": "invalid_long_levels"}
    if side == "SHORT" and (targets[0] >= live_entry or live_entry >= stop):
        return {"recorded": False, "reason": "invalid_short_levels"}

    candle_time = result.get("last_candle_time")
    fingerprint = "|".join(str(x) for x in (
        market, result.get("symbol", ""), result.get("timeframe", ""),
        candle_time or result.get("updated_at", ""), result["signal"]
    ))
    now = _now()
    note = "ورود آزمایشی با قیمت زنده هنگام صدور سیگنال؛ بدون سفارش واقعی."
    with _LOCK, _connect() as con:
        cur = con.execute("""
            INSERT OR IGNORE INTO paper_trades
            (fingerprint, created_at, updated_at, market, symbol, timeframe, side,
             signal_candle_time, entry_price, signal_entry, stop_loss, target_1,
             target_2, target_3, status, source, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
        """, (
            fingerprint, now, now, market, result.get("symbol", ""),
            result.get("timeframe", ""), side, candle_time, live_entry,
            signal_entry, stop, targets[0], targets[1],
            targets[2] if len(targets) > 2 else None,
            quote.get("source"), note
        ))
        if cur.rowcount == 0:
            row = con.execute("SELECT id, status FROM paper_trades WHERE fingerprint=?", (fingerprint,)).fetchone()
            return {"recorded": False, "duplicate": True, "trade_id": row["id"], "status": row["status"]}
        return {"recorded": True, "trade_id": cur.lastrowid, "status": "ACTIVE", "entry_price": live_entry}


def monitor_open_trades(quote_getter):
    """Update open paper trades from fresh quotes; partial close 50% at TP1, rest at TP2."""
    with _LOCK, _connect() as con:
        rows = con.execute("SELECT * FROM paper_trades WHERE status IN ('ACTIVE','TP1_HIT') ORDER BY id").fetchall()
    updated = 0
    for row in rows:
        try:
            quote = quote_getter(row["market"], row["symbol"])
            if not quote.get("available") or quote.get("is_proxy"):
                continue
            age = quote.get("age_seconds")
            if age is None or age > (60 if row["market"] == "crypto" else 900):
                continue
            price = float(quote["price"])
            side = row["side"]
            stop = float(row["stop_loss"])
            tp1 = float(row["target_1"])
            tp2 = float(row["target_2"]) if row["target_2"] is not None else None
            entry = float(row["entry_price"])
            risk = entry - stop if side == "LONG" else stop - entry
            if risk <= 0:
                continue

            status = row["status"]
            realized_r = float(row["realized_r"] or 0)
            realized_pct = float(row["realized_return_percent"] or 0)
            closed_at = None
            exit_price = None
            exit_reason = None
            tp1_at = row["tp1_hit_at"]

            stop_hit = price <= stop if side == "LONG" else price >= stop
            tp1_hit = price >= tp1 if side == "LONG" else price <= tp1
            tp2_hit = tp2 is not None and (price >= tp2 if side == "LONG" else price <= tp2)

            def signed_return(level):
                return ((level - entry) / entry * 100.0) if side == "LONG" else ((entry - level) / entry * 100.0)

            def signed_r(level):
                return ((level - entry) / risk) if side == "LONG" else ((entry - level) / risk)

            if status == "ACTIVE":
                # If both stop and TP are crossed between observations, choose the stop (conservative).
                if stop_hit:
                    realized_r = -1.0
                    realized_pct = signed_return(stop)
                    status, exit_price, exit_reason, closed_at = "SL_HIT", stop, "STOP_LOSS", _now()
                elif tp2_hit:
                    realized_r = 0.5 * signed_r(tp1) + 0.5 * signed_r(tp2)
                    realized_pct = 0.5 * signed_return(tp1) + 0.5 * signed_return(tp2)
                    status, exit_price, exit_reason, closed_at, tp1_at = "TP2_HIT", tp2, "TP1_AND_TP2", _now(), _now()
                elif tp1_hit:
                    realized_r = 0.5 * signed_r(tp1)
                    realized_pct = 0.5 * signed_return(tp1)
                    status, tp1_at = "TP1_HIT", _now()
            else:  # TP1_HIT: half the position remains open
                if stop_hit:
                    realized_r += 0.5 * signed_r(stop)
                    realized_pct += 0.5 * signed_return(stop)
                    status, exit_price, exit_reason, closed_at = "SL_AFTER_TP1", stop, "TP1_THEN_STOP", _now()
                elif tp2_hit:
                    realized_r += 0.5 * signed_r(tp2)
                    realized_pct += 0.5 * signed_return(tp2)
                    status, exit_price, exit_reason, closed_at = "TP2_HIT", tp2, "TP1_AND_TP2", _now()

            if status != row["status"] or closed_at:
                with _LOCK, _connect() as con:
                    con.execute("""
                        UPDATE paper_trades SET updated_at=?, status=?, tp1_hit_at=?,
                        closed_at=?, exit_price=?, exit_reason=?, realized_r=?,
                        realized_return_percent=? WHERE id=?
                    """, ( _now(), status, tp1_at, closed_at, exit_price,
                           exit_reason, realized_r, realized_pct, row["id"]))
                updated += 1
        except Exception:
            continue
    return {"checked": len(rows), "updated": updated}


def list_paper_trades(limit=100):
    limit = max(1, min(int(limit), 300))
    with _LOCK, _connect() as con:
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]
        stats = con.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN status IN ('ACTIVE','TP1_HIT') THEN 1 ELSE 0 END) AS active,
                   SUM(CASE WHEN status IN ('TP1_HIT','TP2_HIT','SL_HIT','SL_AFTER_TP1') THEN 1 ELSE 0 END) AS tp1_touched,
                   SUM(CASE WHEN status='TP2_HIT' THEN 1 ELSE 0 END) AS tp2,
                   SUM(CASE WHEN status IN ('SL_HIT','SL_AFTER_TP1') THEN 1 ELSE 0 END) AS stopped,
                   SUM(CASE WHEN status IN ('TP2_HIT','SL_HIT','SL_AFTER_TP1') THEN 1 ELSE 0 END) AS closed
            FROM paper_trades
        """).fetchone()
        closed_rows = con.execute("""
            SELECT realized_r FROM paper_trades
            WHERE status IN ('TP2_HIT','SL_HIT','SL_AFTER_TP1')
        """).fetchall()
    closed_r = [float(r["realized_r"] or 0) for r in closed_rows]
    wins = sum(1 for x in closed_r if x > 0)
    return {
        "available": True, "mode": "PAPER_ONLY",
        "stats": {
            "total": int(stats["total"] or 0), "active": int(stats["active"] or 0),
            "tp1_touched": int(stats["tp1_touched"] or 0), "tp2": int(stats["tp2"] or 0),
            "stopped": int(stats["stopped"] or 0), "closed": len(closed_r),
            "win_rate_percent": round(wins / len(closed_r) * 100, 2) if closed_r else None,
            "net_r": round(sum(closed_r), 3),
        },
        "trades": rows
    }
