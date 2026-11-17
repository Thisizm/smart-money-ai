"""Persistent paper-trading tracker for signals emitted by the analysis engine.

This module never submits orders to an exchange. It records simulated entries and
tracks exits from fresh public quotes only.
"""
import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone

def _database_path():
    configured = os.getenv("DATABASE_PATH")
    if configured:
        return configured
    root_dir = os.path.dirname(os.path.dirname(__file__))
    persistent_dir = os.getenv("PERSISTENT_DATA_DIR", "/var/data")
    if os.path.isdir(persistent_dir) and os.access(persistent_dir, os.W_OK):
        persistent_path = os.path.join(persistent_dir, "alerts.db")
        legacy_path = os.path.join(root_dir, "alerts.db")
        if not os.path.exists(persistent_path) and os.path.exists(legacy_path):
            try:
                import shutil
                shutil.copy2(legacy_path, persistent_path)
            except OSError:
                pass
        return persistent_path
    return os.path.join(root_dir, "alerts.db")


DB_PATH = _database_path()
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
            entry_low REAL,
            entry_high REAL,
            signal_entry REAL,
            last_quote_price REAL,
            last_quote_at TEXT,
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
            note TEXT,
            quantity REAL NOT NULL DEFAULT 0,
            notional REAL NOT NULL DEFAULT 0,
            margin_used REAL NOT NULL DEFAULT 0,
            leverage REAL NOT NULL DEFAULT 1,
            initial_balance REAL NOT NULL DEFAULT 10000,
            risk_cash REAL NOT NULL DEFAULT 50
        )
        """)
        columns = {row["name"] for row in con.execute("PRAGMA table_info(paper_trades)")}
        for name in ("entry_low", "entry_high", "last_quote_price"):
            if name not in columns:
                con.execute("ALTER TABLE paper_trades ADD COLUMN " + name + " REAL")
        if "last_quote_at" not in columns:
            con.execute("ALTER TABLE paper_trades ADD COLUMN last_quote_at TEXT")
        for name, declaration in (("quantity", "REAL NOT NULL DEFAULT 0"),
                                  ("notional", "REAL NOT NULL DEFAULT 0"),
                                  ("margin_used", "REAL NOT NULL DEFAULT 0"),
                                  ("leverage", "REAL NOT NULL DEFAULT 1"),
                                  ("initial_balance", "REAL NOT NULL DEFAULT 10000"),
                                  ("risk_cash", "REAL NOT NULL DEFAULT 50")):
            if name not in columns:
                con.execute("ALTER TABLE paper_trades ADD COLUMN " + name + " " + declaration)
        con.execute("CREATE INDEX IF NOT EXISTS idx_paper_status ON paper_trades(status)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_paper_created ON paper_trades(created_at DESC)")


def record_signal(result):
    """Create a pending simulated limit entry from the signal structural zone."""
    if not isinstance(result, dict) or result.get("signal") not in ("BUY", "SELL"):
        return {"recorded": False, "reason": "no_directional_signal"}
    if (result.get("risk_management") or {}).get("risk_gate_passed") is False:
        return {"recorded": False, "reason": "risk_gate_blocked"}
    if (result.get("execution_quality") or {}).get("blocked"):
        return {"recorded": False, "reason": "execution_quality_blocked"}
    if (result.get("economic_event_risk") or {}).get("entry_blocked_by_calendar"):
        return {"recorded": False, "reason": "economic_calendar_blocked"}
    quote = result.get("live_quote") or {}
    market = result.get("market", "crypto")
    if not quote.get("available") or quote.get("is_proxy"):
        return {"recorded": False, "reason": "fresh_non_proxy_quote_required"}
    age = quote.get("age_seconds")
    if age is None or age > (60 if market == "crypto" else 900):
        return {"recorded": False, "reason": "stale_quote"}
    side = "LONG" if result["signal"] == "BUY" else "SHORT"
    smart = result.get("smart_entry") or {}
    smart_plan = smart.get("long" if side == "LONG" else "short") or {}
    base_plan = result.get("trade_plan") or {}
    try:
        live_price = float(quote["price"])
        zone = smart_plan.get("zone") if smart_plan.get("available") else None
        if zone and zone.get("low") is not None and zone.get("high") is not None:
            entry_low, entry_high = sorted((float(zone["low"]), float(zone["high"])))
            entry = float(smart_plan.get("entry_reference") or ((entry_low + entry_high) / 2))
            stop = float(smart_plan["stop_loss"])
            targets = [float(x) for x in smart_plan.get("targets", [])]
            source = "SMART_ENTRY_STRUCTURAL_ZONE"
        else:
            entry = float(base_plan.get("entry") or result.get("entry_price"))
            tolerance = max(abs(entry) * 0.0005, float(result.get("atr") or 0) * 0.05)
            entry_low, entry_high = entry - tolerance, entry + tolerance
            stop = float(base_plan["stop_loss"])
            targets = [float(x) for x in base_plan.get("targets", [])]
            source = "SIGNAL_ENTRY_LEVEL"
        if len(targets) < 2:
            raise ValueError("two_targets_required")
    except (TypeError, ValueError, KeyError):
        return {"recorded": False, "reason": "no_valid_entry_zone_or_trade_levels"}
    risk = entry - stop if side == "LONG" else stop - entry
    if risk <= 0:
        return {"recorded": False, "reason": "invalid_structural_stop"}
    if side == "LONG" and (targets[0] <= entry or entry_low <= stop):
        return {"recorded": False, "reason": "invalid_long_levels"}
    if side == "SHORT" and (targets[0] >= entry or entry_high >= stop):
        return {"recorded": False, "reason": "invalid_short_levels"}
    # Virtual account: 10,000 USDT-equivalent, 0.5% risk, auto leverage <= 5x.
    # This sizing is simulation only; no exchange order is submitted.
    initial_balance = float(os.getenv("PAPER_INITIAL_BALANCE", "10000"))
    risk_cash = initial_balance * 0.005
    risk_per_unit = abs(entry - stop)
    quantity = risk_cash / risk_per_unit
    max_leverage = max(1.0, min(5.0, float(os.getenv("PAPER_MAX_LEVERAGE", "5"))))
    quantity = min(quantity, initial_balance * max_leverage / max(entry, 1e-12))
    notional = quantity * entry
    leverage = min(max_leverage, max(1.0, float(math.ceil(notional / initial_balance))))
    margin_used = notional / leverage
    touched = entry_low <= live_price <= entry_high
    status = "ACTIVE" if touched else "PENDING_ENTRY"
    filled_entry = live_price if touched else entry
    candle_time = result.get("last_candle_time")
    fingerprint = "|".join(str(x) for x in (market, result.get("symbol", ""), result.get("timeframe", ""), candle_time or result.get("updated_at", ""), result["signal"]))
    now = _now()
    note = "ورود آزمایشی؛ قیمت داخل محدوده بود." if touched else "سفارش شرطی آزمایشی؛ با رسیدن قیمت به محدوده ورود فعال می‌شود."
    with _LOCK, _connect() as con:
        cur = con.execute("""
            INSERT OR IGNORE INTO paper_trades
            (fingerprint, created_at, updated_at, market, symbol, timeframe, side, signal_candle_time, entry_price, entry_low, entry_high, signal_entry, last_quote_price, last_quote_at, stop_loss, target_1, target_2, target_3, status, source, note, quantity, notional, margin_used, leverage, initial_balance, risk_cash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (fingerprint, now, now, market, result.get("symbol", ""), result.get("timeframe", ""), side, candle_time, filled_entry, entry_low, entry_high, entry, live_price, now, stop, targets[0], targets[1], targets[2] if len(targets) > 2 else None, status, source, note, quantity, notional, margin_used, leverage, initial_balance, risk_cash))
        if cur.rowcount == 0:
            row = con.execute("SELECT id, status FROM paper_trades WHERE fingerprint=?", (fingerprint,)).fetchone()
            return {"recorded": False, "duplicate": True, "trade_id": row["id"], "status": row["status"]}
        return {"recorded": True, "trade_id": cur.lastrowid, "status": status, "entry_zone": [entry_low, entry_high], "entry_price": filled_entry if touched else None}


def monitor_open_trades(quote_getter):
    """Update open paper trades from fresh quotes; partial close 50% at TP1, rest at TP2."""
    with _LOCK, _connect() as con:
        rows = con.execute("SELECT * FROM paper_trades WHERE status IN ('PENDING_ENTRY','ACTIVE','TP1_HIT') ORDER BY id").fetchall()
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
            if row["status"] != "PENDING_ENTRY":
                with _LOCK, _connect() as con:
                    con.execute("UPDATE paper_trades SET last_quote_price=?, last_quote_at=? WHERE id=?", (price, _now(), row["id"]))
            if row["status"] == "PENDING_ENTRY":
                low = float(row["entry_low"] if row["entry_low"] is not None else row["entry_price"])
                high = float(row["entry_high"] if row["entry_high"] is not None else row["entry_price"])
                stop_pending = float(row["stop_loss"])
                stop_crossed = price <= stop_pending if side == "LONG" else price >= stop_pending
                if stop_crossed:
                    with _LOCK, _connect() as con:
                        con.execute("UPDATE paper_trades SET status='INVALIDATED_BEFORE_ENTRY', updated_at=?, closed_at=?, exit_price=?, exit_reason='STOP_CROSSED_BEFORE_ENTRY' WHERE id=?", (_now(), _now(), price, row["id"]))
                    updated += 1
                    continue
                previous_price = float(row["last_quote_price"]) if row["last_quote_price"] is not None else None
                # Detect a directional crossing even when polling jumps over the zone.
                if side == "LONG":
                    crossed_zone = previous_price is not None and previous_price > high and price <= high and previous_price >= low
                    fill_price = high if crossed_zone and price < low else price
                else:
                    crossed_zone = previous_price is not None and previous_price < low and price >= low and previous_price <= high
                    fill_price = low if crossed_zone and price > high else price
                touched = low <= price <= high or crossed_zone
                with _LOCK, _connect() as con:
                    if touched:
                        fill_risk = abs(fill_price - float(row["stop_loss"]))
                        balance = float(row["initial_balance"] or 10000)
                        risk_cash = float(row["risk_cash"] or balance * 0.005)
                        max_lev = max(1.0, min(5.0, float(os.getenv("PAPER_MAX_LEVERAGE", "5"))))
                        qty = min(risk_cash / max(fill_risk, 1e-12), balance * max_lev / max(fill_price, 1e-12))
                        notional = qty * fill_price
                        leverage = min(max_lev, max(1.0, float(math.ceil(notional / balance))))
                        margin = notional / leverage
                        con.execute("UPDATE paper_trades SET status='ACTIVE', entry_price=?, quantity=?, notional=?, leverage=?, margin_used=?, last_quote_price=?, last_quote_at=?, updated_at=?, note=? WHERE id=?", (fill_price, qty, notional, leverage, margin, price, _now(), _now(), "قیمت محدوده ورود را لمس/قطع کرد؛ معامله آزمایشی فعال شد.", row["id"]))
                        updated += 1
                    else:
                        con.execute("UPDATE paper_trades SET last_quote_price=?, last_quote_at=?, updated_at=? WHERE id=?", (price, _now(), _now(), row["id"]))
                continue
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


def close_paper_trade(trade_id, exit_price=None):
    """Manually close/cancel a paper trade at a supplied fresh quote; never places a real order."""
    with _LOCK, _connect() as con:
        row = con.execute("SELECT * FROM paper_trades WHERE id=?", (int(trade_id),)).fetchone()
        if row is None:
            return {"closed": False, "message": "معامله آزمایشی پیدا نشد"}
        if row["status"] not in ("PENDING_ENTRY", "ACTIVE", "TP1_HIT"):
            return {"closed": False, "message": "این معامله قبلاً بسته شده است"}
        now = _now()
        if row["status"] == "PENDING_ENTRY":
            con.execute("UPDATE paper_trades SET status='MANUAL_CLOSED', updated_at=?, closed_at=?, exit_price=NULL, exit_reason='MANUAL_CANCEL_BEFORE_ENTRY' WHERE id=?", (now, now, int(trade_id)))
            return {"closed": True, "trade_id": int(trade_id), "status": "MANUAL_CLOSED", "exit_price": None}
        if exit_price is None or not math.isfinite(float(exit_price)) or float(exit_price) <= 0:
            return {"closed": False, "message": "قیمت خروج معتبر نیست"}
        entry = float(row["entry_price"])
        stop = float(row["stop_loss"])
        risk = entry - stop if row["side"] == "LONG" else stop - entry
        if risk <= 0:
            return {"closed": False, "message": "حد ضرر معامله معتبر نیست"}
        exit_price = float(exit_price)
        sign = 1.0 if row["side"] == "LONG" else -1.0
        remaining = 0.5 if row["status"] == "TP1_HIT" else 1.0
        realized_r = float(row["realized_r"] or 0.0) + remaining * sign * (exit_price - entry) / risk
        realized_pct = float(row["realized_return_percent"] or 0.0) + remaining * sign * (exit_price - entry) / entry * 100.0
        con.execute("UPDATE paper_trades SET status='MANUAL_CLOSED', updated_at=?, closed_at=?, exit_price=?, exit_reason='MANUAL_CLOSE', realized_r=?, realized_return_percent=?, last_quote_price=?, last_quote_at=? WHERE id=?", (now, now, exit_price, realized_r, realized_pct, exit_price, now, int(trade_id)))
        return {"closed": True, "trade_id": int(trade_id), "status": "MANUAL_CLOSED", "exit_price": exit_price, "realized_r": round(realized_r, 4), "realized_return_percent": round(realized_pct, 4)}


def list_paper_trades(limit=100):
    limit = max(1, min(int(limit), 300))
    with _LOCK, _connect() as con:
        active_rows = [dict(r) for r in con.execute(
            "SELECT * FROM paper_trades WHERE status IN ('PENDING_ENTRY','ACTIVE','TP1_HIT') ORDER BY created_at DESC, id DESC"
        ).fetchall()]
        history_rows = [dict(r) for r in con.execute(
            "SELECT * FROM paper_trades WHERE status NOT IN ('PENDING_ENTRY','ACTIVE','TP1_HIT') ORDER BY COALESCE(closed_at, updated_at, created_at) DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()]
        rows = active_rows + history_rows
        stats = con.execute("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN status IN ('ACTIVE','TP1_HIT') THEN 1 ELSE 0 END) AS active,
                   SUM(CASE WHEN status IN ('TP1_HIT','TP2_HIT','SL_HIT','SL_AFTER_TP1') THEN 1 ELSE 0 END) AS tp1_touched,
                   SUM(CASE WHEN status='TP2_HIT' THEN 1 ELSE 0 END) AS tp2,
                   SUM(CASE WHEN status IN ('SL_HIT','SL_AFTER_TP1') THEN 1 ELSE 0 END) AS stopped,
                   SUM(CASE WHEN status IN ('TP2_HIT','SL_HIT','SL_AFTER_TP1','MANUAL_CLOSED','INVALIDATED_BEFORE_ENTRY') THEN 1 ELSE 0 END) AS closed,
                   SUM(CASE WHEN status='PENDING_ENTRY' THEN 1 ELSE 0 END) AS pending_entry
            FROM paper_trades
        """).fetchone()
        closed_rows = con.execute("""
            SELECT realized_r FROM paper_trades
            WHERE status IN ('TP2_HIT','SL_HIT','SL_AFTER_TP1','MANUAL_CLOSED')
              AND COALESCE(exit_reason, '') != 'MANUAL_CANCEL_BEFORE_ENTRY'
        """).fetchall()
    for row in rows:
        entry = float(row.get("entry_price") or 0)
        quantity = float(row.get("quantity") or 0)
        margin = float(row.get("margin_used") or 0)
        price = float(row.get("last_quote_price") or entry)
        side_sign = 1 if row.get("side") == "LONG" else -1
        remaining_fraction = 0.5 if row.get("status") == "TP1_HIT" else (1.0 if row.get("status") == "ACTIVE" else 0.0)
        unrealized = side_sign * (price - entry) * quantity * remaining_fraction
        realized_cash = float(row.get("realized_return_percent") or 0) / 100.0 * float(row.get("notional") or 0)
        total_pnl = realized_cash + unrealized
        row["unrealized_pnl"] = round(unrealized, 4)
        row["realized_pnl"] = round(realized_cash, 4)
        row["total_pnl"] = round(total_pnl, 4)
        row["return_on_margin_percent"] = round(total_pnl / margin * 100.0, 3) if margin > 0 else 0.0
        row["remaining_fraction"] = remaining_fraction
    closed_r = [float(r["realized_r"] or 0) for r in closed_rows]
    wins = sum(1 for x in closed_r if x > 0)
    open_pnl = round(sum(float(row.get("total_pnl") or 0) for row in rows
                         if row.get("status") in ("ACTIVE", "TP1_HIT")), 4)
    active_statuses = {"PENDING_ENTRY", "ACTIVE", "TP1_HIT"}
    active_trades = [row for row in rows if row.get("status") in active_statuses]
    history_trades = [row for row in rows if row.get("status") not in active_statuses]
    history_trades.sort(key=lambda row: (row.get("closed_at") or row.get("updated_at") or row.get("created_at") or ""), reverse=True)
    return {
        "available": True, "mode": "PAPER_ONLY",
        "active_trades": active_trades,
        "history_trades": history_trades,
        "stats": {
            "total": int(stats["total"] or 0), "active": int(stats["active"] or 0),
            "pending_entry": int(stats["pending_entry"] or 0),
            "tp1_touched": int(stats["tp1_touched"] or 0), "tp2": int(stats["tp2"] or 0),
            "stopped": int(stats["stopped"] or 0), "closed": int(stats["closed"] or 0),
            "win_rate_percent": round(wins / len(closed_r) * 100, 2) if closed_r else None,
            "net_r": round(sum(closed_r), 3),
            "open_pnl": open_pnl,
            "virtual_balance_per_trade": float(os.getenv("PAPER_INITIAL_BALANCE", "10000")),
            "risk_per_trade_percent": 0.5,
            "max_leverage": max(1.0, min(5.0, float(os.getenv("PAPER_MAX_LEVERAGE", "5")))),
        },
        "trades": rows
    }
