"""High-selectivity opportunity scanner; never creates paper/live trades."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter, HTTPException

router = APIRouter()
_CACHE = {"at": 0.0, "key": None, "data": None}


def _pivots(candles):
    highs, lows = [], []
    for i in range(2, len(candles) - 2):
        c = candles[i]
        if c["high"] > max(x["high"] for x in candles[i - 2:i]) and c["high"] >= max(x["high"] for x in candles[i + 1:i + 3]):
            highs.append(float(c["high"]))
        if c["low"] < min(x["low"] for x in candles[i - 2:i]) and c["low"] <= min(x["low"] for x in candles[i + 1:i + 3]):
            lows.append(float(c["low"]))
    return highs, lows


def _levels(candles, side, price):
    highs, lows = _pivots(candles)
    raw = [x for x in (highs if side == "long" else lows) if (x > price if side == "long" else x < price)]
    raw = sorted(set(raw), reverse=(side == "short"))
    # Keep distinct structural levels; do not manufacture four targets.
    out = []
    tolerance = max(abs(price) * 0.001, 1e-12)
    for value in raw:
        if not out or all(abs(value - old) > tolerance for old in out):
            out.append(value)
        if len(out) == 4:
            break
    return out


def _serialize_candidate(market, symbol, daily_candles, daily, four_hour, quote, event_risk):
    side = "long" if daily.get("signal") == "BUY" else "short" if daily.get("signal") == "SELL" else None
    four_side = "long" if four_hour.get("signal") == "BUY" else "short" if four_hour.get("signal") == "SELL" else None
    plan = ((daily.get("smart_entry") or {}).get(side) if side else None) or {}
    quote_age = quote.get("age_seconds")
    fresh = bool(quote.get("available") and not quote.get("is_proxy") and quote_age is not None
                 and quote_age <= (60 if market == "crypto" else 900))
    strategy_ok = bool((daily.get("strategy_plan") or {}).get("passed"))
    risk_ok = bool((daily.get("risk_management") or {}).get("risk_gate_passed"))
    execution_ok = fresh and not bool(event_risk.get("entry_blocked_by_calendar"))
    aligned = bool(side and four_side == side)
    plan_zone = plan.get("zone") if isinstance(plan, dict) else None
    live_price = quote.get("price") if fresh else None
    inside_zone = bool(plan_zone and live_price is not None
                       and float(plan_zone.get("low", 0)) <= float(live_price) <= float(plan_zone.get("high", 0)))
    invalidation = plan.get("invalidation_level") or plan.get("stop_loss")
    not_invalidated = bool(
        live_price is not None and invalidation is not None and
        (float(live_price) > float(invalidation) if side == "long" else float(live_price) < float(invalidation))
    ) if side else False

    # A directional call is "confirmed" only when daily + 4h agree, the
    # strategy/risk gates pass, the quote is fresh, and the planned zone is live.
    confirmed = bool(side and aligned and strategy_ok and risk_ok and execution_ok
                     and plan.get("available") and inside_zone and not_invalidated
                     and plan.get("reaction_confirmed") and plan.get("risk_gate_passed"))
    watch = bool(side and strategy_ok and risk_ok and aligned and not_invalidated)
    if confirmed:
        state = "CONFIRMED_SETUP"
        label = "ستاپ تأییدشده؛ بررسی ورود"
    elif watch:
        state = "WAITING_FOR_TRIGGER"
        label = "در انتظار تریگر / رسیدن قیمت به ناحیه"
    else:
        state = "NO_TRADE"
        label = "فعلاً بدون ورود معتبر"

    structural_targets = _levels(daily_candles, side, float(daily.get("price") or 0)) if side else []
    checks = (daily.get("strategy_plan") or {}).get("checks") or []
    quality = 0
    quality += 25 if strategy_ok else 0
    quality += 20 if aligned else 0
    quality += 15 if risk_ok else 0
    quality += 15 if fresh else 0
    quality += 15 if plan.get("available") else 0
    quality += 10 if inside_zone else 0
    quality = min(100, quality)
    candles = daily_candles[-500:]
    chart = [{
        "time": int(c["time"] // 1000), "open": c["open"], "high": c["high"],
        "low": c["low"], "close": c["close"]
    } for c in candles]
    levels = {
        "entry_zone": plan_zone if plan_zone else None,
        "entry": plan.get("entry") if plan else None,
        "stop_loss": plan.get("stop_loss") if plan else None,
        "invalidation": invalidation,
        "targets": structural_targets,
        "support": daily.get("chart_levels", {}).get("support"),
        "resistance": daily.get("chart_levels", {}).get("resistance"),
        "breakout_trigger": ((daily.get("conditional_scenarios") or {}).get("bullish") or {}).get("trigger")
            if side == "long" else ((daily.get("conditional_scenarios") or {}).get("bearish") or {}).get("trigger")
    }
    return {
        "market": market, "symbol": symbol, "side": side, "state": state, "state_label": label,
        "quality_score": quality, "score_is_probability": False,
        "daily_signal": daily.get("signal"), "four_hour_signal": four_hour.get("signal"),
        "daily_bias": daily.get("bias"), "four_hour_bias": four_hour.get("bias"),
        "price": live_price if live_price is not None else daily.get("price"),
        "analysis_price": daily.get("price"), "quote": quote,
        "strategy": daily.get("strategy_plan"), "strategy_ok": strategy_ok,
        "risk_ok": risk_ok, "execution_ok": execution_ok, "multi_timeframe_aligned": aligned,
        "entry_plan": plan, "levels": levels, "chart_candles": chart,
        "rsi": daily.get("rsi"), "atr": daily.get("atr"),
        "market_structure": daily.get("market_structure"), "bos": daily.get("bos"),
        "choch": daily.get("choch"), "liquidity_sweep": daily.get("liquidity_sweep"),
        "fvg_zone": daily.get("fvg_zone"), "order_block": daily.get("order_block_approx"),
        "liquidity_sweep_level": daily.get("liquidity_sweep_level"),
        "liquidity_sweep_time": daily.get("liquidity_sweep_time"),
        "chart_markers": ([{"time": int(daily_candles[-1]["time"] // 1000),
            "position": "belowBar" if side == "long" else "aboveBar",
            "color": "#28c997" if side == "long" else "#ff7182",
            "shape": "arrowUp" if side == "long" else "arrowDown",
            "text": "BUY" if side == "long" else "SELL"}] if side else []),
        "event_risk": event_risk, "candle_source": daily_candles[-1].get("source"),
        "last_candle_time": daily.get("last_candle_time"),
        "reasons": daily.get("analysis") or [], "wait_reason": daily.get("wait_reason") or [],
        "targets_note": "هدف‌ها فقط از سوئینگ‌های تأییدشده مخالف گرفته می‌شوند؛ اگر سطح کافی نباشد، هدف ساختگی نمایش داده نمی‌شود.",
        "updated_at": daily.get("updated_at")
    }


@router.get("/api/opportunities")
def opportunities(market: str = "all", limit: int = 12, refresh: bool = False):
    # Imports are intentionally lazy to avoid circular imports during app startup.
    from . import main as core

    if market not in ("all", "crypto", "gold", "silver"):
        raise HTTPException(400, "بازار نامعتبر است")
    limit = max(1, min(int(limit), 20))
    cache_key = market + ":" + str(limit)
    now = time.time()
    if not refresh and _CACHE["data"] is not None and _CACHE["key"] == cache_key and now - _CACHE["at"] < 240:
        result = dict(_CACHE["data"])
        result["cache_hit"] = True
        return result

    tasks = []
    universe = {}
    if market in ("all", "crypto"):
        selected, total, eligible, method = core.get_scanner_universe(limit)
        tasks.extend([("crypto", s) for s in selected])
        universe = {"crypto_pairs_available": total, "liquid_pairs_selected": eligible,
                    "ranking_method": method}
    if market in ("all", "gold"):
        tasks.append(("gold", core.METALS["gold"]))
    if market in ("all", "silver"):
        tasks.append(("silver", core.METALS["silver"]))

    results, errors = [], []
    def analyze_one(m, s):
        daily_raw = core.get_candles(m, s, "1d")
        daily = core.completed_candles(daily_raw, "1d")
        if len(daily) < 100:
            raise RuntimeError("برای تحلیل روزانه حداقل ۱۰۰ کندل بسته‌شده لازم است")
        four_raw = core.get_candles(m, s, "4h")
        four = core.completed_candles(four_raw, "4h")
        if len(four) < 100:
            raise RuntimeError("برای تأیید ۴ ساعته حداقل ۱۰۰ کندل بسته‌شده لازم است")
        daily_result = core.analyze(daily, m, s, "1d")
        four_result = core.analyze(four, m, s, "4h")
        quote = core.get_live_quote(m, s)
        event_risk = core.economic_event_risk(m)
        return _serialize_candidate(m, s, daily, daily_result, four_result, quote, event_risk)

    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(analyze_one, m, s): (m, s) for m, s in tasks}
        for future in as_completed(future_map):
            m, s = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"market": m, "symbol": s, "error": str(exc)[:180]})

    # Confirmed first, then watch candidates. NO_TRADE rows remain visible but
    # are ranked below actual setups so the scanner never hides a failed gate.
    order = {"CONFIRMED_SETUP": 0, "WAITING_FOR_TRIGGER": 1, "NO_TRADE": 2}
    results.sort(key=lambda x: (order.get(x["state"], 3), -x["quality_score"]))
    payload = {
        "market": market, "results": results, "errors": errors, "universe": universe,
        "updated_at": core.utc_now(), "cache_hit": False,
        "method": "Binance Spot 24h quote-volume universe + closed 1D/4H candles + existing SMC strategy/risk gates + fresh spot quote",
        "disclaimer": "امتیاز کیفیت، احتمال برد نیست. ستاپ تأییدشده هم تضمین سود نیست. اسکنر فقط تحلیل می‌کند و هیچ معامله آزمایشی یا واقعی ایجاد نمی‌کند."
    }
    _CACHE.update({"at": now, "key": cache_key, "data": payload})
    return payload
