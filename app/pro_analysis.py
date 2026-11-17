"""Advanced, dependency-free market analytics for Smart Money AI.

All pattern and profile outputs are heuristic approximations based on OHLCV candles.
They are descriptive features, not institutional order-flow data or probabilities.
"""
from datetime import datetime, timezone
import math


def _ema_series(values, period):
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(value * alpha + out[-1] * (1.0 - alpha))
    return out


def _sma(values, period):
    return sum(values[-period:]) / min(period, len(values)) if values else 0.0


def _rsi_wilder(closes, period=14):
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    if len(changes) < period:
        return 50.0
    gains = [max(x, 0.0) for x in changes]
    losses = [max(-x, 0.0) for x in changes]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss <= 1e-15:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _true_ranges(candles):
    out = []
    for i, candle in enumerate(candles):
        if i == 0:
            out.append(candle["high"] - candle["low"])
        else:
            prior = candles[i - 1]["close"]
            out.append(max(
                candle["high"] - candle["low"],
                abs(candle["high"] - prior),
                abs(candle["low"] - prior)
            ))
    return out


def _adx(candles, period=14):
    trs = _true_ranges(candles)
    plus_dm, minus_dm = [0.0], [0.0]
    for i in range(1, len(candles)):
        up = candles[i]["high"] - candles[i - 1]["high"]
        down = candles[i - 1]["low"] - candles[i]["low"]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    dx_values = []
    latest_plus = latest_minus = 0.0
    for end in range(period, len(candles) + 1):
        tr_sum = sum(trs[end - period:end])
        if tr_sum <= 1e-15:
            continue
        pdi = 100.0 * sum(plus_dm[end - period:end]) / tr_sum
        mdi = 100.0 * sum(minus_dm[end - period:end]) / tr_sum
        denom = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / denom if denom > 1e-15 else 0.0
        dx_values.append(dx)
        latest_plus, latest_minus = pdi, mdi
    return (_sma(dx_values, period), latest_plus, latest_minus)


def _pivot_points(candles, lookback=80):
    highs, lows = [], []
    start = max(2, len(candles) - lookback)
    for i in range(start, len(candles) - 2):
        h, lo = candles[i]["high"], candles[i]["low"]
        if h > max(c["high"] for c in candles[i - 2:i]) and h >= max(c["high"] for c in candles[i + 1:i + 3]):
            highs.append({"index": i, "price": h})
        if lo < min(c["low"] for c in candles[i - 2:i]) and lo <= min(c["low"] for c in candles[i + 1:i + 3]):
            lows.append({"index": i, "price": lo})
    return highs, lows


def _volume_profile(candles, bins=24, lookback=100):
    sample = candles[-lookback:]
    usable = [c for c in sample if c.get("volume", 0) > 0]
    if len(usable) < 10:
        return {"available": False, "note": "حجم معتبر برای ساخت پروفایل در منبع داده موجود نیست."}
    low = min(c["low"] for c in usable)
    high = max(c["high"] for c in usable)
    if high <= low:
        return {"available": False, "note": "دامنه قیمت برای پروفایل کافی نیست."}
    step = (high - low) / bins
    volumes = [0.0] * bins
    for candle in usable:
        typical = (candle["high"] + candle["low"] + candle["close"]) / 3.0
        idx = min(bins - 1, max(0, int((typical - low) / step)))
        volumes[idx] += candle.get("volume", 0.0)
    poc = max(range(bins), key=lambda i: volumes[i])
    total = sum(volumes)
    target = total * 0.70
    left = right = poc
    accumulated = volumes[poc]
    while accumulated < target and (left > 0 or right < bins - 1):
        next_left = volumes[left - 1] if left > 0 else -1
        next_right = volumes[right + 1] if right < bins - 1 else -1
        if next_left >= next_right:
            left -= 1
            accumulated += volumes[left]
        else:
            right += 1
            accumulated += volumes[right]
    return {
        "available": True,
        "poc": low + (poc + 0.5) * step,
        "value_area_low": low + left * step,
        "value_area_high": low + (right + 1) * step,
        "method": "OHLCV typical-price binning; approximate, not exchange tick volume profile",
        "lookback_candles": len(sample)
    }


def _session_levels(candles):
    days = {}
    for candle in candles:
        day = datetime.fromtimestamp(candle["time"] / 1000, tz=timezone.utc).date().isoformat()
        item = days.setdefault(day, {"high": candle["high"], "low": candle["low"], "close": candle["close"]})
        item["high"] = max(item["high"], candle["high"])
        item["low"] = min(item["low"], candle["low"])
        item["close"] = candle["close"]
    keys = sorted(days)
    if not keys:
        return {"available": False}
    current = days[keys[-1]]
    previous = days[keys[-2]] if len(keys) > 1 else None
    result = {"available": True, "session": "UTC", "current_high": current["high"], "current_low": current["low"]}
    if previous:
        pivot = (previous["high"] + previous["low"] + previous["close"]) / 3.0
        result.update({
            "previous_high": previous["high"], "previous_low": previous["low"],
            "previous_close": previous["close"], "classic_pivot": pivot,
            "resistance_1": 2 * pivot - previous["low"],
            "support_1": 2 * pivot - previous["high"]
        })
    return result


def advanced_analysis(candles, timeframe):
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c.get("volume", 0.0) or 0.0 for c in candles]
    last = candles[-1]
    price = closes[-1]
    tr = _true_ranges(candles)
    atr = _sma(tr, 14)
    atr_pct = (atr / price * 100.0) if price else 0.0
    rsi = _rsi_wilder(closes, 14)

    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd_line = ema12[-1] - ema26[-1]
    macd_series = [a - b for a, b in zip(ema12, ema26)]
    macd_signal = _ema_series(macd_series, 9)[-1]
    macd_hist = macd_line - macd_signal

    adx, plus_di, minus_di = _adx(candles, 14)
    if adx >= 25:
        regime = "trending"
    elif adx <= 18:
        regime = "ranging"
    else:
        regime = "transition"

    bb_period = 20
    bb_mid = _sma(closes, bb_period)
    bb_std = math.sqrt(sum((x - bb_mid) ** 2 for x in closes[-bb_period:]) / bb_period)
    bb_upper, bb_lower = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
    bb_width = ((bb_upper - bb_lower) / bb_mid * 100.0) if bb_mid else 0.0
    widths = []
    for end in range(max(20, len(closes) - 80), len(closes) + 1):
        window = closes[end - 20:end]
        mid = sum(window) / 20
        sd = math.sqrt(sum((x - mid) ** 2 for x in window) / 20)
        widths.append((4 * sd / mid * 100.0) if mid else 0.0)
    width_rank = sum(1 for x in widths if x <= bb_width) / max(1, len(widths))
    squeeze = width_rank <= 0.20

    stoch_window_high = max(highs[-14:])
    stoch_window_low = min(lows[-14:])
    stoch_k = ((price - stoch_window_low) / (stoch_window_high - stoch_window_low) * 100.0) if stoch_window_high > stoch_window_low else 50.0
    stoch_values = []
    for end in range(max(14, len(candles) - 2), len(candles) + 1):
        hi = max(highs[end - 14:end])
        lo = min(lows[end - 14:end])
        stoch_values.append(((closes[end - 1] - lo) / (hi - lo) * 100.0) if hi > lo else 50.0)
    stoch_d = _sma(stoch_values, 3)

    average_volume = _sma(volumes[:-1], 20) if len(volumes) > 1 else 0.0
    relative_volume = (volumes[-1] / average_volume) if average_volume > 0 else None

    # UTC-day anchored VWAP. It is omitted when the provider has no usable volume.
    last_day = datetime.fromtimestamp(last["time"] / 1000, tz=timezone.utc).date()
    day_candles = [c for c in candles if datetime.fromtimestamp(c["time"] / 1000, tz=timezone.utc).date() == last_day]
    day_volume = sum(c.get("volume", 0.0) or 0.0 for c in day_candles)
    vwap = None
    if day_volume > 0:
        vwap = sum(((c["high"] + c["low"] + c["close"]) / 3.0) * (c.get("volume", 0.0) or 0.0) for c in day_candles) / day_volume

    # Candle patterns and displacement are rule-based descriptions, not forecasts.
    prev = candles[-2]
    body = abs(last["close"] - last["open"])
    candle_range = max(1e-15, last["high"] - last["low"])
    upper_wick = last["high"] - max(last["open"], last["close"])
    lower_wick = min(last["open"], last["close"]) - last["low"]
    bullish_engulf = last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] <= prev["close"] and last["close"] >= prev["open"]
    bearish_engulf = last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] >= prev["close"] and last["close"] <= prev["open"]
    bullish_pin = lower_wick >= body * 2 and upper_wick <= max(body, candle_range * 0.15) and last["close"] > last["low"] + candle_range * 0.55
    bearish_pin = upper_wick >= body * 2 and lower_wick <= max(body, candle_range * 0.15) and last["close"] < last["high"] - candle_range * 0.55
    inside_bar = last["high"] < prev["high"] and last["low"] > prev["low"]
    displacement = body >= 1.2 * atr if atr > 0 else False
    candle_patterns = []
    if bullish_engulf: candle_patterns.append("bullish_engulfing")
    if bearish_engulf: candle_patterns.append("bearish_engulfing")
    if bullish_pin: candle_patterns.append("bullish_pin_bar")
    if bearish_pin: candle_patterns.append("bearish_pin_bar")
    if inside_bar: candle_patterns.append("inside_bar")
    if displacement: candle_patterns.append("displacement_candle")

    pivot_highs, pivot_lows = _pivot_points(candles)
    tolerance = max(atr * 0.15, abs(price) * 0.0003)
    equal_highs = None
    equal_lows = None
    if len(pivot_highs) >= 2 and abs(pivot_highs[-1]["price"] - pivot_highs[-2]["price"]) <= tolerance:
        equal_highs = (pivot_highs[-1]["price"] + pivot_highs[-2]["price"]) / 2.0
    if len(pivot_lows) >= 2 and abs(pivot_lows[-1]["price"] - pivot_lows[-2]["price"]) <= tolerance:
        equal_lows = (pivot_lows[-1]["price"] + pivot_lows[-2]["price"]) / 2.0

    range_high = max(highs[-50:])
    range_low = min(lows[-50:])
    midpoint = (range_high + range_low) / 2.0
    premium_discount = "premium" if price > midpoint else "discount" if price < midpoint else "equilibrium"

    fib = {"available": False}
    if pivot_highs and pivot_lows:
        ph, pl = pivot_highs[-1], pivot_lows[-1]
        if ph["index"] > pl["index"]:
            lo, hi, leg = pl["price"], ph["price"], "upswing"
            fib = {"available": True, "leg": leg, "0.382": hi - (hi - lo) * 0.382, "0.5": hi - (hi - lo) * 0.5, "0.618": hi - (hi - lo) * 0.618, "0.786": hi - (hi - lo) * 0.786}
        elif pl["index"] > ph["index"]:
            hi, lo, leg = ph["price"], pl["price"], "downswing"
            fib = {"available": True, "leg": leg, "0.382": lo + (hi - lo) * 0.382, "0.5": lo + (hi - lo) * 0.5, "0.618": lo + (hi - lo) * 0.618, "0.786": lo + (hi - lo) * 0.786}

    volume_profile = _volume_profile(candles)
    session = _session_levels(candles)
    candle_age = max(0, int(datetime.now(timezone.utc).timestamp() * 1000 - last["time"]))
    volume_available = any(v > 0 for v in volumes[-30:])

    return {
        "rsi_wilder_14": round(rsi, 2),
        "macd": {"line": macd_line, "signal": macd_signal, "histogram": macd_hist, "bias": "bullish" if macd_hist > 0 else "bearish" if macd_hist < 0 else "neutral"},
        "adx": {"value": round(adx, 2), "plus_di": round(plus_di, 2), "minus_di": round(minus_di, 2), "regime": regime},
        "volatility": {"atr": atr, "atr_percent": round(atr_pct, 4), "bollinger_upper": bb_upper, "bollinger_mid": bb_mid, "bollinger_lower": bb_lower, "bollinger_width_percent": round(bb_width, 4), "bollinger_width_percentile": round(width_rank * 100, 1), "squeeze": squeeze},
        "momentum": {"stochastic_k": round(stoch_k, 2), "stochastic_d": round(stoch_d, 2), "relative_volume": round(relative_volume, 3) if relative_volume is not None else None, "vwap_utc_day": vwap, "vwap_available": vwap is not None},
        "candle_patterns": candle_patterns,
        "liquidity_map": {"equal_highs_pool": equal_highs, "equal_lows_pool": equal_lows, "tolerance": tolerance},
        "premium_discount": {"zone": premium_discount, "range_high_50": range_high, "range_low_50": range_low, "equilibrium": midpoint},
        "fibonacci": fib,
        "volume_profile": volume_profile,
        "session_levels": session,
        "data_quality": {"closed_candle_analysis": True, "candles_used": len(candles), "last_candle_time_ms": last["time"], "last_candle_age_seconds": candle_age // 1000, "volume_available": volume_available, "timeframe": timeframe},
        "risk_template": {"default_risk_percent": 0.5, "risk_budget_per_10000_account_value": 50.0, "position_size_formula": "risk_budget / abs(entry - stop_loss)", "note": "فرمول آموزشی است؛ ضریب قرارداد، اهرم، کارمزد، اسلیپیج و موجودی واقعی باید جداگانه لحاظ شوند."},
        "methodology_note": "ADX/RSI و VWAP/پروفایل بر اساس OHLCV عمومی محاسبه می‌شوند؛ حجم پروفایل تقریبی است و داده تیک، دفتر سفارش تاریخی یا جریان سفارش نهادی نیست."
    }
