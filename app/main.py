import os
import json
import sqlite3
import urllib.parse
import urllib.request
import urllib.error
import hmac
import time
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .pro_analysis import advanced_analysis
from .backtesting import run_research_backtest

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
DB_PATH = os.getenv("DATABASE_PATH", os.path.join(ROOT_DIR, "alerts.db"))

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gpt-6-luna")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")

app = FastAPI(title="Smart Money AI | Crypto, Gold & Silver")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "static")), name="static")

CRYPTO = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
    "TRXUSDT", "DOTUSDT", "BCHUSDT", "NEARUSDT", "UNIUSDT",
    "SUIUSDT", "TONUSDT", "ETCUSDT", "APTUSDT", "ARBUSDT"
]
METALS = {"gold": "XAU/USD", "silver": "XAG/USD"}
TIMEFRAMES = {
    "5m": {"binance": "5m", "twelve": "5min"},
    "15m": {"binance": "15m", "twelve": "15min"},
    "1h": {"binance": "1h", "twelve": "1h"},
    "4h": {"binance": "4h", "twelve": "4h"},
    "1d": {"binance": "1d", "twelve": "1day"},
    "1w": {"binance": "1w", "twelve": "1week"},
}
NEWS_CACHE = {}
GDELT_LOCK = threading.Lock()
GDELT_LAST_REQUEST_AT = 0.0
MACRO_CACHE = {"at": 0, "data": None}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            symbol TEXT,
            timeframe TEXT,
            signal TEXT,
            price REAL,
            message TEXT
        )
        """)


init_db()


def fetch_json(url, headers=None, timeout=15):
    request_headers = {"User-Agent": "SmartMoneyAI/2.0"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        raise RuntimeError("Provider HTTP " + str(exc.code) + ": " + body[:180])
    except Exception as exc:
        raise RuntimeError("Provider connection error: " + str(exc)[:160])



def aggregate_candles(candles, timeframe):
    """Aggregate lower-timeframe Yahoo candles into 4h or weekly bars."""
    buckets = {}
    for candle in candles:
        stamp = datetime.fromtimestamp(candle["time"] / 1000, tz=timezone.utc)
        if timeframe == "4h":
            bucket_time = int(stamp.timestamp() // 14400 * 14400)
        elif timeframe == "1w":
            monday = (stamp - __import__("datetime").timedelta(days=stamp.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            bucket_time = int(monday.timestamp())
        else:
            bucket_time = int(stamp.timestamp())
        if bucket_time not in buckets:
            buckets[bucket_time] = {
                "time": bucket_time * 1000, "open": candle["open"],
                "high": candle["high"], "low": candle["low"],
                "close": candle["close"], "volume": candle.get("volume", 0) or 0
            }
        else:
            bar = buckets[bucket_time]
            bar["high"] = max(bar["high"], candle["high"])
            bar["low"] = min(bar["low"], candle["low"])
            bar["close"] = candle["close"]
            bar["volume"] += candle.get("volume", 0) or 0
    return [buckets[key] for key in sorted(buckets)]


def get_yahoo_metals_candles(market, timeframe):
    """No-key fallback: try spot symbols first, then exchange-traded futures."""
    tickers = {
        "gold": ["XAUUSD=X", "GC=F"],
        "silver": ["XAGUSD=X", "SI=F"],
    }.get(market)
    if not tickers:
        raise RuntimeError("Unsupported metal market")

    settings = {
        "5m": ("5m", "5d", False),
        "15m": ("15m", "1mo", False),
        "1h": ("60m", "3mo", False),
        "4h": ("60m", "6mo", True),
        "1d": ("1d", "2y", False),
        "1w": ("1d", "5y", True),
    }
    interval, period, aggregate = settings[timeframe]
    ticker_errors = []
    for ticker in tickers:
        try:
            url = (
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                + urllib.parse.quote(ticker, safe="=")
                + "?" + urllib.parse.urlencode({
                    "range": period, "interval": interval, "events": "history"
                })
            )
            data = fetch_json(url, headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; SmartMoneyAI/2.2)"
            }, timeout=20)
            chart = data.get("chart", {})
            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))
            result = (chart.get("result") or [None])[0]
            if not result:
                raise RuntimeError("Yahoo Finance returned no chart data")
            timestamps = result.get("timestamp") or []
            quote = (result.get("indicators", {}).get("quote") or [{}])[0]
            candles = []
            for i, stamp in enumerate(timestamps):
                try:
                    op, hi, lo, cl = (
                        quote.get("open", [])[i], quote.get("high", [])[i],
                        quote.get("low", [])[i], quote.get("close", [])[i]
                    )
                    if None in (op, hi, lo, cl):
                        continue
                    vol = (quote.get("volume") or [None] * len(timestamps))[i]
                    candles.append({
                        "time": int(stamp) * 1000,
                        "open": float(op), "high": float(hi), "low": float(lo),
                        "close": float(cl), "volume": float(vol or 0)
                    })
                except (IndexError, TypeError, ValueError):
                    continue
            if aggregate:
                candles = aggregate_candles(candles, timeframe)
            if len(candles) < 60:
                raise RuntimeError(
                    "Yahoo Finance supplied only " + str(len(candles)) +
                    " usable candles for " + timeframe
                )
            if ticker in ("GC=F", "SI=F"):
                # Futures are a fallback proxy; their price can differ from spot.
                for candle in candles:
                    candle["source_symbol"] = ticker
            return candles
        except Exception as exc:
            ticker_errors.append(ticker + ": " + str(exc)[:150])

    raise RuntimeError("Yahoo Finance symbols failed: " + " | ".join(ticker_errors))


def get_twelve_data_metals_candles(symbol, timeframe):
    tf = TIMEFRAMES[timeframe]
    params = urllib.parse.urlencode({
        "symbol": symbol, "interval": tf["twelve"], "outputsize": 250,
        "apikey": TWELVE_DATA_API_KEY, "format": "JSON"
    })
    data = fetch_json("https://api.twelvedata.com/time_series?" + params)
    if "values" not in data:
        message = data.get("message") or data.get("status") or "No time-series values"
        raise RuntimeError("Twelve Data: " + str(message)[:180])
    candles = []
    for row in reversed(data["values"]):
        stamp = row.get("datetime", "")
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                pass
        if parsed is None:
            continue
        candles.append({
            "time": int(parsed.timestamp() * 1000),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row.get("volume") or 0)
        })
    if len(candles) < 50:
        raise RuntimeError("Twelve Data returned insufficient candles")
    return candles


def get_candles(market, symbol, timeframe):
    if timeframe not in TIMEFRAMES:
        raise HTTPException(400, "تایم‌فریم نامعتبر است")
    tf = TIMEFRAMES[timeframe]

    if market == "crypto":
        if symbol not in CRYPTO:
            raise HTTPException(400, "نماد کریپتو در فهرست مجاز نیست")
        params = urllib.parse.urlencode({
            "symbol": symbol, "interval": tf["binance"], "limit": 250
        })
        try:
            raw = fetch_json("https://data-api.binance.vision/api/v3/klines?" + params)
        except Exception as exc:
            raise HTTPException(502, "خطا از منبع داده کریپتو: " + str(exc))
        return [{
            "time": int(row[0]), "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]), "volume": float(row[5])
        } for row in raw]

    if market in METALS:
        if symbol != METALS[market]:
            raise HTTPException(400, "نماد فلز نامعتبر است")

        provider_errors = []
        if TWELVE_DATA_API_KEY:
            try:
                return get_twelve_data_metals_candles(symbol, timeframe)
            except Exception as exc:
                provider_errors.append("Twelve Data: " + str(exc)[:130])

        try:
            return get_yahoo_metals_candles(market, timeframe)
        except Exception as exc:
            provider_errors.append("Yahoo Finance: " + str(exc)[:160])
            raise HTTPException(
                502,
                "دریافت قیمت " + ("طلا" if market == "gold" else "نقره") +
                " از منابع عمومی ناموفق بود. " + " | ".join(provider_errors) +
                ". در صورت تداوم خطا می‌توانی TWELVE_DATA_API_KEY را در Render اضافه کنی."
            )

    raise HTTPException(400, "فقط کریپتو، طلا و نقره پشتیبانی می‌شوند")


def ema(values, period):
    if len(values) < period:
        return values[-1]
    k = 2 / (period + 1)
    value = sum(values[:period]) / period
    for item in values[period:]:
        value = item * k + value * (1 - k)
    return value


TIMEFRAME_MILLISECONDS = {
    "5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000, "1w": 7 * 24 * 60 * 60 * 1000
}


def completed_candles(candles, timeframe):
    """Exclude the currently forming bar to reduce repainting and look-ahead errors."""
    interval = TIMEFRAME_MILLISECONDS.get(timeframe)
    if interval and len(candles) > 1:
        last_open_ms = int(candles[-1]["time"])
        if int(time.time() * 1000) < last_open_ms + interval:
            return candles[:-1]
    return candles


def analyze(candles, market, symbol, timeframe):
    candles = completed_candles(candles, timeframe)
    if len(candles) < 60:
        raise HTTPException(502, "برای تحلیل معتبرتر حداقل ۶۰ کندل بسته‌شده لازم است")

    advanced = advanced_analysis(candles, timeframe)
    closes = [c["close"] for c in candles]
    last = candles[-1]
    previous = candles[-2]
    e9, e21, e50 = ema(closes, 9), ema(closes, 21), ema(closes, 50)

    # Wilder-smoothed RSI and MACD line/signal histogram from the advanced engine.
    rsi_value = advanced["rsi_wilder_14"]
    macd_data = advanced["macd"]
    macd_bull = macd_data["line"] > 0 and macd_data["histogram"] > 0
    macd_bear = macd_data["line"] < 0 and macd_data["histogram"] < 0
    momentum_up = last["close"] > candles[-4]["close"]
    momentum_down = last["close"] < candles[-4]["close"]

    # ATR is used to filter tiny gaps and size the risk levels.
    trs = []
    for index in range(1, len(candles)):
        current, prior = candles[index], candles[index - 1]
        trs.append(max(
            current["high"] - current["low"],
            abs(current["high"] - prior["close"]),
            abs(current["low"] - prior["close"])
        ))
    atr = advanced["volatility"]["atr"]

    # Confirmed swing points: two candles on each side.
    pivot_highs, pivot_lows = [], []
    for i in range(2, len(candles) - 2):
        high = candles[i]["high"]
        low = candles[i]["low"]
        left_highs = [c["high"] for c in candles[i - 2:i]]
        right_highs = [c["high"] for c in candles[i + 1:i + 3]]
        left_lows = [c["low"] for c in candles[i - 2:i]]
        right_lows = [c["low"] for c in candles[i + 1:i + 3]]
        if high > max(left_highs) and high >= max(right_highs):
            pivot_highs.append({"index": i, "price": high})
        if low < min(left_lows) and low <= min(right_lows):
            pivot_lows.append({"index": i, "price": low})

    recent_window = candles[-30:-1]
    swing_high = pivot_highs[-1]["price"] if pivot_highs else max(c["high"] for c in recent_window)
    swing_low = pivot_lows[-1]["price"] if pivot_lows else min(c["low"] for c in recent_window)

    structure_trend = "unknown"
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        higher_highs = pivot_highs[-1]["price"] > pivot_highs[-2]["price"]
        higher_lows = pivot_lows[-1]["price"] > pivot_lows[-2]["price"]
        lower_highs = pivot_highs[-1]["price"] < pivot_highs[-2]["price"]
        lower_lows = pivot_lows[-1]["price"] < pivot_lows[-2]["price"]
        if higher_highs and higher_lows:
            structure_trend = "bullish"
        elif lower_highs and lower_lows:
            structure_trend = "bearish"
        elif higher_highs and lower_lows:
            structure_trend = "expanding_range"

    # A close beyond a confirmed swing level is a break; a wick alone is not.
    broke_up = any(c["close"] > swing_high for c in candles[-3:])
    broke_down = any(c["close"] < swing_low for c in candles[-3:])
    choch_up = broke_up and structure_trend == "bearish"
    choch_down = broke_down and structure_trend == "bullish"
    bos_up = broke_up and not choch_up
    bos_down = broke_down and not choch_down

    # Search recent candles for liquidity sweeps against already-confirmed pivots.
    sweep_high_event = None
    sweep_low_event = None
    for i in range(max(2, len(candles) - 10), len(candles)):
        candle = candles[i]
        eligible_highs = [p for p in pivot_highs if p["index"] <= i - 2]
        eligible_lows = [p for p in pivot_lows if p["index"] <= i - 2]
        if eligible_highs:
            level = eligible_highs[-1]["price"]
            if candle["high"] > level and candle["close"] < level:
                sweep_high_event = {"level": level, "time": candle["time"]}
        if eligible_lows:
            level = eligible_lows[-1]["price"]
            if candle["low"] < level and candle["close"] > level:
                sweep_low_event = {"level": level, "time": candle["time"]}
    sweep_high = sweep_high_event is not None
    sweep_low = sweep_low_event is not None

    # Keep the latest unfilled, meaningful three-candle FVG.
    bullish_fvg_zone = None
    bearish_fvg_zone = None
    min_gap = max(atr * 0.05, abs(last["close"]) * 0.00005)
    for i in range(max(2, len(candles) - 30), len(candles)):
        older = candles[i - 2]
        current = candles[i]
        if current["low"] - older["high"] >= min_gap:
            zone_low, zone_high = older["high"], current["low"]
            filled = any(candles[j]["low"] <= zone_low for j in range(i + 1, len(candles)))
            if not filled:
                bullish_fvg_zone = {"low": zone_low, "high": zone_high, "time": current["time"]}
        if older["low"] - current["high"] >= min_gap:
            zone_low, zone_high = current["high"], older["low"]
            filled = any(candles[j]["high"] >= zone_high for j in range(i + 1, len(candles)))
            if not filled:
                bearish_fvg_zone = {"low": zone_low, "high": zone_high, "time": current["time"]}

    bullish_fvg = bullish_fvg_zone is not None
    bearish_fvg = bearish_fvg_zone is not None
    avg_volume = sum(c["volume"] for c in candles[-21:-1]) / 20
    volume_confirm = last["volume"] > avg_volume if avg_volume > 0 else False

    # Confluence scores include momentum as well as market structure.
    bull = 0
    bear = 0
    bull += 10 if e9 > e21 else 0
    bear += 10 if e9 < e21 else 0
    bull += 8 if e21 > e50 else 0
    bear += 8 if e21 < e50 else 0
    bull += 6 if last["close"] > e21 else 0
    bear += 6 if last["close"] < e21 else 0
    bull += 8 if structure_trend == "bullish" else 0
    bear += 8 if structure_trend == "bearish" else 0
    bull += 18 if bos_up else 0
    bear += 18 if bos_down else 0
    bull += 12 if choch_up else 0
    bear += 12 if choch_down else 0
    bull += 14 if sweep_low else 0
    bear += 14 if sweep_high else 0
    bull += 6 if bullish_fvg else 0
    bear += 6 if bearish_fvg else 0
    bull += 8 if rsi_value >= 52 else 0
    bear += 8 if rsi_value <= 48 else 0
    bull += 6 if macd_bull else 0
    bear += 6 if macd_bear else 0
    bull += 4 if momentum_up else 0
    bear += 4 if momentum_down else 0
    adx_data = advanced["adx"]
    bull += 4 if adx_data["value"] >= 20 and adx_data["plus_di"] > adx_data["minus_di"] else 0
    bear += 4 if adx_data["value"] >= 20 and adx_data["minus_di"] > adx_data["plus_di"] else 0
    bull += 4 if volume_confirm and last["close"] >= previous["close"] else 0
    bear += 4 if volume_confirm and last["close"] <= previous["close"] else 0
    bull = min(100, bull)
    bear = min(100, bear)

    # If a formal structure event is absent, require at least three momentum/trend checks.
    bull_structure_ok = bos_up or choch_up or sweep_low or bullish_fvg
    bear_structure_ok = bos_down or choch_down or sweep_high or bearish_fvg
    bull_momentum_checks = sum((
        e9 > e21, e21 > e50, last["close"] > e21,
        rsi_value >= 52, macd_bull, momentum_up
    ))
    bear_momentum_checks = sum((
        e9 < e21, e21 < e50, last["close"] < e21,
        rsi_value <= 48, macd_bear, momentum_down
    ))
    bull_confirmed = bull_structure_ok or bull_momentum_checks >= 3
    bear_confirmed = bear_structure_ok or bear_momentum_checks >= 3

    direction = "BUY" if bull >= 42 and bull - bear >= 8 and bull_confirmed else (
        "SELL" if bear >= 42 and bear - bull >= 8 and bear_confirmed else "WAIT"
    )
    bias = "صعودی" if bull - bear >= 6 else "نزولی" if bear - bull >= 6 else "خنثی"
    setup_score = min(100, max(bull, bear))
    price = last["close"]

    if direction == "BUY":
        structural_gap = price - swing_low
        buffer = max(atr * 0.20, price * 0.0005)
        stop = swing_low - buffer if 0 < structural_gap <= 3 * atr else price - 1.5 * atr
        risk = max(price - stop, atr * 0.5)
        targets = [price + risk * r for r in (1.5, 2.5, 3.5)]
    elif direction == "SELL":
        structural_gap = swing_high - price
        buffer = max(atr * 0.20, price * 0.0005)
        stop = swing_high + buffer if 0 < structural_gap <= 3 * atr else price + 1.5 * atr
        risk = max(stop - price, atr * 0.5)
        targets = [price - risk * r for r in (1.5, 2.5, 3.5)]
    else:
        stop, targets, risk = None, [None, None, None], None

    if choch_up:
        choch = "صعودی"
    elif choch_down:
        choch = "نزولی"
    else:
        choch = "تأیید نشده"

    order_block = None
    if bos_up or choch_up:
        for candle in reversed(candles[-16:-1]):
            if candle["close"] < candle["open"]:
                order_block = {"type": "تقریبی صعودی", "low": candle["low"], "high": candle["high"]}
                break
    elif bos_down or choch_down:
        for candle in reversed(candles[-16:-1]):
            if candle["close"] > candle["open"]:
                order_block = {"type": "تقریبی نزولی", "low": candle["low"], "high": candle["high"]}
                break

    notes = [
        "EMA 9/21: " + ("صعودی" if e9 > e21 else "نزولی"),
        "EMA 21/50: " + ("صعودی" if e21 > e50 else "نزولی"),
        "RSI(14): " + str(round(rsi_value, 1)),
        "MACD مومنتوم: " + ("صعودی" if macd_bull else "نزولی" if macd_bear else "ضعیف/متناقض"),
        "ساختار سوئینگ: " + {
            "bullish": "سقف و کف بالاتر (صعودی)",
            "bearish": "سقف و کف پایین‌تر (نزولی)",
            "expanding_range": "دامنه در حال گسترش/متناقض",
            "unknown": "ساختار سوئینگ روشن نیست"
        }[structure_trend],
        "شکست ساختار: " + ("BOS صعودی" if bos_up else "BOS نزولی" if bos_down else "شکست تأییدشده‌ای دیده نشد"),
        "تغییر کاراکتر (CHOCH): " + choch,
        "لیکوییدیتی: " + ("سوییپ نقدینگی زیر کف در ۱۰ کندل اخیر" if sweep_low else "سوییپ نقدینگی بالای سقف در ۱۰ کندل اخیر" if sweep_high else "سوییپ تأییدشده‌ای در ۱۰ کندل اخیر پیدا نشد"),
        "FVG: " + ("ناحیه صعودی باز در کندل‌های اخیر" if bullish_fvg else "ناحیه نزولی باز در کندل‌های اخیر" if bearish_fvg else "FVG باز و معناداری پیدا نشد"),
        "حجم: " + ("بالاتر از میانگین ۲۰ کندل" if volume_confirm else "تأیید حجمی ندارد"),
        "امتیاز خریدار/فروشنده: " + str(bull) + "/" + str(bear)
    ]

    wait_reasons = []
    if direction == "WAIT":
        if max(bull, bear) < 42:
            wait_reasons.append("امتیاز هم‌جهتی به حداقل ۴۲ نرسیده")
        if abs(bull - bear) < 8:
            wait_reasons.append("اختلاف امتیاز خریدار و فروشنده کمتر از ۸ است")
        if bull - bear >= 10 and not bull_confirmed:
            wait_reasons.append("تأیید کافی برای ورود خرید وجود ندارد")
        if bear - bull >= 10 and not bear_confirmed:
            wait_reasons.append("تأیید کافی برای ورود فروش وجود ندارد")
        notes.append("علت WAIT: " + "؛ ".join(wait_reasons or ["شواهد کافی برای ورود وجود ندارد"]))

    return {
        "market": market, "symbol": symbol, "timeframe": timeframe,
        "signal": direction, "bias": bias,
        "setup_score": setup_score, "bull_score": bull, "bear_score": bear,
        "rsi": round(rsi_value, 2), "macd_momentum": "bullish" if macd_bull else "bearish" if macd_bear else "neutral",
        "bull_momentum_checks": bull_momentum_checks, "bear_momentum_checks": bear_momentum_checks,
        "score_note": "امتیاز تجربی هم‌جهتی شروط است و احتمال موفقیت معامله نیست.",
        "wait_reason": wait_reasons,
        "market_structure": structure_trend,
        "price": price, "atr": atr, "stop_loss": stop,
        "take_profit_1": targets[0], "take_profit_2": targets[1], "take_profit_3": targets[2],
        "risk_per_unit": risk, "swing_high": swing_high, "swing_low": swing_low,
        "bos": "صعودی" if bos_up else "نزولی" if bos_down else "ندارد",
        "choch": choch, "liquidity_sweep": "پایین" if sweep_low else "بالا" if sweep_high else "ندارد",
        "liquidity_sweep_level": (sweep_low_event or sweep_high_event or {}).get("level"),
        "fvg": "صعودی" if bullish_fvg else "نزولی" if bearish_fvg else "ندارد",
        "fvg_zone": bullish_fvg_zone or bearish_fvg_zone,
        "order_block_approx": order_block, "analysis": notes,
        "candles_count": len(candles), "last_candle_time": last["time"],
        "advanced_analysis": advanced,
        "risk_management": {
            "risk_per_trade_percent_reference": 0.5,
            "risk_budget_per_10000_account_value": 50.0,
            "position_size_formula": "risk_budget / abs(entry - stop_loss)",
            "max_daily_loss_reference_percent": 2.0,
            "note": "مقادیر صرفاً الگوی مدیریت ریسک هستند؛ اندازه قرارداد، اهرم، کارمزد، اسلیپیج و سرمایه واقعی باید لحاظ شوند."
        },
        "updated_at": utc_now()
    }
@app.get("/", response_class=HTMLResponse)
def home():
    path = os.path.join(ROOT_DIR, "static", "index.html")
    with open(path, encoding="utf-8") as file:
        return file.read()


@app.get("/health")
def health():
    return {"status": "ok", "app": "Smart Money AI", "time": utc_now()}


@app.get("/api/status")
def status():
    return {
        "status": "online",
        "markets": ["crypto", "gold", "silver"],
        "forex_enabled": False,
        "providers": {
            "crypto_prices": "Binance public market data",
            "crypto_derivatives": "Binance public futures funding/OI/long-short + spot order-book snapshot",
            "metals_prices": ("Twelve Data primary + Yahoo Finance fallback" if TWELVE_DATA_API_KEY else "Yahoo Finance public fallback; Twelve Data key optional"),
            "crypto_dominance": "CoinGecko " + ("key_configured" if COINGECKO_API_KEY else "public_endpoint_attempt"),
            "global_news": "GDELT DOC 2.0",
            "macro_indicators": "FRED " + ("configured" if FRED_API_KEY else "key_required"),
            "ai_synthesis": "OpenAI " + ("configured" if AI_API_KEY else "key_required")
        },
        "live_trading": False,
        "supported_timeframes": list(TIMEFRAMES.keys())
    }


@app.get("/api/options")
def options():
    return {
        "crypto": CRYPTO,
        "gold": [METALS["gold"]],
        "silver": [METALS["silver"]],
        "timeframes": list(TIMEFRAMES.keys()),
        "forex": []
    }


DERIVATIVES_CACHE = {}


def get_crypto_derivatives(symbol):
    now = time.time()
    cached = DERIVATIVES_CACHE.get(symbol)
    if cached and now - cached["at"] < 30:
        return cached["data"]

    base = urllib.parse.quote(symbol, safe="")
    urls = {
        "funding": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=" + base + "&limit=1",
        "open_interest": "https://fapi.binance.com/fapi/v1/openInterest?symbol=" + base,
        "long_short": "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=" + base + "&period=1h&limit=1",
        "taker_ratio": "https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=" + base + "&period=1h&limit=1",
        "order_book": "https://data-api.binance.vision/api/v3/depth?symbol=" + base + "&limit=20"
    }
    values, errors = {}, []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_json, url, None, 7): key for key, url in urls.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                values[key] = future.result()
            except Exception as exc:
                errors.append(key + ": " + str(exc)[:100])

    funding_rows = values.get("funding") or []
    funding = float(funding_rows[-1]["fundingRate"]) if funding_rows else None
    oi_data = values.get("open_interest") or {}
    long_rows = values.get("long_short") or []
    taker_rows = values.get("taker_ratio") or []
    depth = values.get("order_book") or {}
    bids, asks = depth.get("bids", []), depth.get("asks", [])
    bid_qty = sum(float(row[1]) for row in bids) if bids else 0.0
    ask_qty = sum(float(row[1]) for row in asks) if asks else 0.0
    total_qty = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total_qty if total_qty > 0 else None
    spread_bps = None
    if bids and asks:
        bid, ask = float(bids[0][0]), float(asks[0][0])
        mid = (bid + ask) / 2.0
        spread_bps = ((ask - bid) / mid * 10000.0) if mid else None

    result = {
        "available": bool(values),
        "source": "Binance public USD-M futures metrics + spot order-book snapshot",
        "funding_rate": funding,
        "funding_rate_percent": funding * 100 if funding is not None else None,
        "open_interest_contracts": float(oi_data["openInterest"]) if oi_data.get("openInterest") is not None else None,
        "long_short_account_ratio": float(long_rows[-1]["longShortRatio"]) if long_rows else None,
        "taker_buy_sell_ratio": float(taker_rows[-1]["buySellRatio"]) if taker_rows else None,
        "spot_orderbook_imbalance_top20": imbalance,
        "spot_spread_basis_points": spread_bps,
        "errors": errors,
        "note": "فاندینگ و OI مشتقات عمومی‌اند؛ نسبت لانگ/شورت تعداد حساب‌هاست. دفتر سفارش فقط snapshot اسپات است، نه جریان سفارش تاریخی یا سفارش نهادی."
    }
    DERIVATIVES_CACHE[symbol] = {"at": now, "data": result}
    return result


@app.get("/api/analyze")
def api_analyze(market: str = "crypto", symbol: str = "BTCUSDT", timeframe: str = "15m"):
    candles = get_candles(market, symbol, timeframe)
    result = analyze(candles, market, symbol, timeframe)
    if market == "crypto":
        result["derivatives"] = get_crypto_derivatives(symbol)
    else:
        result["derivatives"] = {
            "available": False,
            "note": "منبع عمومی فعلی برای مشتقات و دفتر سفارش طلا/نقره به این برنامه متصل نیست."
        }
    return result


@app.get("/api/backtest")
def api_backtest(market: str = "crypto", symbol: str = "BTCUSDT", timeframe: str = "1h"):
    candles = completed_candles(get_candles(market, symbol, timeframe), timeframe)
    result = run_research_backtest(candles, timeframe)
    result.update({"market": market, "symbol": symbol, "updated_at": utc_now()})
    return result


@app.get("/api/multi-timeframe")
def multi_timeframe(market: str = "crypto", symbol: str = "BTCUSDT"):
    timeframes = ("5m", "15m", "1h", "4h", "1d", "1w")
    weights = {"5m": 1, "15m": 1, "1h": 2, "4h": 3, "1d": 4, "1w": 5}
    by_timeframe = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        future_map = {
            pool.submit(get_candles, market, symbol, tf): tf for tf in timeframes
        }
        for future in as_completed(future_map):
            tf = future_map[future]
            try:
                by_timeframe[tf] = analyze(future.result(), market, symbol, tf)
            except Exception as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)[:150]
                by_timeframe[tf] = {"timeframe": tf, "error": detail}
    results = [by_timeframe[tf] for tf in timeframes]
    available = [item for item in results if "signal" in item]
    weighted_score = sum(
        weights[item["timeframe"]] * (1 if item["signal"] == "BUY" else -1 if item["signal"] == "SELL" else 0)
        for item in available
    )
    overall = "BUY" if weighted_score >= 5 else "SELL" if weighted_score <= -5 else "WAIT"
    total_weight = sum(weights[item["timeframe"]] for item in available)
    alignment = round(abs(weighted_score) / total_weight * 100, 1) if total_weight else 0.0
    return {
        "market": market, "symbol": symbol, "overall": overall,
        "weighted_score": weighted_score, "alignment_percent": alignment,
        "timeframes": results, "updated_at": utc_now(),
        "note": "تایم‌فریم‌های بالاتر وزن بیشتری دارند؛ امتیاز وزنی احتمال موفقیت نیست."
    }


@app.get("/api/dominance")
def dominance():
    headers = {"x-cg-demo-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
    try:
        data = fetch_json("https://api.coingecko.com/api/v3/global", headers=headers)
        global_data = data.get("data", {})
        shares = global_data.get("market_cap_percentage", {})
        btc = float(shares.get("btc", 0))
        eth = float(shares.get("eth", 0))
        usdt = float(shares.get("usdt", 0))
        return {
            "total_market_cap_usd": global_data.get("total_market_cap", {}).get("usd"),
            "total_volume_24h_usd": global_data.get("total_volume", {}).get("usd"),
            "btc": btc, "eth": eth, "usdt": usdt,
            "others_ex_btc_eth_usdt": max(0, 100 - btc - eth - usdt),
            "updated_at": utc_now(), "source": "CoinGecko",
            "note": "Others is calculated as the remaining share after BTC, ETH and USDT."
        }
    except Exception as exc:
        raise HTTPException(502, "داده دامیننس فعلاً در دسترس نیست؛ وضعیت API CoinGecko را بررسی کن. " + str(exc)[:120])


def news_query(asset):
    if asset == "gold":
        return '("gold" OR "bullion" OR "XAU") (inflation OR "Federal Reserve" OR dollar OR yields OR war OR sanctions OR "central bank")'
    if asset == "silver":
        return '("silver" OR "XAG") (solar OR industrial OR mine OR supply OR demand OR inflation OR dollar OR yields)'
    return '(bitcoin OR ethereum OR cryptocurrency OR crypto) (ETF OR regulation OR exchange OR hack OR liquidity OR "interest rates")'


def classify_news(title, asset):
    text = title.lower()
    tags = []
    if any(word in text for word in ("war", "attack", "conflict", "sanction", "invasion", "crisis")):
        tags.append("ریسک ژئوپلیتیک")
    if any(word in text for word in ("rate cut", "cuts rates", "dovish", "lower rates")):
        tags.append("انتظارات سیاست پولی انبساطی")
    if any(word in text for word in ("rate hike", "higher rates", "hawkish", "yield rises", "yields rise")):
        tags.append("فشار احتمالی نرخ بهره/بازده")
    if any(word in text for word in ("inflation", "cpi", "pce")):
        tags.append("تورم")
    if any(word in text for word in ("dollar", "dxy", "usd")):
        tags.append("دلار آمریکا")
    if asset == "silver" and any(word in text for word in ("solar", "electronics", "industrial", "mine", "mining")):
        tags.append("عرضه یا تقاضای صنعتی نقره")
    if asset == "crypto" and any(word in text for word in ("etf", "sec", "regulation", "exchange", "hack")):
        tags.append("قانون‌گذاری/جریان سرمایه کریپتو")
    return tags or ["خبر مرتبط؛ اثر نیازمند بررسی متن کامل"]


@app.get("/api/news")
def news(asset: str = "gold"):
    global GDELT_LAST_REQUEST_AT
    if asset not in ("gold", "silver", "crypto"):
        raise HTTPException(400, "asset باید gold، silver یا crypto باشد")
    now = datetime.now(timezone.utc).timestamp()
    cached = NEWS_CACHE.get(asset)
    if cached:
        cache_ttl = 600 if cached["data"].get("available", True) else 60
        if now - cached["at"] < cache_ttl:
            return cached["data"]

    params = urllib.parse.urlencode({
        "query": news_query(asset), "mode": "ArtList", "format": "json",
        "maxrecords": 15, "timespan": "3d", "sort": "DateDesc"
    })
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + params
    try:
        # GDELT asks clients to keep requests at least five seconds apart.
        # Serialize requests so simultaneous dashboard calls respect that limit.
        with GDELT_LOCK:
            wait_seconds = 5.2 - (time.monotonic() - GDELT_LAST_REQUEST_AT)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            GDELT_LAST_REQUEST_AT = time.monotonic()
            try:
                raw = fetch_json(url, timeout=20)
            except RuntimeError as first_error:
                if "HTTP 429" not in str(first_error):
                    raise
                # One delayed retry handles a transient upstream rate-limit response.
                time.sleep(5.5)
                GDELT_LAST_REQUEST_AT = time.monotonic()
                raw = fetch_json(url, timeout=20)

        articles = []
        for item in raw.get("articles", []):
            title = item.get("title", "")
            articles.append({
                "title": title, "url": item.get("url"),
                "source": item.get("domain") or item.get("sourceCountry") or "GDELT",
                "published": item.get("seendate"),
                "tags": classify_news(title, asset)
            })
        result = {
            "asset": asset, "articles": articles, "count": len(articles),
            "available": True, "stale": False,
            "source": "GDELT DOC 2.0", "updated_at": utc_now(),
            "disclaimer": "برچسب‌ها بر اساس کلمات تیتر هستند؛ تحلیل قطعی اثر قیمت یا جایگزین مطالعه خبر نیستند."
        }
        NEWS_CACHE[asset] = {"at": now, "data": result}
        return result
    except Exception as exc:
        # Prefer clearly labelled cached headlines over breaking the whole news panel.
        if cached and cached.get("data", {}).get("articles"):
            stale_result = dict(cached["data"])
            stale_result["stale"] = True
            stale_result["warning"] = "اتصال به GDELT موقتاً محدود شده؛ این‌ها خبرهای ذخیره‌شده قبلی هستند."
            stale_result["cache_age_seconds"] = int(max(0, now - cached["at"]))
            return stale_result

        result = {
            "asset": asset, "articles": [], "count": 0,
            "available": False, "stale": False,
            "source": "GDELT DOC 2.0",
            "warning": "دریافت خبر موقتاً به‌دلیل محدودیت درخواست یا اختلال منبع ممکن نیست.",
            "error": str(exc)[:180], "updated_at": utc_now()
        }
        NEWS_CACHE[asset] = {"at": now, "data": result}
        return result

FRED_SERIES = {
    "DGS10": {"label": "بازده اوراق ۱۰ ساله آمریکا", "unit": "%"},
    "DGS2": {"label": "بازده اوراق ۲ ساله آمریکا", "unit": "%"},
    "DFII10": {"label": "بازده واقعی اوراق ۱۰ ساله آمریکا", "unit": "%"},
    "T10YIE": {"label": "انتظارات تورمی سربه‌سر ۱۰ ساله", "unit": "%"},
    "FEDFUNDS": {"label": "نرخ مؤثر فدرال رزرو", "unit": "%"},
    "CPIAUCSL": {"label": "شاخص قیمت مصرف‌کننده آمریکا", "unit": "index"},
    "PCEPI": {"label": "شاخص قیمت مخارج مصرف شخصی PCE", "unit": "index"},
    "UNRATE": {"label": "نرخ بیکاری آمریکا", "unit": "%"},
    "PAYEMS": {"label": "اشتغال غیرکشاورزی آمریکا", "unit": "thousands"},
    "DTWEXBGS": {"label": "شاخص گسترده وزنی دلار آمریکا (نه DXY)", "unit": "index"},
    "VIXCLS": {"label": "شاخص نوسان VIX", "unit": "index"},
    "WTISPLC": {"label": "قیمت نفت WTI", "unit": "USD/barrel"},
    "BAMLH0A0HYM2": {"label": "اسپرد اوراق پربازده آمریکا", "unit": "%"}
}


def fred_observation(series_id):
    params = urllib.parse.urlencode({
        "series_id": series_id, "api_key": FRED_API_KEY,
        "file_type": "json", "sort_order": "desc", "limit": 2
    })
    data = fetch_json("https://api.stlouisfed.org/fred/series/observations?" + params)
    values = [x for x in data.get("observations", []) if x.get("value") not in (None, ".", "")]
    if not values:
        return {"series": series_id, "available": False}
    latest = values[0]
    previous = values[1] if len(values) > 1 else None
    return {
        "series": series_id, "label": FRED_SERIES[series_id]["label"],
        "unit": FRED_SERIES[series_id]["unit"], "value": float(latest["value"]),
        "date": latest["date"],
        "previous_value": float(previous["value"]) if previous else None,
        "previous_date": previous["date"] if previous else None,
        "available": True
    }


@app.get("/api/macro")
def macro():
    if not FRED_API_KEY:
        return {
            "available": False, "source": "FRED",
            "message": "برای دریافت شاخص‌های رسمی اقتصاد آمریکا، FRED_API_KEY را در Render تنظیم کن.",
            "indicators": [],
            "not_connected": ["ICE DXY exact index", "COT positioning", "Gold ETF flows", "Silver ETF flows", "central-bank gold purchases"]
        }
    now = datetime.now(timezone.utc).timestamp()
    if MACRO_CACHE["data"] and now - MACRO_CACHE["at"] < 3600:
        return MACRO_CACHE["data"]
    indicators_by_id = {}
    errors = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        future_map = {pool.submit(fred_observation, series_id): series_id for series_id in FRED_SERIES}
        for future in as_completed(future_map):
            series_id = future_map[future]
            try:
                indicators_by_id[series_id] = future.result()
            except Exception as exc:
                errors.append(series_id + ": " + str(exc)[:100])
    indicators = [indicators_by_id[key] for key in FRED_SERIES if key in indicators_by_id]
    result = {
        "available": bool(indicators), "source": "Federal Reserve Economic Data (FRED)",
        "indicators": indicators, "errors": errors,
        "not_connected": ["ICE DXY exact index", "COT positioning", "Gold ETF flows", "Silver ETF flows", "central-bank gold purchases"],
        "updated_at": utc_now(),
        "note": "برخی شاخص‌ها با تأخیر منتشر می‌شوند؛ تاریخ هر مشاهده کنار مقدار آمده است."
    }
    MACRO_CACHE.update({"at": now, "data": result})
    return result


@app.get("/api/market-context")
def market_context(asset: str = "gold"):
    if asset not in ("gold", "silver", "crypto"):
        raise HTTPException(400, "asset نامعتبر است")
    macro_data = macro()
    news_data = news(asset)
    factors = [
        {"factor": "دلار آمریکا", "status": "شاخص گسترده دلار FRED موجود است؛ جایگزین دقیق ICE DXY نیست" if any(x.get("series") == "DTWEXBGS" and x.get("available") for x in macro_data.get("indicators", [])) else "نیازمند منبع داده دلار", "impact": "دلار قوی می‌تواند بر فلزات و دارایی‌های ریسکی فشار وارد کند؛ رابطه قطعی نیست"},
        {"factor": "بازده واقعی اوراق", "status": "داده FRED موجود است" if any(x.get("series") == "DFII10" and x.get("available") for x in macro_data.get("indicators", [])) else "نیازمند FRED_API_KEY", "impact": "افزایش بازده واقعی معمولاً هزینه فرصت نگهداری طلا را بالا می‌برد"},
        {"factor": "نرخ بهره و تورم آمریکا", "status": "داده FRED موجود است" if macro_data.get("available") else "نیازمند FRED_API_KEY", "impact": "انتظارات سیاست پولی می‌تواند بر دلار و فلزات اثر بگذارد"},
        {"factor": "جنگ و تحریم‌ها", "status": "رصد تیترهای خبری فعال", "impact": "ممکن است تقاضای دارایی امن را تغییر دهد"},
        {"factor": "ETF، COT و خرید بانک‌های مرکزی", "status": "هنوز منبع داده متصل نیست", "impact": "برای تحلیل کامل جریان سرمایه نیازمند منبع تخصصی است"}
    ]
    if asset == "silver":
        factors.append({"factor": "تقاضای صنعتی و معادن نقره", "status": "رصد تیترهای خبری فعال", "impact": "تقاضای صنعتی و اختلال عرضه می‌تواند بر توازن بازار اثر بگذارد"})
    return {"asset": asset, "factors": factors, "macro": macro_data, "news": news_data, "updated_at": utc_now()}


@app.get("/api/scanner")
def scanner(market: str = "all", timeframe: str = "15m", limit: int = 10):
    if timeframe not in TIMEFRAMES:
        raise HTTPException(400, "تایم‌فریم نامعتبر است")
    limit = max(1, min(limit, 20))
    tasks = []
    if market in ("all", "crypto"):
        tasks.extend([("crypto", symbol) for symbol in CRYPTO[:limit]])
    if market in ("all", "gold"):
        tasks.append(("gold", METALS["gold"]))
    if market in ("all", "silver"):
        tasks.append(("silver", METALS["silver"]))
    if market not in ("all", "crypto", "gold", "silver"):
        raise HTTPException(400, "بازار نامعتبر است")

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {
            pool.submit(get_candles, m, s, timeframe): (m, s) for m, s in tasks
        }
        for future in as_completed(future_map):
            m, s = future_map[future]
            try:
                result = analyze(future.result(), m, s, timeframe)
                results.append({
                    "market": m, "symbol": s, "signal": result["signal"],
                    "setup_score": result["setup_score"], "price": result["price"],
                    "bias": result["bias"]
                })
            except Exception as exc:
                results.append({"market": m, "symbol": s, "signal": "ERROR", "error": str(exc)[:150]})
    results.sort(key=lambda item: item.get("setup_score", 0), reverse=True)
    return {"timeframe": timeframe, "results": results, "updated_at": utc_now()}


@app.post("/api/ai-synthesis")
async def ai_synthesis(request: Request):
    if not AI_API_KEY:
        raise HTTPException(503, "برای تحلیل هوش مصنوعی، AI_API_KEY را در Render تنظیم کن. این کلید هزینه جداگانه API دارد.")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "بدنه درخواست JSON معتبر نیست")
    context = json.dumps(payload, ensure_ascii=False)[:12000]
    prompt = (
        "تو تحلیلگر محتاط بازار هستی. فقط بر اساس داده‌های ورودی تحلیل کن و هیچ عدد یا خبری نساز. "
        "بین داده زنده، استنباط و داده ناموجود تفاوت بگذار. سود قطعی وعده نده. "
        "خروجی فارسی و ساختاریافته باشد: خلاصه، عوامل صعودی، عوامل نزولی، تعارض‌ها، سناریوها، "
        "شرایط بی‌اعتبار شدن سناریو و نتیجه WAIT در صورت نبود شواهد کافی. "
        "امتیازها احتمال موفقیت نیستند. داده ورودی: " + context
    )
    body = json.dumps({"model": AI_MODEL, "input": prompt, "max_output_tokens": 700}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses", data=body,
        headers={"Authorization": "Bearer " + AI_API_KEY, "Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        chunks = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        chunks.append(part.get("text", ""))
        return {"analysis": "\n".join(chunks), "model": data.get("model", AI_MODEL), "updated_at": utc_now()}
    except Exception as exc:
        raise HTTPException(502, "درخواست AI انجام نشد؛ کلید، دسترسی مدل و اعتبار API را بررسی کن. " + str(exc)[:120])


@app.post("/webhook")
async def webhook(request: Request):
    if not WEBHOOK_TOKEN:
        raise HTTPException(503, "Webhook غیرفعال است؛ ابتدا WEBHOOK_TOKEN را در Render تنظیم کن.")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "JSON نامعتبر است")
    if not hmac.compare_digest(str(data.get("token", "")), WEBHOOK_TOKEN):
        raise HTTPException(401, "Invalid webhook token")
    with db() as con:
        con.execute("""
        INSERT INTO alerts (created_at, symbol, timeframe, signal, price, message)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            utc_now(), str(data.get("symbol", "UNKNOWN"))[:40],
            str(data.get("timeframe", ""))[:12], str(data.get("side", data.get("signal", "")))[:12],
            data.get("price"), json.dumps(data, ensure_ascii=False)[:4000]
        ))
    return {"ok": True}


@app.get("/api/alerts")
def alerts():
    with db() as con:
        rows = con.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 50").fetchall()
    return [dict(row) for row in rows]
