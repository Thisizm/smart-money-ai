import os
import json
import sqlite3
import urllib.parse
import urllib.request
import urllib.error
import hmac
import math
import time
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .pro_analysis import advanced_analysis
from .backtesting import run_research_backtest
from .paper_trading import init_paper_db, record_signal, monitor_open_trades, list_paper_trades

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
DB_PATH = os.getenv("DATABASE_PATH", os.path.join(ROOT_DIR, "alerts.db"))

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
ALLOW_METAL_FUTURES_PROXY = os.getenv("ALLOW_METAL_FUTURES_PROXY", "false").lower() in ("1", "true", "yes")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")

app = FastAPI(title="Smart Money AI | Crypto, Gold & Silver")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "static")), name="static")

CRYPTO_DEFAULT = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
    "TRXUSDT", "DOTUSDT", "BCHUSDT", "NEARUSDT", "UNIUSDT",
    "SUIUSDT", "TONUSDT", "ETCUSDT", "APTUSDT", "ARBUSDT"
]
CRYPTO = list(CRYPTO_DEFAULT)
METALS = {"gold": "XAUUSDT", "silver": "XAGUSDT"}
TWELVE_DATA_METAL_SYMBOLS = {"XAUUSDT": "XAU/USD", "XAGUSDT": "XAG/USD"}
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
QUOTE_CACHE = {}
QUOTE_CACHE_LOCK = threading.Lock()
BINANCE_SYMBOLS_CACHE = {"at": 0.0, "symbols": list(CRYPTO_DEFAULT)}
BINANCE_SYMBOLS_LOCK = threading.Lock()
DOMINANCE_CACHE = {"at": 0.0, "data": None}
DOMINANCE_LOCK = threading.Lock()
MARGINPAD_CACHE = {}
MARGINPAD_LOCK = threading.Lock()
ECONOMIC_CALENDAR_CACHE = {"at": 0.0, "data": None}
MARGINPAD_SCREENER_CACHE = {"at": 0.0, "data": None}


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
init_paper_db()


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


def get_binance_spot_symbols():
    """Return all currently trading Binance Spot USDT pairs, with major pairs first."""
    now = time.time()
    with BINANCE_SYMBOLS_LOCK:
        cached = BINANCE_SYMBOLS_CACHE.get("symbols") or []
        if cached and now - BINANCE_SYMBOLS_CACHE.get("at", 0) < 3600:
            return list(cached)
        try:
            data = fetch_json(
                "https://data-api.binance.vision/api/v3/exchangeInfo?permissions=SPOT",
                timeout=15
            )
            eligible = set()
            for item in data.get("symbols", []):
                if item.get("status") != "TRADING" or item.get("quoteAsset") != "USDT":
                    continue
                if item.get("isSpotTradingAllowed") is False:
                    continue
                symbol = item.get("symbol")
                if symbol:
                    eligible.add(symbol)
            if not eligible:
                raise RuntimeError("Binance returned no active USDT spot pairs")
            preferred = [symbol for symbol in CRYPTO_DEFAULT if symbol in eligible]
            symbols = preferred + sorted(eligible.difference(preferred))
            BINANCE_SYMBOLS_CACHE.update({"at": now, "symbols": symbols})
            return list(symbols)
        except Exception:
            # Keep the app usable during a temporary Binance metadata outage.
            BINANCE_SYMBOLS_CACHE.update({
                "at": now - 3540,
                "symbols": list(CRYPTO_DEFAULT)
            })
            return list(CRYPTO_DEFAULT)



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
    spot_tickers = {"gold": ["XAUUSD=X"], "silver": ["XAGUSD=X"]}
    futures_tickers = {"gold": ["GC=F"], "silver": ["SI=F"]}
    tickers = spot_tickers.get(market)
    if not tickers:
        raise RuntimeError("Unsupported metal market")
    # Never silently substitute futures for spot. A futures proxy is opt-in
    # because it can materially diverge from XAG/USD/XAU/USD.
    if ALLOW_METAL_FUTURES_PROXY:
        tickers = tickers + futures_tickers[market]

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
            for candle in candles:
                if ticker in ("GC=F", "SI=F"):
                    # Futures are a fallback proxy; their price can differ from spot.
                    candle["source"] = "Yahoo Finance futures proxy"
                else:
                    candle["source"] = "Yahoo Finance spot"
                candle["source_symbol"] = ticker
            return candles
        except Exception as exc:
            ticker_errors.append(ticker + ": " + str(exc)[:150])

    raise RuntimeError("Yahoo Finance symbols failed: " + " | ".join(ticker_errors))


def get_twelve_data_metals_candles(symbol, timeframe):
    """Fetch spot precious-metal candles from Twelve Data as the primary metals source."""
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY در تنظیمات Render ثبت نشده است")
    provider_symbol = TWELVE_DATA_METAL_SYMBOLS.get(symbol)
    if not provider_symbol:
        raise RuntimeError("Unsupported Twelve Data metal symbol")
    tf = TIMEFRAMES[timeframe]
    params = urllib.parse.urlencode({
        "symbol": provider_symbol, "interval": tf["twelve"], "outputsize": 250,
        "apikey": TWELVE_DATA_API_KEY, "format": "JSON", "timezone": "UTC"
    })
    data = fetch_json("https://api.twelvedata.com/time_series?" + params, timeout=20)
    if "values" not in data:
        message = data.get("message") or data.get("status") or "No time-series values"
        raise RuntimeError("Twelve Data " + provider_symbol + ": " + str(message)[:180])
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
            "volume": float(row.get("volume") or 0),
            "source": "Twelve Data Commodity Aggregate",
            "source_symbol": provider_symbol
        })
    if len(candles) < 50:
        raise RuntimeError("Twelve Data returned insufficient candles for " + provider_symbol)
    return candles


def get_yahoo_metals_quote(market):
    """No-key fallback: spot first, then a clearly labelled COMEX futures proxy."""
    spot_tickers = {"gold": "XAUUSD=X", "silver": "XAGUSD=X"}
    futures_tickers = {"gold": "GC=F", "silver": "SI=F"}
    if market not in spot_tickers:
        raise RuntimeError("Unsupported Yahoo Finance metal")

    tickers = [spot_tickers[market]]
    # A COMEX continuous future is not a spot quote. Only use it when explicitly enabled.
    if ALLOW_METAL_FUTURES_PROXY:
        tickers.append(futures_tickers[market])

    errors = []
    for ticker in tickers:
        try:
            url = (
                "https://query1.finance.yahoo.com/v8/finance/chart/"
                + urllib.parse.quote(ticker, safe="=")
                + "?" + urllib.parse.urlencode({"range": "1d", "interval": "1m"})
            )
            data = fetch_json(url, headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; SmartMoneyAI/2.2)"
            }, timeout=15)
            chart = data.get("chart", {})
            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))
            result = (chart.get("result") or [None])[0]
            if not result:
                raise RuntimeError("Yahoo Finance returned no quote data for " + ticker)
            meta = result.get("meta", {})
            price = meta.get("regularMarketPrice")
            stamp = meta.get("regularMarketTime")
            if price is None:
                quote = (result.get("indicators", {}).get("quote") or [{}])[0]
                closes = [value for value in (quote.get("close") or []) if value is not None]
                if closes:
                    price = closes[-1]
                    timestamps = result.get("timestamp") or []
                    stamp = timestamps[-1] if timestamps else stamp
            if price is None:
                raise RuntimeError("Yahoo Finance returned no usable price for " + ticker)

            is_proxy = ticker in futures_tickers.values()
            stamp = int(stamp or time.time())
            return {
                "available": True, "price": float(price),
                "source": "Yahoo Finance COMEX futures quote" if is_proxy else "Yahoo Finance spot quote",
                "source_symbol": ticker,
                "source_timestamp": datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat(),
                "age_seconds": max(0, int(time.time() - stamp)),
                "is_proxy": is_proxy,
                "note": (
                    "هشدار: SI=F قرارداد آتی پیوسته COMEX و قیمت آن ممکن است با اسپات XAG/USD و قرارداد Binance تفاوت داشته باشد؛ داده Yahoo نیز با تأخیر است."
                    if is_proxy else
                    "قیمت مرجع اسپات از Yahoo Finance؛ ممکن است با تأخیر باشد و با فیدهای دیگر تفاوت داشته باشد."
                )
            }
        except Exception as exc:
            errors.append(ticker + ": " + str(exc)[:150])

    raise RuntimeError("Yahoo Finance quote symbols failed: " + " | ".join(errors))

def get_live_quote(market, symbol):
    """Fetch a separate live/last quote; candle analysis continues to use closed bars."""
    cache_key = market + ":" + symbol
    now = time.time()
    with QUOTE_CACHE_LOCK:
        cached = QUOTE_CACHE.get(cache_key)
        cache_ttl = 30 if market in METALS else 3
        if cached and now - cached["at"] < cache_ttl:
            cached_data = dict(cached["data"])
            source_timestamp = cached_data.get("source_timestamp")
            if source_timestamp:
                try:
                    source_epoch = datetime.fromisoformat(source_timestamp).timestamp()
                    cached_data["age_seconds"] = max(0, int(now - source_epoch))
                except (TypeError, ValueError):
                    pass
            return cached_data

    errors = []
    result = None
    if market == "crypto":
        try:
            params = urllib.parse.urlencode({"symbol": symbol})
            data = fetch_json("https://data-api.binance.vision/api/v3/ticker/price?" + params, timeout=8)
            price = float(data["price"])
            # This is the time our server observed the latest last-traded price.
            stamp_ms = int(now * 1000)
            result = {
                "available": True, "price": price, "source": "Binance Spot last-price (polling)",
                "source_symbol": symbol, "source_timestamp": datetime.fromtimestamp(stamp_ms / 1000, tz=timezone.utc).isoformat(),
                "age_seconds": max(0, int(now - stamp_ms / 1000)), "is_proxy": False,
                "note": "قیمت آخرین معامله اسپات Binance؛ با نماد یا صرافی دیگر TradingView ممکن است تفاوت داشته باشد."
            }
        except Exception as exc:
            errors.append("Binance ticker: " + str(exc)[:140])

    elif market in METALS:
        # Prefer Twelve Data spot quotes for both metals; never route metals through Binance.
        if TWELVE_DATA_API_KEY:
            try:
                provider_symbol = TWELVE_DATA_METAL_SYMBOLS[symbol]
                params = urllib.parse.urlencode({
                    "symbol": provider_symbol, "apikey": TWELVE_DATA_API_KEY,
                    "format": "JSON", "timezone": "UTC"
                })
                data = fetch_json("https://api.twelvedata.com/quote?" + params, timeout=15)
                if "close" not in data:
                    raise RuntimeError(data.get("message") or "Twelve Data quote has no close price")
                price = float(data["close"])
                stamp = int(data.get("last_quote_at") or data.get("timestamp") or 0)
                result = {
                    "available": True, "price": price,
                    "source": "Twelve Data Commodity Aggregate",
                    "source_symbol": provider_symbol,
                    "source_timestamp": datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat() if stamp else None,
                    "age_seconds": max(0, int(now - stamp)) if stamp else None,
                    "is_proxy": False,
                    "is_market_open": data.get("is_market_open"),
                    "note": "قیمت اسپات " + provider_symbol + " از Twelve Data؛ ممکن است با فید CAPITALCOM در TradingView اختلاف جزئی داشته باشد."
                }
            except Exception as twelve_exc:
                errors.append("Twelve Data quote: " + str(twelve_exc)[:140])

        if result is None:
            try:
                result = get_yahoo_metals_quote(market)
            except Exception as yahoo_exc:
                errors.append("Yahoo Finance spot quote: " + str(yahoo_exc)[:140])


    if result is None:
        result = {
            "available": False, "price": None, "source": None,
            "age_seconds": None, "errors": errors,
            "note": "قیمت زنده دریافت نشد؛ قیمت کندل بسته‌شده را به‌عنوان قیمت زنده فرض نکن."
        }

    with QUOTE_CACHE_LOCK:
        QUOTE_CACHE[cache_key] = {"at": now, "data": result}
    return result


def get_candles(market, symbol, timeframe):
    if timeframe not in TIMEFRAMES:
        raise HTTPException(400, "تایم‌فریم نامعتبر است")
    tf = TIMEFRAMES[timeframe]

    if market == "crypto":
        if symbol not in get_binance_spot_symbols():
            raise HTTPException(400, "نماد باید یک جفت فعال USDT در Binance Spot باشد")
        # Binance limits each request to 1000 bars. Page backwards from the
        # newest bar so the analysis/backtest receives a longer contiguous history.
        target = max(250, min(int(os.getenv("HISTORICAL_CANDLE_LIMIT", "1500")), 3000))
        rows, end_time = [], None
        try:
            while len(rows) < target:
                query = {"symbol": symbol, "interval": tf["binance"],
                         "limit": min(1000, target - len(rows))}
                if end_time is not None:
                    query["endTime"] = end_time
                batch = fetch_json("https://data-api.binance.vision/api/v3/klines?" + urllib.parse.urlencode(query))
                if not batch:
                    break
                rows = batch + rows
                oldest = int(batch[0][0])
                if len(batch) < query["limit"] or oldest <= 0:
                    break
                end_time = oldest - 1
            # Remove any overlap and preserve chronological order.
            unique = {int(row[0]): row for row in rows}
            rows = [unique[key] for key in sorted(unique)][-target:]
        except Exception as exc:
            raise HTTPException(502, "خطا از منبع کندل Binance Spot: " + str(exc))
        return [{
            "time": int(row[0]), "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]), "volume": float(row[5]),
            "source": "Binance Spot", "source_symbol": symbol
        } for row in rows]

    if market in METALS:
        if symbol != METALS[market]:
            raise HTTPException(400, "نماد فلز نامعتبر است")
        try:
            # Twelve Data is the primary source for BOTH gold and silver spot candles.
            return get_twelve_data_metals_candles(symbol, timeframe)
        except Exception as twelve_exc:
            try:
                # Yahoo Finance is a no-key fallback; source and possible delay are labelled.
                return get_yahoo_metals_candles(market, timeframe)
            except Exception as yahoo_exc:
                raise HTTPException(
                    502,
                    "دریافت کندل اسپات فلز از Twelve Data ناموفق بود (" +
                    str(twelve_exc)[:130] + "). منبع جایگزین Yahoo Finance نیز در دسترس نیست (" +
                    str(yahoo_exc)[:130] + ")."
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
                bullish_fvg_zone = {"low": zone_low, "high": zone_high, "time": current["time"], "formation_index": i}
        if older["low"] - current["high"] >= min_gap:
            zone_low, zone_high = current["high"], older["low"]
            filled = any(candles[j]["high"] >= zone_high for j in range(i + 1, len(candles)))
            if not filled:
                bearish_fvg_zone = {"low": zone_low, "high": zone_high, "time": current["time"], "formation_index": i}

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

    risk_flags = []
    risk_blocked = False
    atr_percent = advanced["volatility"]["atr_percent"]
    max_atr_percent = {"5m": 2.5, "15m": 3.5, "1h": 5.5, "4h": 8.5, "1d": 18.0, "1w": 30.0}.get(timeframe, 10.0)
    structural_event = bos_up or bos_down or choch_up or choch_down or sweep_high or sweep_low
    if atr_percent > max_atr_percent:
        risk_flags.append("BLOCK: نوسان غیرعادی نسبت به تایم‌فریم؛ ورود جدید متوقف شد")
        risk_blocked = True
    if adx_data["value"] < 12 and not structural_event:
        risk_flags.append("BLOCK: قدرت روند بسیار ضعیف و شکست ساختاری معتبر وجود ندارد")
        risk_blocked = True
    if not advanced["data_quality"]["volume_available"]:
        risk_flags.append("حجم قابل اتکا از منبع فعلی موجود نیست؛ تأیید حجم غیرفعال است")
    if direction == "BUY" and advanced["premium_discount"]["zone"] == "premium":
        risk_flags.append("خرید در نیمه Premium دامنه؛ خطر تعقیب قیمت")
    if direction == "SELL" and advanced["premium_discount"]["zone"] == "discount":
        risk_flags.append("فروش در نیمه Discount دامنه؛ خطر تعقیب قیمت")
    if risk_blocked:
        direction = "WAIT"

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

    if direction in ("BUY", "SELL") and price > 0 and risk is not None and risk / price > 0.06:
        risk_flags.append("BLOCK: فاصله حدضرر بیش از ۶٪ قیمت است؛ نسبت ریسک به سرمایه نامناسب است")
        risk_blocked = True
        direction = "WAIT"
        stop, targets, risk = None, [None, None, None], None

    if choch_up:
        choch = "صعودی"
    elif choch_down:
        choch = "نزولی"
    else:
        choch = "تأیید نشده"

    order_block = None
    search_start = max(0, len(candles) - 24)
    if bos_up or choch_up:
        for idx in range(len(candles) - 2, search_start - 1, -1):
            candidate = candles[idx]
            if candidate["close"] >= candidate["open"]:
                continue
            impulse_index = None
            for j in range(idx + 1, min(len(candles), idx + 6)):
                impulse = candles[j]
                body_size = abs(impulse["close"] - impulse["open"])
                if impulse["close"] > impulse["open"] and impulse["close"] > candidate["high"] and body_size >= 0.8 * atr:
                    impulse_index = j
                    break
            if impulse_index is not None:
                mitigated = any(
                    candles[j]["low"] <= candidate["high"] and candles[j]["high"] >= candidate["low"]
                    for j in range(impulse_index + 1, len(candles))
                )
                order_block = {
                    "type": "تقریبی صعودی", "low": candidate["low"], "high": candidate["high"],
                    "origin_time": candidate["time"], "displacement_time": candles[impulse_index]["time"],
                    "mitigated": mitigated, "validation": "opposite candle + bullish displacement close above zone"
                }
                break
    elif bos_down or choch_down:
        for idx in range(len(candles) - 2, search_start - 1, -1):
            candidate = candles[idx]
            if candidate["close"] <= candidate["open"]:
                continue
            impulse_index = None
            for j in range(idx + 1, min(len(candles), idx + 6)):
                impulse = candles[j]
                body_size = abs(impulse["close"] - impulse["open"])
                if impulse["close"] < impulse["open"] and impulse["close"] < candidate["low"] and body_size >= 0.8 * atr:
                    impulse_index = j
                    break
            if impulse_index is not None:
                mitigated = any(
                    candles[j]["high"] >= candidate["low"] and candles[j]["low"] <= candidate["high"]
                    for j in range(impulse_index + 1, len(candles))
                )
                order_block = {
                    "type": "تقریبی نزولی", "low": candidate["low"], "high": candidate["high"],
                    "origin_time": candidate["time"], "displacement_time": candles[impulse_index]["time"],
                    "mitigated": mitigated, "validation": "opposite candle + bearish displacement close below zone"
                }
                break

    order_block_adjustment = 0
    if order_block and atr > 0:
        zone_low = float(order_block["low"])
        zone_high = float(order_block["high"])
        distance = max(0.0, zone_low - price, price - zone_high)
        distance_atr = distance / atr
        order_block["distance_atr"] = round(distance_atr, 3)
        order_block["relevant"] = bool(not order_block.get("mitigated") and distance_atr <= 2.0)
        if order_block["relevant"] and direction == "BUY" and "صعودی" in order_block.get("type", ""):
            bull = min(100, bull + 2)
            order_block_adjustment = 2
        elif order_block["relevant"] and direction == "SELL" and "نزولی" in order_block.get("type", ""):
            bear = min(100, bear + 2)
            order_block_adjustment = 2
        setup_score = min(100, max(bull, bear))
        bias = "صعودی" if bull - bear >= 6 else "نزولی" if bear - bull >= 6 else "خنثی"
        order_block["score_adjustment"] = order_block_adjustment

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
        "Order Block: " + ("ناحیه فعال و نزدیک قیمت؛ تأیید کم‌وزن +۲ امتیاز" if order_block_adjustment else "ناحیه معتبرِ نزدیک قیمت برای تأیید سیگنال پیدا نشد"),
        "حجم: " + ("بالاتر از میانگین ۲۰ کندل" if volume_confirm else "تأیید حجمی ندارد"),
        "امتیاز خریدار/فروشنده: " + str(bull) + "/" + str(bear)
    ]

    wait_reasons = []
    if direction == "WAIT":
        wait_reasons.extend(risk_flags)
        if max(bull, bear) < 42:
            wait_reasons.append("امتیاز هم‌جهتی به حداقل ۴۲ نرسیده")
        if abs(bull - bear) < 8:
            wait_reasons.append("اختلاف امتیاز خریدار و فروشنده کمتر از ۸ است")
        if bull - bear >= 10 and not bull_confirmed:
            wait_reasons.append("تأیید کافی برای ورود خرید وجود ندارد")
        if bear - bull >= 10 and not bear_confirmed:
            wait_reasons.append("تأیید کافی برای ورود فروش وجود ندارد")
        notes.append("علت WAIT: " + "؛ ".join(wait_reasons or ["شواهد کافی برای ورود وجود ندارد"]))

    # Conditional breakout scenarios are separate from an active trade signal.
    scenario_window = candles[-21:-1] if len(candles) > 21 else candles[:-1]
    range_high = max(c["high"] for c in scenario_window) if scenario_window else swing_high
    range_low = min(c["low"] for c in scenario_window) if scenario_window else swing_low
    breakout_buffer = max(atr * 0.12, abs(price) * 0.0002)
    bullish_trigger = range_high + breakout_buffer
    bearish_trigger = max(0.0, range_low - breakout_buffer)
    bullish_risk = max(atr * 1.2, breakout_buffer * 3)
    bearish_risk = max(atr * 1.2, breakout_buffer * 3)
    bullish_stop = max(0.0, bullish_trigger - bullish_risk)
    bearish_stop = bearish_trigger + bearish_risk
    bullish_targets = [bullish_trigger + bullish_risk * r for r in (1.5, 2.5, 3.5)]
    bearish_targets = [max(0.0, bearish_trigger - bearish_risk * r) for r in (1.5, 2.5, 3.5)]
    entry_price = price if direction in ("BUY", "SELL") else None
    if price >= range_high - atr * 0.25:
        location_text = "قیمت در نزدیکی سقف محدوده اخیر و مقاومت کوتاه‌مدت قرار دارد."
    elif price <= range_low + atr * 0.25:
        location_text = "قیمت در نزدیکی کف محدوده اخیر و حمایت کوتاه‌مدت قرار دارد."
    else:
        location_text = "قیمت فعلاً داخل محدوده نوسان اخیر قرار دارد و هنوز از سقف/کف آن فاصله دارد."

    if direction == "BUY":
        signal_text = (
            "سیگنال خرید فعال است. قیمت ورود مرجع بر اساس آخرین کندل بسته‌شده محاسبه شده؛ "
            "ورود با قیمت زنده ممکن است متفاوت باشد. حد ضرر و اهداف روی نمودار مشخص شده‌اند."
        )
    elif direction == "SELL":
        signal_text = (
            "سیگنال فروش فعال است. قیمت ورود مرجع بر اساس آخرین کندل بسته‌شده محاسبه شده؛ "
            "ورود با قیمت زنده ممکن است متفاوت باشد. حد ضرر و اهداف روی نمودار مشخص شده‌اند."
        )
    else:
        signal_text = "فعلاً سیگنال ورود معتبر نداریم؛ وضعیت WAIT است و سطوح زیر فقط سناریوهای مشروط هستند."

    if price > bullish_trigger:
        bullish_text = (
            "سناریوی صعودی: آخرین کندل بسته‌شده بالای سطح شکست قرار گرفته است؛ "
            "از تعقیب قیمت خودداری کن و تثبیت/پولبک موفق به محدوده "
            + str(round(bullish_trigger, 8)) + " را برای تأیید بررسی کن."
        )
    else:
        bullish_text = (
            "سناریوی صعودی: فقط با بسته‌شدن کندل " + timeframe +
            " بالای " + str(round(bullish_trigger, 8)) +
            " و ترجیحاً افزایش حجم، شکست تأیید می‌شود؛ سپس حفظ سطح در پولبک می‌تواند ورود را معتبرتر کند."
        )
    if price < bearish_trigger:
        bearish_text = (
            "سناریوی نزولی: آخرین کندل بسته‌شده زیر سطح شکست قرار گرفته است؛ "
            "از تعقیب قیمت خودداری کن و پولبک ناموفق به محدوده "
            + str(round(bearish_trigger, 8)) + " را برای تأیید بررسی کن."
        )
    else:
        bearish_text = (
            "سناریوی نزولی: فقط با بسته‌شدن کندل " + timeframe +
            " زیر " + str(round(bearish_trigger, 8)) +
            " و ترجیحاً افزایش حجم، شکست نزولی تأیید می‌شود؛ سپس ناتوانی قیمت در بازپس‌گیری سطح را بررسی کن."
        )
    risk_text = (
        "فیلتر ریسک ورود را مسدود کرده است؛ تا رفع هشدارها هیچ‌یک از سناریوها مجوز ورود نیست."
        if risk_blocked else
        "شکست صرفاً با سایه کندل تأیید نمی‌شود؛ مبنا بسته‌شدن کندل و سپس بررسی پولبک است."
    )
    scenario_analysis = {
        "location": location_text,
        "signal": signal_text,
        "bullish": bullish_text,
        "bearish": bearish_text,
        "risk": risk_text,
        "timeframe": timeframe
    }
    trade_plan = {
        "entry": entry_price,
        "stop_loss": stop if direction in ("BUY", "SELL") else None,
        "targets": targets if direction in ("BUY", "SELL") else [],
        "risk_reward": [1.5, 2.5, 3.5] if direction in ("BUY", "SELL") else [],
        "entry_basis": "آخرین کندل بسته‌شده؛ قیمت اجرای واقعی ممکن است با قیمت زنده و اسلیپیج متفاوت باشد."
    }
    conditional_scenarios = {
        "bullish": {
            "trigger": bullish_trigger, "entry": bullish_trigger,
            "stop_loss": bullish_stop, "targets": bullish_targets,
            "confirmation": "بسته‌شدن کندل بالای سطح شکست و ترجیحاً پولبک موفق"
        },
        "bearish": {
            "trigger": bearish_trigger, "entry": bearish_trigger,
            "stop_loss": bearish_stop, "targets": bearish_targets,
            "confirmation": "بسته‌شدن کندل زیر سطح شکست و ترجیحاً پولبک ناموفق"
        }
    }

    # Smart Entry: propose a pullback zone from unmitigated OB/FVG or EMA21.
    # A zone touch alone is never enough; confirmation requires a closed candle reaction.
    ema21 = e21
    smart_candidates = {"long": [], "short": []}
    if order_block and not order_block.get("mitigated"):
        ob_low, ob_high = float(order_block["low"]), float(order_block["high"])
        if "صعودی" in order_block.get("type", "") and ob_high <= price + atr * 0.25:
            smart_candidates["long"].append(("order_block", ob_low, ob_high, candles.index(next(c for c in candles if c["time"] == order_block.get("displacement_time")))))
        elif "نزولی" in order_block.get("type", "") and ob_low >= price - atr * 0.25:
            smart_candidates["short"].append(("order_block", ob_low, ob_high, candles.index(next(c for c in candles if c["time"] == order_block.get("displacement_time")))))
    if bullish_fvg_zone and bullish_fvg_zone["high"] <= price + atr * 0.25:
        smart_candidates["long"].append(("fvg", float(bullish_fvg_zone["low"]), float(bullish_fvg_zone["high"]), int(bullish_fvg_zone["formation_index"])))
    if bearish_fvg_zone and bearish_fvg_zone["low"] >= price - atr * 0.25:
        smart_candidates["short"].append(("fvg", float(bearish_fvg_zone["low"]), float(bearish_fvg_zone["high"]), int(bearish_fvg_zone["formation_index"])))
    ema_band = max(atr * 0.12, abs(price) * 0.0002)
    if ema21 < price:
        smart_candidates["long"].append(("ema21", max(0.0, ema21 - ema_band), ema21 + ema_band, len(candles) - 2))
    if ema21 > price:
        smart_candidates["short"].append(("ema21", max(0.0, ema21 - ema_band), ema21 + ema_band, len(candles) - 2))

    def make_pullback_plan(side):
        candidates = []
        for source, low, high, formation_index in smart_candidates[side]:
            if low <= 0 or high <= low or formation_index >= len(candles) - 1:
                continue
            distance = max(0.0, low - price, price - high)
            if distance <= atr * 3.0:
                candidates.append((distance, source, low, high, formation_index))
        if not candidates:
            return {
                "available": False, "status": "NO_ZONE",
                "note": "ناحیه پولبک معتبر و نزدیک بر اساس OB/FVG/EMA21 پیدا نشد؛ از ساختن سطح مصنوعی خودداری شد."
            }

        _, source, zone_low, zone_high, formation_index = min(candidates, key=lambda item: item[0])
        midpoint = (zone_low + zone_high) / 2.0
        post_formation = candles[formation_index + 1:]
        retest_candle = post_formation[-1] if post_formation else None
        touched = bool(retest_candle and retest_candle["low"] <= zone_high
                       and retest_candle["high"] >= zone_low)
        retest_time = retest_candle["time"] if touched else None

        # A valid rejection must show directional body strength and close near
        # the candle extreme; a simple green/red candle inside the zone is not enough.
        candle_range = max(1e-12, retest_candle["high"] - retest_candle["low"]) if retest_candle else 1.0
        body_ratio = (abs(retest_candle["close"] - retest_candle["open"]) / candle_range) if retest_candle else 0.0
        close_location = ((retest_candle["close"] - retest_candle["low"]) / candle_range) if retest_candle else 0.5
        prior_close = post_formation[-2]["close"] if len(post_formation) >= 2 else None
        reaction = bool(
            touched and body_ratio >= 0.45 and prior_close is not None and
            ((retest_candle["close"] > retest_candle["open"]
              and retest_candle["close"] >= midpoint
              and retest_candle["close"] > prior_close
              and close_location >= 0.65)
             if side == "long" else
             (retest_candle["close"] < retest_candle["open"]
              and retest_candle["close"] <= midpoint
              and retest_candle["close"] < prior_close
              and close_location <= 0.35))
        )
        confirmed = bool(touched and reaction and not risk_blocked)
        # Before confirmation use the zone midpoint only as a reference. After
        # confirmation use the actual closed rejection candle close as trigger.
        entry = float(retest_candle["close"]) if confirmed else midpoint
        buffer = max(atr * 0.20, abs(entry) * 0.0005)
        structural_stop = swing_low - buffer if side == "long" else swing_high + buffer
        stop_price = structural_stop
        unit_risk = entry - stop_price if side == "long" else stop_price - entry

        # A structural stop must sit beyond the invalidation swing and remain
        # proportionate. If not, reject the setup rather than inventing an ATR stop.
        if unit_risk <= 0 or unit_risk > atr * 3.0:
            return {
                "available": False, "status": "NO_VALID_STRUCTURAL_STOP",
                "source": source, "zone": {"low": round(zone_low, 10), "high": round(zone_high, 10)},
                "note": "حد ضرر ساختاری معتبر و متناسب پشت سوئینگ اخیر پیدا نشد؛ این ستاپ حذف شد."
            }

        # The first target must be a real opposing swing-liquidity level with
        # at least 1.5R room. No fixed-R target is substituted when structure
        # offers insufficient space.
        minimum_target = entry + unit_risk * 1.5 if side == "long" else entry - unit_risk * 1.5
        liquidity_levels = (
            [p["price"] for p in pivot_highs if p["price"] >= minimum_target]
            if side == "long" else
            [p["price"] for p in pivot_lows if p["price"] <= minimum_target]
        )
        if not liquidity_levels:
            return {
                "available": False, "status": "NO_STRUCTURAL_TARGET",
                "source": source, "zone": {"low": round(zone_low, 10), "high": round(zone_high, 10)},
                "stop_loss": round(stop_price, 10),
                "note": "نقدینگی/سوئینگ مخالف با حداقل نسبت سود به زیان ۱.۵ پیدا نشد؛ ورود پیشنهاد نمی‌شود."
            }
        first_target = min(liquidity_levels) if side == "long" else max(liquidity_levels)
        first_r = abs(first_target - entry) / unit_risk
        second_r = max(2.5, first_r + 0.5)
        third_r = max(3.5, second_r + 0.5)
        targets_smart = (
            [first_target, entry + unit_risk * second_r, entry + unit_risk * third_r]
            if side == "long" else
            [first_target, max(0.0, entry - unit_risk * second_r), max(0.0, entry - unit_risk * third_r)]
        )
        return {
            "available": True,
            "status": "CONFIRMED_ON_CLOSED_CANDLE" if confirmed else "WAIT_FOR_PULLBACK_CONFIRMATION",
            "setup_state": "TRIGGERED" if confirmed else ("WAITING_FOR_CONFIRMATION" if touched else "WAITING_FOR_RETEST"),
            "zone": {"low": round(zone_low, 10), "high": round(zone_high, 10)},
            "entry": round(entry, 10), "entry_reference": round(midpoint, 10),
            "stop_loss": round(stop_price, 10),
            "targets": [round(value, 10) for value in targets_smart],
            "first_target_source": "opposing_confirmed_swing_liquidity",
            "first_target_r": round(first_r, 2),
            "source": source, "formation_time": candles[formation_index]["time"],
            "retest_time": retest_time,
            "touched_by_post_formation_closed_candle": bool(touched),
            "reaction_confirmed": bool(reaction), "reaction_body_ratio": round(body_ratio, 3),
            "risk_gate_passed": not risk_blocked,
            "invalidation_level": round(stop_price, 10),
            "confirmation": "ورود فقط پس از لمس ناحیه و کندل رد معتبر بسته‌شده با بدنه قوی و بسته‌شدن نزدیک انتهای کندل؛ ورود تریگر برابر بسته‌شدن همان کندل است.",
            "note": "حد ضرر پشت سوئینگ ساختاری است؛ هدف اول نقدینگی مخالف واقعی با حداقل ۱.۵R است. سطح انتظار، سفارش فعال نیست."
        }

    smart_long = make_pullback_plan("long")
    smart_short = make_pullback_plan("short")
    # Prefer the final signal; if it is WAIT, fall back to the directional bias.
    # Risk-blocked means no trade is permitted, even when a directional bias exists.
    preferred_side = (
        "long" if direction == "BUY" else
        "short" if direction == "SELL" else
        "long" if bias == "صعودی" else
        "short" if bias == "نزولی" else "neutral"
    )
    primary_alignment = (
        "aligned" if (direction == "BUY" and preferred_side == "long") or
                     (direction == "SELL" and preferred_side == "short") else
        "wait_or_blocked" if direction == "WAIT" else "neutral"
    )
    smart_entry = {
        "preferred_side": preferred_side,
        "primary_signal": direction,
        "primary_bias": bias,
        "primary_alignment": primary_alignment,
        "long": smart_long, "short": smart_short,
        "risk_gate_passed": not risk_blocked,
        "risk_flags": risk_flags,
        "method": "nearest unmitigated OB/FVG or EMA21 pullback; closed-candle reaction required",
        "note": "جهت ترجیحی از سیگنال نهایی و سوگیری تحلیل گرفته شده است؛ فیلتر ریسک مسدود یعنی ورود مجاز نیست. ناحیه‌ها تخمینی‌اند و امتیازها احتمال آماری موفقیت نیستند."
    }
    return {
        "market": market, "symbol": symbol, "timeframe": timeframe,
        "signal": direction, "bias": bias,
        "setup_score": setup_score, "bull_score": bull, "bear_score": bear,
        "rsi": round(rsi_value, 2), "macd_momentum": "bullish" if macd_bull else "bearish" if macd_bear else "neutral",
        "bull_momentum_checks": bull_momentum_checks, "bear_momentum_checks": bear_momentum_checks,
        "score_note": "امتیاز تجربی هم‌جهتی شروط است و احتمال موفقیت معامله نیست.",
        "wait_reason": wait_reasons,
        "market_structure": structure_trend,
        "price": price, "atr": atr, "entry_price": entry_price, "trade_plan": trade_plan,
        "conditional_scenarios": conditional_scenarios, "scenario_analysis": scenario_analysis,
        "smart_entry": smart_entry,
        "chart_levels": {
            "support": range_low, "resistance": range_high,
            "long_trigger": bullish_trigger, "short_trigger": bearish_trigger,
            "entry": entry_price, "stop_loss": stop if direction in ("BUY", "SELL") else None,
            "targets": targets if direction in ("BUY", "SELL") else [],
            "fvg": bullish_fvg_zone or bearish_fvg_zone,
            "order_block": order_block
        },
        "price": price, "atr": atr, "stop_loss": stop,
        "take_profit_1": targets[0], "take_profit_2": targets[1], "take_profit_3": targets[2],
        "risk_per_unit": risk, "swing_high": swing_high, "swing_low": swing_low,
        "bos": "صعودی" if bos_up else "نزولی" if bos_down else "ندارد",
        "choch": choch, "liquidity_sweep": "پایین" if sweep_low else "بالا" if sweep_high else "ندارد",
        "liquidity_sweep_level": (sweep_low_event or sweep_high_event or {}).get("level"),
        "liquidity_sweep_time": (sweep_low_event or sweep_high_event or {}).get("time"),
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
            "risk_gate_passed": not risk_blocked,
            "risk_flags": risk_flags,
            "max_atr_percent_by_timeframe": max_atr_percent,
            "note": "فیلترهای نوسان/قدرت روند محافظه‌کارانه و تجربی‌اند؛ مقادیر سرمایه، اندازه قرارداد، اهرم، کارمزد و اسلیپیج باید جداگانه لحاظ شوند."
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
            "crypto_prices": "Binance Spot live ticker + closed OHLCV candles for all active USDT pairs",
            "crypto_derivatives": "Binance public futures funding/OI/long-short + spot order-book snapshot",
            "metals_prices": "Twelve Data spot when entitled; Yahoo spot fallback only. COMEX futures proxy is disabled unless ALLOW_METAL_FUTURES_PROXY=true.",
            "metal_instrument_note": "Spot and futures are never mixed by default. If no spot feed is available, the metal analysis returns unavailable instead of silently substituting a futures contract.",
            "crypto_dominance": "Market-cap dominance from CoinGecko" + ("; CoinMarketCap fallback key configured" if COINMARKETCAP_API_KEY else "; optional CoinMarketCap fallback key not set") + "; Binance Spot is used for exchange prices, not global dominance",
            "liquidation_heatmap": "غیرفعال؛ خوشه‌های تخمینی از امتیازدهی و رابط حذف شدند",
            "global_news": "GDELT DOC 2.0",
            "macro_indicators": "FRED " + ("configured" if FRED_API_KEY else "key_required"),
            "ai_synthesis": ("OpenAI configured" if AI_API_KEY else "Gemini free-tier " + ("configured" if GEMINI_API_KEY else "GEMINI_API_KEY required")),
            "economic_calendar": "XOOMAR public calendar API, keyless, cached 15 minutes",
            "external_crypto_confirmation": "MarginPad public screener, keyless, cached 5 minutes; validation only, not a signal source",
            "cot_positions": "CFTC public weekly disaggregated futures-only reports",
            "cross_asset_correlations": "BTC/ETH/gold/silver daily returns",
            "research_backtesting": "chronological holdout with fee/slippage stress",
            "paper_trading": "automatic simulated signal capture and live-quote monitoring; no exchange orders"
        },
        "live_trading": False,
        "supported_timeframes": list(TIMEFRAMES.keys())
    }


@app.get("/api/options")
def options():
    symbols = get_binance_spot_symbols()
    return {
        "crypto": symbols,
        "crypto_count": len(symbols),
        "crypto_source": "Binance Spot — active USDT pairs",
        "gold": [METALS["gold"]],
        "silver": [METALS["silver"]],
        "metal_source": "Twelve Data spot primary; Yahoo Finance spot/futures fallback (silver SI=F is COMEX futures proxy)",
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
    long_rows = values.get("long_short") or []
    taker_rows = values.get("taker_ratio") or []
    funding_rows = funding_rows if isinstance(funding_rows, list) else []
    long_rows = long_rows if isinstance(long_rows, list) else []
    taker_rows = taker_rows if isinstance(taker_rows, list) else []
    def safe_float(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    funding = safe_float(funding_rows[-1].get("fundingRate")) if funding_rows else None
    oi_data = values.get("open_interest") or {}
    oi_data = oi_data if isinstance(oi_data, dict) else {}
    depth = values.get("order_book") or {}
    depth = depth if isinstance(depth, dict) else {}
    bids, asks = depth.get("bids", []), depth.get("asks", [])
    bid_qty = sum(safe_float(row[1]) or 0.0 for row in bids if len(row) > 1) if bids else 0.0
    ask_qty = sum(safe_float(row[1]) or 0.0 for row in asks if len(row) > 1) if asks else 0.0
    total_qty = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total_qty if total_qty > 0 else None
    spread_bps = None
    if bids and asks:
        bid, ask = float(bids[0][0]), float(asks[0][0])
        mid = (bid + ask) / 2.0
        spread_bps = ((ask - bid) / mid * 10000.0) if mid else None

    metric_status = {
        "funding": funding is not None,
        "open_interest": safe_float(oi_data.get("openInterest")) is not None,
        "long_short": bool(long_rows),
        "taker_ratio": bool(taker_rows),
        "spot_order_book": bool(bids and asks),
    }
    result = {
        # A spot order-book response alone must not imply futures data is available.
        "available": any(metric_status[key] for key in ("funding", "open_interest", "long_short", "taker_ratio")),
        "metric_status": metric_status,
        "spot_order_book_available": metric_status["spot_order_book"],
        "source": "Binance public USD-M futures metrics + separate spot order-book snapshot",
        "funding_rate": funding,
        "funding_rate_percent": funding * 100 if funding is not None else None,
        "open_interest_contracts": safe_float(oi_data.get("openInterest")),
        "long_short_account_ratio": safe_float(long_rows[-1].get("longShortRatio")) if long_rows else None,
        "taker_buy_sell_ratio": safe_float(taker_rows[-1].get("buySellRatio")) if taker_rows else None,
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
    result["analysis_price"] = result.get("price")
    result["candle_source"] = (candles[-1].get("source") if candles else None) or ("Binance Spot" if market == "crypto" else "Twelve Data / Yahoo Finance metals fallback")
    result["live_quote"] = get_live_quote(market, symbol)
    quote = result["live_quote"]
    smart = result.get("smart_entry") or {}
    for side in ("long", "short"):
        plan = smart.get(side)
        if not isinstance(plan, dict) or not plan.get("available") or plan.get("status") == "BLOCKED_BY_ECONOMIC_CALENDAR":
            continue
        # Closed-candle reaction is necessary but not sufficient: require a fresh,
        # non-proxy quote still trading inside the proposed zone before confirmation.
        quote_fresh = bool(quote.get("available") and not quote.get("is_proxy")
                           and quote.get("age_seconds") is not None
                           and quote.get("age_seconds") <= (120 if market == "crypto" else 900))
        live_price = quote.get("price")
        zone = plan.get("zone") or {}
        live_value = float(live_price) if quote_fresh and live_price is not None else None
        stop_value = float(plan.get("stop_loss")) if plan.get("stop_loss") is not None else None
        invalidated = bool(live_value is not None and stop_value is not None and
                           ((side == "long" and live_value <= stop_value) or
                            (side == "short" and live_value >= stop_value)))
        in_zone = bool(live_value is not None
                       and float(zone.get("low", 0)) <= live_value <= float(zone.get("high", 0)))
        trigger_entry = float(plan.get("entry")) if plan.get("entry") is not None else None
        trigger_tolerance = max(float(result.get("atr") or 0) * 0.35, abs(trigger_entry or 0) * 0.0005)
        near_trigger = bool(live_value is not None and trigger_entry is not None
                             and abs(live_value - trigger_entry) <= trigger_tolerance)
        live_trigger_valid = bool(quote_fresh and (in_zone or (
            plan.get("reaction_confirmed") and near_trigger
        )))
        plan["live_quote_confirmed"] = live_trigger_valid
        plan["live_quote_source"] = quote.get("source") if quote_fresh else None
        plan["live_price_at_check"] = live_value
        if invalidated:
            plan["status"] = "INVALIDATED"
            plan["setup_state"] = "INVALIDATED"
            plan["live_quote_confirmed"] = False
            plan["note"] = "سناریو با عبور قیمت زنده از حد ابطال ساختاری باطل شد؛ ورود و اهداف قبلی دیگر معتبر نیستند."
        elif plan.get("status") == "CONFIRMED_ON_CLOSED_CANDLE" and not live_trigger_valid:
            plan["status"] = "WAIT_FOR_LIVE_PRICE_CONFIRMATION"
        elif plan.get("status") == "WAIT_FOR_PULLBACK_CONFIRMATION" and plan.get("reaction_confirmed") and live_trigger_valid and plan.get("risk_gate_passed"):
            plan["status"] = "CONFIRMED_ON_CLOSED_CANDLE"
        plan["note"] = (plan.get("note", "") + " قیمت زنده باید تازه و در محدوده مجاز باشد؛ در غیر این صورت ورود تأیید نمی‌شود.").strip()
    preferred_plan = smart.get(smart.get("preferred_side", ""))
    if isinstance(preferred_plan, dict) and preferred_plan.get("status") == "INVALIDATED":
        result["signal"] = "WAIT"
        result["entry_price"] = None
        result["stop_loss"] = None
        result["take_profit_1"] = result["take_profit_2"] = result["take_profit_3"] = None
        result["trade_plan"] = {"entry": None, "stop_loss": None, "targets": [],
                                "risk_reward": [], "entry_basis": "سناریوی ترجیحی با عبور قیمت زنده از حد ابطال باطل شده است."}
        result.setdefault("wait_reason", []).append("سناریوی ورود قبلی با شکست حد ابطال ساختاری باطل شد؛ تحلیل تازه لازم است.")
    result["smart_entry"] = smart
    if market == "crypto":
        # Skip the hard-to-read third-party heatmap fetch. Keep observable Binance
        # derivatives metrics as the source for positioning context.
        result["derivatives"] = get_crypto_derivatives(symbol)
    else:
        result["derivatives"] = {
            "available": False,
            "note": "داده مشتقات عمومی طلا/نقره در این نسخه متصل نیست."
        }

    # Execution-quality gates: stale/unavailable quote or an abnormally wide
    # observable spot spread blocks a fresh entry, without changing historical analysis.
    quote_age_limit = 60 if market == "crypto" else 900
    quote_is_fresh = bool(result["live_quote"].get("available")
                          and result["live_quote"].get("age_seconds") is not None
                          and result["live_quote"].get("age_seconds") <= quote_age_limit)
    execution_flags = []
    execution_blocked = False
    if not quote_is_fresh:
        execution_flags.append("BLOCK: قیمت زنده ناموجود یا قدیمی است؛ ورود جدید مجاز نیست")
        execution_blocked = True
    spread_bps = (result.get("derivatives") or {}).get("spot_spread_basis_points")
    if market == "crypto" and spread_bps is not None:
        max_spread = float(os.getenv("MAX_ENTRY_SPREAD_BPS", "20"))
        if spread_bps > max_spread:
            execution_flags.append("BLOCK: اسپرد دفتر سفارش از آستانه مجاز بیشتر است")
            execution_blocked = True
    if execution_blocked:
        result["signal"] = "WAIT"
        result["entry_price"] = None
        result["stop_loss"] = None
        result["take_profit_1"] = result["take_profit_2"] = result["take_profit_3"] = None
        result["trade_plan"] = {"entry": None, "stop_loss": None, "targets": [],
                                "risk_reward": [], "entry_basis": "کیفیت قیمت/اجرا برای ورود کافی نیست."}
        result.setdefault("wait_reason", []).extend(execution_flags)
        risk = result.setdefault("risk_management", {})
        risk["risk_gate_passed"] = False
        risk.setdefault("risk_flags", []).extend(execution_flags)
        smart = result.get("smart_entry") or {}
        smart["risk_gate_passed"] = False
        smart["risk_flags"] = list(smart.get("risk_flags") or []) + execution_flags
        for side in ("long", "short"):
            if isinstance(smart.get(side), dict) and smart[side].get("available"):
                smart[side]["status"] = "BLOCKED_BY_EXECUTION_QUALITY"
        result["smart_entry"] = smart
    result["execution_quality"] = {
        "quote_fresh": quote_is_fresh, "quote_age_limit_seconds": quote_age_limit,
        "spread_basis_points": spread_bps,
        "max_spread_basis_points": float(os.getenv("MAX_ENTRY_SPREAD_BPS", "20")),
        "blocked": execution_blocked, "flags": execution_flags
    }
    result["liquidation_heatmap"] = {"available": False, "disabled": True, "note": "نقشه شلوغ قبلی حذف شد."}
    result["liquidation_context"] = {"available": False, "note": "برای جلوگیری از امتیازدهی به خوشه‌های تخمینی نامطمئن، از امتیاز سیگنال حذف شد."}
    result["economic_event_risk"] = economic_event_risk(market)
    event_risk = result["economic_event_risk"]
    if event_risk.get("entry_blocked_by_calendar"):
        # A high-impact release within 15 minutes blocks new entries at the API layer.
        result["signal"] = "WAIT"
        result["entry_price"] = None
        result["stop_loss"] = None
        result["take_profit_1"] = result["take_profit_2"] = result["take_profit_3"] = None
        result["trade_plan"] = {"entry": None, "stop_loss": None, "targets": [],
                                "risk_reward": [], "entry_basis": "ورود به‌دلیل رویداد اقتصادی پرریسک مسدود شد."}
        result.setdefault("wait_reason", []).append("BLOCK: رویداد اقتصادی مهم در ۱۵ دقیقه آینده")
        risk = result.setdefault("risk_management", {})
        risk["risk_gate_passed"] = False
        risk.setdefault("risk_flags", []).append("BLOCK: تقویم اقتصادی ورود جدید را متوقف کرد")
        smart = result.get("smart_entry") or {}
        smart["risk_gate_passed"] = False
        smart["risk_flags"] = list(smart.get("risk_flags") or []) + ["BLOCK: تقویم اقتصادی ورود جدید را متوقف کرد"]
        for side in ("long", "short"):
            if isinstance(smart.get(side), dict) and smart[side].get("available"):
                smart[side]["status"] = "BLOCKED_BY_ECONOMIC_CALENDAR"
        result["smart_entry"] = smart
    if market == "crypto":
        result["external_confirmation"] = get_marginpad_screener_confirmation(symbol)
    else:
        result["external_confirmation"] = {"available": False, "source": "MarginPad", "note": "اسکنر مشتقات MarginPad فقط برای کریپتو است."}
    chart_bars = completed_candles(candles, timeframe)[-220:]
    result["chart_candles"] = [{
        "time": int(c["time"] // 1000),
        "open": c["open"], "high": c["high"],
        "low": c["low"], "close": c["close"],
        "volume": c.get("volume", 0) or 0
    } for c in chart_bars]
    result["chart_source"] = result.get("candle_source")
    result["chart_source_symbol"] = (chart_bars[-1].get("source_symbol") if chart_bars else None) or (symbol if market == "crypto" else None)
    rsi_values = (result.get("advanced_analysis") or {}).get("rsi_series") or []
    rsi_values = rsi_values[-len(chart_bars):] if chart_bars else []
    result["rsi_chart"] = [
        {"time": int(candle["time"] // 1000), "value": value}
        for candle, value in zip(chart_bars, rsi_values) if value is not None
    ]
    marker_time = int(chart_bars[-1]["time"] // 1000) if chart_bars else None
    result["chart_markers"] = []
    if marker_time and result.get("signal") == "BUY":
        result["chart_markers"].append({
            "time": marker_time, "position": "belowBar", "color": "#28c997",
            "shape": "arrowUp", "text": "BUY"
        })
    elif marker_time and result.get("signal") == "SELL":
        result["chart_markers"].append({
            "time": marker_time, "position": "aboveBar", "color": "#ff7182",
            "shape": "arrowDown", "text": "SELL"
        })
    if result.get("liquidity_sweep_time"):
        result["chart_markers"].append({
            "time": int(result["liquidity_sweep_time"] // 1000),
            "position": "aboveBar" if result.get("liquidity_sweep") == "بالا" else "belowBar",
            "color": "#e9b95e", "shape": "circle", "text": "Liquidity sweep"
        })
    result["paper_trade"] = record_signal(result)
    return result


@app.get("/api/paper-trading")
def api_paper_trading():
    return list_paper_trades(150)


@app.post("/api/paper-trading/monitor")
def api_paper_trading_monitor():
    check = monitor_open_trades(get_live_quote)
    return {"monitor": check, **list_paper_trades(150)}


@app.get("/api/quote")
def api_quote(market: str = "crypto", symbol: str = "BTCUSDT"):
    if market == "crypto" and symbol not in get_binance_spot_symbols():
        raise HTTPException(400, "نماد باید یک جفت فعال USDT در Binance Spot باشد")
    if market in METALS and symbol != METALS[market]:
        raise HTTPException(400, "نماد فلز نامعتبر است")
    if market not in ("crypto", "gold", "silver"):
        raise HTTPException(400, "بازار نامعتبر است")
    return {"market": market, "symbol": symbol, "quote": get_live_quote(market, symbol), "checked_at": utc_now()}


@app.get("/api/backtest")
def api_backtest(market: str = "crypto", symbol: str = "BTCUSDT", timeframe: str = "1h"):
    candles = completed_candles(get_candles(market, symbol, timeframe), timeframe)
    # Full production analysis is computationally heavier than the baseline.
    # Bound replay size to keep the public endpoint responsive; users can raise
    # this limit via BACKTEST_CANDLE_LIMIT after measuring their instance capacity.
    replay_limit = max(120, min(int(os.getenv("BACKTEST_CANDLE_LIMIT", "500")), 1000))
    candles = candles[-replay_limit:]
    engine = lambda history: analyze(history, market, symbol, timeframe)
    result = run_research_backtest(candles, timeframe, signal_engine=engine)
    result.update({
        "market": market, "symbol": symbol, "updated_at": utc_now(),
        "production_engine_replay": True,
        "history_limit": replay_limit,
        "history_note": "بازپخش موتور تحلیل تولیدی روی آخرین کندل‌های بسته‌شده؛ فیلترهای وابسته به قیمت زنده و تقویم در داده تاریخی قابل بازسازی کامل نیستند."
    })
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
        "live_quote": get_live_quote(market, symbol),
        "timeframes": results, "updated_at": utc_now(),
        "note": "قیمت زنده جدا از قیمت کندل بسته‌شده است؛ تایم‌فریم‌های بالاتر وزن بیشتری دارند و امتیاز وزنی احتمال موفقیت نیست."
    }


CORRELATION_CACHE = {"at": 0, "data": None}


def pearson_correlation(xs, ys):
    if len(xs) != len(ys) or len(xs) < 10:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = (sum(x * x for x in dx) * sum(y * y for y in dy)) ** 0.5
    return sum(x * y for x, y in zip(dx, dy)) / denom if denom > 0 else None


@app.get("/api/correlations")
def correlations():
    now = time.time()
    if CORRELATION_CACHE["data"] and now - CORRELATION_CACHE["at"] < 900:
        return CORRELATION_CACHE["data"]

    instruments = {
        "BTC": ("crypto", "BTCUSDT"),
        "ETH": ("crypto", "ETHUSDT"),
        "GOLD": ("gold", METALS["gold"]),
        "SILVER": ("silver", METALS["silver"])
    }
    closes_by_asset, errors = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {
            pool.submit(get_candles, market, symbol, "1d"): key
            for key, (market, symbol) in instruments.items()
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                candles = completed_candles(future.result(), "1d")
                closes = {}
                for candle in candles:
                    day = datetime.fromtimestamp(candle["time"] / 1000, tz=timezone.utc).date().isoformat()
                    closes[day] = candle["close"]
                closes_by_asset[key] = closes
            except Exception as exc:
                errors.append(key + ": " + str(exc)[:120])

    returns_by_asset = {}
    for key, closes in closes_by_asset.items():
        dates = sorted(closes)
        returns_by_asset[key] = {
            dates[i]: math.log(closes[dates[i]] / closes[dates[i - 1]])
            for i in range(1, len(dates))
            if closes[dates[i]] > 0 and closes[dates[i - 1]] > 0
        }

    pairs = (("BTC", "ETH"), ("BTC", "GOLD"), ("BTC", "SILVER"),
             ("ETH", "GOLD"), ("ETH", "SILVER"), ("GOLD", "SILVER"))
    results = []
    for left, right in pairs:
        common = sorted(set(returns_by_asset.get(left, {})) & set(returns_by_asset.get(right, {})))
        pair_data = {"pair": left + "/" + right, "observations": len(common)}
        for window in (30, 90):
            dates = common[-window:]
            value = pearson_correlation(
                [returns_by_asset[left][d] for d in dates],
                [returns_by_asset[right][d] for d in dates]
            )
            pair_data["correlation_" + str(window) + "d"] = round(value, 3) if value is not None else None
        results.append(pair_data)

    result = {
        "available": any(item.get("correlation_30d") is not None or item.get("correlation_90d") is not None for item in results),
        "source": "Binance daily candles + Yahoo Finance/Twelve Data metals daily candles",
        "pairs": results,
        "errors": errors,
        "updated_at": utc_now(),
        "note": "همبستگی از بازده لگاریتمی روزانه و تاریخ‌های مشترک محاسبه شده؛ رابطه ثابت یا علیت نیست و در رژیم‌های مختلف تغییر می‌کند."
    }
    CORRELATION_CACHE.update({"at": now, "data": result})
    return result


def _as_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _dominance_timestamp(value):
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)) or str(value).isdigit():
            stamp = int(float(value))
            return datetime.fromtimestamp(stamp, tz=timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _build_dominance_payload(source, raw):
    if source == "CoinGecko":
        global_data = raw.get("data", {})
        shares = global_data.get("market_cap_percentage", {})
        btc = _as_float(shares.get("btc"))
        eth = _as_float(shares.get("eth"))
        usdt = _as_float(shares.get("usdt"))
        updated = _dominance_timestamp(global_data.get("updated_at"))
        market_cap = (global_data.get("total_market_cap") or {}).get("usd")
        volume = (global_data.get("total_volume") or {}).get("usd")
    else:
        global_data = raw.get("data", {})
        quote = (global_data.get("quote") or {}).get("USD", {})
        btc = _as_float(global_data.get("btc_dominance"))
        eth = _as_float(global_data.get("eth_dominance"))
        usdt = _as_float(global_data.get("usdt_dominance"))
        updated = _dominance_timestamp(global_data.get("last_updated") or quote.get("last_updated"))
        market_cap = quote.get("total_market_cap")
        volume = quote.get("total_volume_24h")

    if btc is None or eth is None or not (0 <= btc <= 100 and 0 <= eth <= 100):
        raise RuntimeError(source + " returned incomplete or invalid market-cap dominance values")
    others = max(0.0, 100.0 - btc - eth - usdt) if usdt is not None else None
    return {
        "available": True,
        "btc": round(btc, 4),
        "eth": round(eth, 4),
        "usdt": round(usdt, 4) if usdt is not None else None,
        "others_ex_btc_eth_usdt": round(others, 4) if others is not None else None,
        "total_market_cap_usd": market_cap,
        "total_volume_24h_usd": volume,
        "updated_at": utc_now(),
        "source_updated_at": updated.isoformat() if updated else None,
        "source_age_seconds": max(0, int(time.time() - updated.timestamp())) if updated else None,
        "source": source + " market-cap data",
        "stale": False,
        "note": (
            "دامیننس بر اساس سهم ارزش بازار و عرضه در گردش است، نه حجم معاملات Binance. "
            "Binance Spot برای قیمت و حجم همان صرافی استفاده می‌شود و API عمومی آن دامیننس کل بازار را منتشر نمی‌کند. "
            "روش پوشش بازار و عرضه در گردش CoinGecko/CoinMarketCap ممکن است با TradingView CRYPTOCAP تفاوت داشته باشد. "
            "Others در این صفحه باقیمانده پس از BTC، ETH و USDT است و لزوماً با OTHERS.D برابر نیست."
        )
    }


@app.get("/api/dominance")
def dominance():
    now = time.time()
    with DOMINANCE_LOCK:
        cached = DOMINANCE_CACHE.get("data")
        if cached and now - DOMINANCE_CACHE.get("at", 0) < 90:
            result = dict(cached)
            result["cache_hit"] = True
            return result

        errors = []
        headers = {"x-cg-demo-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
        try:
            raw = fetch_json("https://api.coingecko.com/api/v3/global", headers=headers, timeout=12)
            result = _build_dominance_payload("CoinGecko", raw)
            DOMINANCE_CACHE.update({"at": now, "data": result})
            result["cache_hit"] = False
            return result
        except Exception as exc:
            errors.append("CoinGecko: " + str(exc)[:140])

        if COINMARKETCAP_API_KEY:
            try:
                raw = fetch_json(
                    "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest?convert=USD",
                    headers={"X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY},
                    timeout=12
                )
                result = _build_dominance_payload("CoinMarketCap", raw)
                result["fallback_used"] = True
                DOMINANCE_CACHE.update({"at": now, "data": result})
                result["cache_hit"] = False
                return result
            except Exception as exc:
                errors.append("CoinMarketCap: " + str(exc)[:140])

        if cached and now - DOMINANCE_CACHE.get("at", 0) < 1800:
            result = dict(cached)
            result["stale"] = True
            result["cache_age_seconds"] = max(0, int(now - DOMINANCE_CACHE.get("at", now)))
            result["warning"] = "منبع دامیننس موقتاً پاسخ نداد؛ آخرین داده معتبر به‌صورت موقت نمایش داده می‌شود."
            result["errors"] = errors
            return result

    raise HTTPException(502, "داده دامیننس از منابع ارزش بازار دریافت نشد. " + " | ".join(errors))


def _normalize_liquidation_side(value):
    side = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "short" in side:
        return "short"
    if "long" in side:
        return "long"
    return "unknown"


def _marginpad_data_object(raw):
    if not isinstance(raw, dict):
        return {}
    nested = raw.get("data")
    return nested if isinstance(nested, dict) else raw


def _marginpad_timestamp(value):
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)) or str(value).strip().replace(".", "", 1).isdigit():
            stamp = float(value)
            if stamp > 1e12:
                stamp /= 1000.0
            return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat()
        text = str(value).strip()
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def normalize_marginpad_liquidation_data(cluster_raw, event_raw, symbol):
    """Keep modelled liquidation levels separate from observed forced-close events."""
    cluster_data = _marginpad_data_object(cluster_raw)
    event_data = _marginpad_data_object(event_raw)
    raw_clusters = cluster_data.get("clusters") or []
    raw_events = event_data.get("events") or []
    parsed_clusters = []

    for item in raw_clusters[:5000] if isinstance(raw_clusters, list) else []:
        if not isinstance(item, dict):
            continue
        price = _as_float(item.get("price"))
        if price is None or price <= 0:
            continue
        est_notional = _as_float(item.get("est_notional"))
        side = _normalize_liquidation_side(item.get("side"))
        parsed_clusters.append({
            "price": price,
            "side": side,
            "est_notional": max(0.0, est_notional or 0.0),
            "intensity": max(0.0, est_notional or 0.0),
            "relative_intensity": 0.0,
            "kind": "model_estimate"
        })

    max_notional = max((item["est_notional"] for item in parsed_clusters), default=0.0)
    for item in parsed_clusters:
        item["relative_intensity"] = round(item["est_notional"] / max_notional, 4) if max_notional else 0.0
    parsed_clusters.sort(key=lambda item: item["est_notional"], reverse=True)

    parsed_events = []
    for item in raw_events[:400] if isinstance(raw_events, list) else []:
        if not isinstance(item, dict):
            continue
        price = _as_float(item.get("price"))
        if price is None or price <= 0:
            continue
        qty = _as_float(item.get("qty"))
        notional = _as_float(item.get("notional"))
        if notional is None and qty is not None:
            notional = abs(qty * price)
        raw_side = str(item.get("side") or "").strip().lower()
        side_class = _normalize_liquidation_side(raw_side)
        parsed_events.append({
            "ts": item.get("ts"),
            "timestamp": _marginpad_timestamp(item.get("ts")),
            "exchange": str(item.get("exchange") or "نامشخص")[:40],
            "symbol": str(item.get("symbol") or symbol)[:30],
            "side": raw_side or "unknown",
            "side_class": side_class,
            "price": price,
            "qty": qty,
            "notional": max(0.0, notional or 0.0),
            "kind": "observed_event"
        })

    long_events = [item for item in parsed_events if item["side_class"] == "long"]
    short_events = [item for item in parsed_events if item["side_class"] == "short"]
    long_notional = sum(item["notional"] for item in long_events)
    short_notional = sum(item["notional"] for item in short_events)
    updated_at = _marginpad_timestamp(
        cluster_data.get("updatedAt") or cluster_data.get("updated_at")
        or event_data.get("updatedAt") or event_data.get("updated_at")
    )
    if not updated_at and parsed_events:
        updated_at = parsed_events[-1].get("timestamp")
    return {
        "available": bool(parsed_clusters or parsed_events),
        "source": "MarginPad liquidation collector",
        "symbol": str(symbol).upper().replace("USDT", "").replace("USD", ""),
        "range": "Live",
        "clusters": parsed_clusters[:100],
        "model_clusters": parsed_clusters[:100],
        "events": parsed_events,
        "event_count": len(parsed_events),
        "event_summary": {
            "long_liquidated_count": len(long_events),
            "short_liquidated_count": len(short_events),
            "long_liquidated_notional": round(long_notional, 2),
            "short_liquidated_notional": round(short_notional, 2)
        },
        "updated_at": updated_at or utc_now(),
        "note": (
            "خوشه‌های قیمتی، برآورد مدل MarginPad هستند و قیمت دقیق لیکوییدیشن افراد نیستند. "
            "نقاط رویدادها، لیکوییدیشن‌های واقعیِ گزارش‌شده از فید صرافی‌ها هستند؛ پوشش همه بازار تضمین‌شده نیست."
        )
    }


def get_marginpad_liquidation_heatmap(symbol):
    base_symbol = str(symbol).upper().replace("USDT", "").replace("USD", "")
    if not base_symbol or not base_symbol.isalnum():
        return {"available": False, "source": "MarginPad", "note": "نماد برای MarginPad معتبر نیست."}

    now = time.time()
    cache_key = base_symbol
    with MARGINPAD_LOCK:
        cached = MARGINPAD_CACHE.get(cache_key)
        if cached and now - cached["at"] < 45:
            result = dict(cached["data"])
            result["cache_age_seconds"] = max(0, int(now - cached["at"]))
            return result

        params = urllib.parse.urlencode({"symbol": base_symbol})
        clusters_url = "https://marginpad.io/api/v1/clusters?" + params
        events_url = "https://marginpad.io/api/v1/liquidations/live?" + urllib.parse.urlencode({
            "symbol": base_symbol, "limit": 400
        })
        cluster_raw, event_raw = {}, {}
        errors = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            cluster_future = pool.submit(fetch_json, clusters_url, None, 9)
            event_future = pool.submit(fetch_json, events_url, None, 9)
            try:
                cluster_raw = cluster_future.result()
            except Exception as exc:
                errors.append("clusters: " + str(exc)[:120])
            try:
                event_raw = event_future.result()
            except Exception as exc:
                errors.append("events: " + str(exc)[:120])

        result = normalize_marginpad_liquidation_data(cluster_raw, event_raw, base_symbol)
        result["errors"] = errors
        result["stale"] = False
        if errors:
            result["note"] += " یکی از دو endpoint موقتاً پاسخ نداد؛ داده موجود ممکن است ناقص باشد."
        if result["available"]:
            MARGINPAD_CACHE[cache_key] = {"at": now, "data": result}
            return result

        if cached and now - cached["at"] < 900:
            stale_result = dict(cached["data"])
            stale_result["stale"] = True
            stale_result["cache_age_seconds"] = max(0, int(now - cached["at"]))
            stale_result["warning"] = "MarginPad موقتاً پاسخ نداد؛ آخرین داده معتبر به‌صورت قدیمی نمایش داده می‌شود."
            stale_result["errors"] = errors
            return stale_result

        result["note"] += " داده‌ای از MarginPad دریافت نشد."
        return result


def build_liquidation_context(result, heatmap, current_price):
    if not heatmap or not heatmap.get("available") or current_price is None:
        return {"available": False, "note": (heatmap or {}).get("note", "نقشه لیکوییدیشن در دسترس نیست.")}
    clusters = heatmap.get("model_clusters") or heatmap.get("clusters") or []
    above = [item for item in clusters if item.get("price") is not None and item["price"] > current_price]
    below = [item for item in clusters if item.get("price") is not None and item["price"] < current_price]
    nearest_above = min(above, key=lambda item: item["price"] - current_price) if above else None
    nearest_below = min(below, key=lambda item: current_price - item["price"]) if below else None
    atr = result.get("atr") or 0
    adjustment = 0
    reason = "خوشه‌ها برآورد مدل‌اند؛ رویدادهای واقعی جداگانه نمایش داده می‌شوند و این داده به‌تنهایی جهت معامله را تعیین نمی‌کند."
    if atr > 0 and result.get("signal") in ("BUY", "SELL"):
        if result["signal"] == "BUY" and nearest_above:
            distance = nearest_above["price"] - current_price
            opposing = nearest_below.get("est_notional", nearest_below.get("intensity", 0)) if nearest_below else 0
            strength = nearest_above.get("est_notional", nearest_above.get("intensity", 0))
            if distance <= atr * 3 and nearest_above.get("side") == "short" and strength >= opposing:
                adjustment = 1
                reason = "خوشه تخمینی لیکوییدیشن شورت بالای قیمت با پلن خرید هم‌جهت است؛ فقط ۱ امتیاز تأییدی کم‌وزن اضافه شد."
        elif result["signal"] == "SELL" and nearest_below:
            distance = current_price - nearest_below["price"]
            opposing = nearest_above.get("est_notional", nearest_above.get("intensity", 0)) if nearest_above else 0
            strength = nearest_below.get("est_notional", nearest_below.get("intensity", 0))
            if distance <= atr * 3 and nearest_below.get("side") == "long" and strength >= opposing:
                adjustment = 1
                reason = "خوشه تخمینی لیکوییدیشن لانگ زیر قیمت با پلن فروش هم‌جهت است؛ فقط ۱ امتیاز تأییدی کم‌وزن اضافه شد."

    if adjustment:
        if result.get("signal") == "BUY":
            result["bull_score"] = min(100, int(result.get("bull_score", 0)) + adjustment)
        elif result.get("signal") == "SELL":
            result["bear_score"] = min(100, int(result.get("bear_score", 0)) + adjustment)
        result["setup_score"] = max(int(result.get("bull_score", 0)), int(result.get("bear_score", 0)))
        bull_score = int(result.get("bull_score", 0))
        bear_score = int(result.get("bear_score", 0))
        result["bias"] = "صعودی" if bull_score - bear_score >= 6 else "نزولی" if bear_score - bull_score >= 6 else "خنثی"
        if isinstance(result.get("analysis"), list):
            result["analysis"].append(reason)

    return {
        "available": True,
        "current_price": current_price,
        "nearest_above": nearest_above,
        "nearest_below": nearest_below,
        "score_adjustment": adjustment,
        "event_count": heatmap.get("event_count", 0),
        "event_summary": heatmap.get("event_summary") or {},
        "note": reason
    }


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
            "not_connected": ["ICE DXY exact index", "Gold ETF flows", "Silver ETF flows", "central-bank gold purchases"]
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
        "available": any(item.get("available") for item in indicators), "source": "Federal Reserve Economic Data (FRED)",
        "indicators": indicators, "errors": errors,
        "not_connected": ["ICE DXY exact index", "Gold ETF flows", "Silver ETF flows", "central-bank gold purchases"],
        "updated_at": utc_now(),
        "note": "برخی شاخص‌ها با تأخیر منتشر می‌شوند؛ تاریخ هر مشاهده کنار مقدار آمده است."
    }
    MACRO_CACHE.update({"at": now, "data": result})
    return result


COT_CACHE = {}


def cot_positions(asset):
    if asset not in ("gold", "silver"):
        return {"available": False, "note": "COT برای این دارایی در این اتصال پشتیبانی نمی‌شود."}
    cached = COT_CACHE.get(asset)
    cache_ttl = 21600 if cached and cached["data"].get("available") else 900
    if cached and time.time() - cached["at"] < cache_ttl:
        return cached["data"]

    commodity = "GOLD" if asset == "gold" else "SILVER"
    params = urllib.parse.urlencode({
        "$where": "commodity_name = '" + commodity + "'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": 60
    })
    try:
        rows = fetch_json("https://publicreporting.cftc.gov/resource/72hh-3qpy.json?" + params, timeout=20)
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("CFTC returned no matching rows")

        grouped = {}
        for row in rows:
            key = row.get("contract_market_name") or row.get("market_and_exchange_names") or commodity
            grouped.setdefault(key, []).append(row)
        contract, selected = max(grouped.items(), key=lambda item: len(item[1]))
        selected.sort(key=lambda row: row.get("report_date_as_yyyy_mm_dd", ""), reverse=True)

        parsed = []
        for row in selected:
            try:
                long_pos = float(row.get("m_money_positions_long_all") or 0)
                short_pos = float(row.get("m_money_positions_short_all") or 0)
                open_interest = float(row.get("open_interest_all") or 0)
                parsed.append({
                    "date": row.get("report_date_as_yyyy_mm_dd"),
                    "managed_money_long": long_pos,
                    "managed_money_short": short_pos,
                    "managed_money_net": long_pos - short_pos,
                    "open_interest": open_interest,
                    "net_percent_open_interest": ((long_pos - short_pos) / open_interest * 100) if open_interest else None
                })
            except (TypeError, ValueError):
                continue
        if not parsed:
            raise RuntimeError("CFTC rows did not contain usable positioning fields")

        current = parsed[0]
        previous = parsed[1] if len(parsed) > 1 else None
        nets = [item["managed_money_net"] for item in parsed]
        low_net, high_net = min(nets), max(nets)
        cot_index = ((current["managed_money_net"] - low_net) / (high_net - low_net) * 100) if high_net > low_net else 50.0
        result = {
            "available": True,
            "source": "CFTC Disaggregated Futures Only",
            "contract": contract,
            "report_date": current["date"],
            "managed_money_long": current["managed_money_long"],
            "managed_money_short": current["managed_money_short"],
            "managed_money_net": current["managed_money_net"],
            "net_percent_open_interest": current["net_percent_open_interest"],
            "weekly_net_change": current["managed_money_net"] - previous["managed_money_net"] if previous else None,
            "cot_index_lookback": round(cot_index, 1),
            "lookback_reports": len(parsed),
            "note": "گزارش هفتگی است؛ موقعیت‌ها مربوط به سه‌شنبه و معمولاً جمعه منتشر می‌شوند. شاخص COT فقط نسبت به نمونه دریافت‌شده محاسبه شده و سیگنال کوتاه‌مدت نیست."
        }
    except Exception as exc:
        result = {
            "available": False, "source": "CFTC Disaggregated Futures Only",
            "error": str(exc)[:180],
            "note": "داده COT فعلاً دریافت نشد؛ این بخش در امتیاز تکنیکال دخالت داده نمی‌شود."
        }
    COT_CACHE[asset] = {"at": time.time(), "data": result}
    return result


@app.get("/api/cot")
def api_cot(asset: str = "gold"):
    if asset not in ("gold", "silver"):
        raise HTTPException(400, "COT فقط برای طلا و نقره در دسترس است")
    return cot_positions(asset)



@app.get("/api/economic-calendar")
def economic_calendar():
    """Free US macro calendar sourced from agency schedules via XOOMAR."""
    now = time.time()
    cached = ECONOMIC_CALENDAR_CACHE.get("data")
    if cached and now - ECONOMIC_CALENDAR_CACHE.get("at", 0) < 900:
        result = dict(cached)
        result["cache_age_seconds"] = int(now - ECONOMIC_CALENDAR_CACHE["at"])
        return result
    try:
        data = fetch_json("https://xoomar.com/api/markets/calendar?importance=high", timeout=12)
        rows = data.get("data", []) if isinstance(data, dict) else []
        if not isinstance(rows, list):
            raise RuntimeError("Calendar response has no data array")
        now_dt = datetime.now(timezone.utc)
        events = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("scheduledAt"):
                continue
            try:
                stamp = datetime.fromisoformat(str(row["scheduledAt"]).replace("Z", "+00:00"))
            except ValueError:
                continue
            minutes = int((stamp - now_dt).total_seconds() / 60)
            events.append({**row, "minutes_until": minutes, "is_upcoming": minutes >= 0})
        events.sort(key=lambda x: x.get("scheduledAt", ""))
        result = {"available": True, "source": "XOOMAR / official US agency schedules",
            "events": events, "next_release_at": data.get("nextReleaseAt"),
            "updated_at": data.get("updatedAt") or utc_now(),
            "attribution": data.get("attribution") or "Data: XOOMAR",
            "source_attribution": (data.get("meta") or {}).get("sourceAttribution"),
            "note": "تقویم زمان‌بندی انتشار رسمی است؛ پیش‌بینی اجماع بازار ندارد. زمان‌ها UTC هستند."}
        ECONOMIC_CALENDAR_CACHE.update({"at": now, "data": result})
        return result
    except Exception as exc:
        if cached:
            stale = dict(cached)
            stale.update({"stale": True, "warning": str(exc)[:160]})
            return stale
        return {"available": False, "source": "XOOMAR", "events": [], "warning": str(exc)[:180]}


def economic_event_risk(market):
    calendar = economic_calendar()
    if not calendar.get("available"):
        return {"available": False, "note": "تقویم اقتصادی در دسترس نیست؛ فیلتر خبر اعمال نشد."}
    now = datetime.now(timezone.utc)
    upcoming = []
    for event in calendar.get("events", []):
        try:
            stamp = datetime.fromisoformat(str(event.get("scheduledAt", "")).replace("Z", "+00:00"))
            minutes = int((stamp - now).total_seconds() / 60)
        except ValueError:
            continue
        if 0 <= minutes <= 24 * 60:
            upcoming.append({"name": event.get("eventName"), "scheduled_at": event.get("scheduledAt"), "minutes_until": minutes, "importance": event.get("importance"), "source": event.get("source")})
    next_event = min(upcoming, key=lambda x: x["minutes_until"]) if upcoming else None
    blocked = bool(next_event and next_event["minutes_until"] <= 15)
    return {"available": True, "source": calendar.get("source"), "next_high_event": next_event,
        "high_events_next_24h": len(upcoming), "new_entry_caution": bool(next_event and next_event["minutes_until"] <= 60),
        "entry_blocked_by_calendar": blocked,
        "note": ("خبر مهم در ۱۵ دقیقه آینده؛ از ورود جدید تا عبور خبر و آرام‌شدن نوسان خودداری کن." if blocked else "خبر مهم در یک ساعت آینده؛ ریسک خبر را در اندازه موقعیت لحاظ کن." if next_event and next_event["minutes_until"] <= 60 else "در ۲۴ ساعت آینده خبر مهمی در تقویم این منبع پیدا نشد؛ این به معنی نبود همه رویدادهای بازار نیست.")}


def get_marginpad_screener_confirmation(symbol):
    """Independent, low-weight crypto screener cross-check; never changes our core score."""
    now = time.time()
    cached = MARGINPAD_SCREENER_CACHE.get("data")
    if cached and now - MARGINPAD_SCREENER_CACHE.get("at", 0) < 300:
        data = cached
    else:
        try:
            raw = fetch_json("https://marginpad.io/api/v1/screener", timeout=12)
            envelope = raw.get("data", raw) if isinstance(raw, dict) else {}
            rows = envelope.get("rows", []) if isinstance(envelope, dict) else []
            data = rows if isinstance(rows, list) else []
            MARGINPAD_SCREENER_CACHE.update({"at": now, "data": data})
        except Exception as exc:
            return {"available": False, "source": "MarginPad screener", "note": "تأیید مستقل دریافت نشد: " + str(exc)[:120]}
    base = str(symbol).upper().replace("USDT", "").replace("USD", "")
    match = next((row for row in data if str(row.get("symbol", "")).upper().replace("USDT", "").replace("USD", "") == base), None)
    if not match:
        return {"available": False, "source": "MarginPad screener", "note": "این نماد در فهرست منتخب اسکنر MarginPad نبود؛ عدم حضور به معنی سیگنال منفی نیست."}
    return {"available": True, "source": "MarginPad independent screener",
        "score": match.get("score"), "verdict": match.get("verdict"),
        "trend": match.get("trend"), "rsi": match.get("rsi"), "setup": match.get("setup"),
        "note": "این خروجی مدل مستقل ارائه‌دهنده است؛ فقط تأیید جانبی است و در امتیاز اصلی یا ورود/خروج دخالت داده نمی‌شود."}

@app.get("/api/market-context")
def market_context(asset: str = "gold"):
    if asset not in ("gold", "silver", "crypto"):
        raise HTTPException(400, "asset نامعتبر است")
    macro_data = macro()
    news_data = news(asset)
    cot_data = cot_positions(asset) if asset in ("gold", "silver") else {"available": False, "note": "COT برای کریپتو در این بخش ارائه نمی‌شود."}
    factors = [
        {"factor": "دلار آمریکا", "status": "شاخص گسترده دلار FRED موجود است؛ جایگزین دقیق ICE DXY نیست" if any(x.get("series") == "DTWEXBGS" and x.get("available") for x in macro_data.get("indicators", [])) else "نیازمند منبع داده دلار", "impact": "دلار قوی می‌تواند بر فلزات و دارایی‌های ریسکی فشار وارد کند؛ رابطه قطعی نیست"},
        {"factor": "بازده واقعی اوراق", "status": "داده FRED موجود است" if any(x.get("series") == "DFII10" and x.get("available") for x in macro_data.get("indicators", [])) else "نیازمند FRED_API_KEY", "impact": "افزایش بازده واقعی معمولاً هزینه فرصت نگهداری طلا را بالا می‌برد"},
        {"factor": "نرخ بهره و تورم آمریکا", "status": "داده FRED موجود است" if macro_data.get("available") else "نیازمند FRED_API_KEY", "impact": "انتظارات سیاست پولی می‌تواند بر دلار و فلزات اثر بگذارد"},
        {"factor": "جنگ و تحریم‌ها", "status": "رصد تیترهای خبری فعال", "impact": "ممکن است تقاضای دارایی امن را تغییر دهد"},
    ]
    if asset in ("gold", "silver"):
        factors.append({"factor": "COT و سایر جریان‌های سرمایه", "status": "داده COT هفتگی در دسترس است؛ ETF و خرید بانک‌های مرکزی هنوز متصل نیستند" if cot_data.get("available") else "داده COT فعلاً در دسترس نیست؛ ETF و خرید بانک‌های مرکزی متصل نیستند", "impact": "موقعیت‌گیری مدیران پولی دید میان‌مدت می‌دهد اما داده هفتگی و با تأخیر است"})
    if asset == "silver":
        factors.append({"factor": "تقاضای صنعتی و معادن نقره", "status": "رصد تیترهای خبری فعال", "impact": "تقاضای صنعتی و اختلال عرضه می‌تواند بر توازن بازار اثر بگذارد"})
    return {"asset": asset, "factors": factors, "macro": macro_data, "news": news_data, "cot": cot_data, "updated_at": utc_now()}


def get_scanner_universe(limit):
    """Rank the eligible spot universe by 24h quote volume before expensive analysis."""
    symbols = get_binance_spot_symbols()
    try:
        tickers = fetch_json("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=15)
        eligible = set(symbols)
        ranked = []
        for row in tickers if isinstance(tickers, list) else []:
            symbol = row.get("symbol")
            if symbol not in eligible:
                continue
            base = symbol[:-4] if symbol.endswith("USDT") else symbol
            # Exclude leveraged-token naming patterns from the candidate list.
            if any(base.endswith(suffix) for suffix in ("UP", "DOWN", "BULL", "BEAR")):
                continue
            quote_volume = _as_float(row.get("quoteVolume")) or 0.0
            if quote_volume < float(os.getenv("SCANNER_MIN_QUOTE_VOLUME_USDT", "5000000")):
                continue
            ranked.append((quote_volume, symbol))
        ranked.sort(reverse=True)
        if ranked:
            return [symbol for _, symbol in ranked[:limit]], len(symbols), len(ranked), "24h quote volume; minimum USDT 5m"
    except Exception:
        pass
    preferred = [s for s in CRYPTO_DEFAULT if s in symbols]
    return (preferred + [s for s in symbols if s not in preferred])[:limit], len(symbols), None, "major-pair fallback; volume ranking unavailable"


@app.get("/api/scanner")
def scanner(market: str = "all", timeframe: str = "15m", limit: int = 10):
    if timeframe not in TIMEFRAMES:
        raise HTTPException(400, "تایم‌فریم نامعتبر است")
    limit = max(1, min(limit, 20))
    tasks = []
    universe_count, eligible_count, ranking_method = 0, 0, None
    if market in ("all", "crypto"):
        selected, universe_count, eligible_count, ranking_method = get_scanner_universe(limit)
        tasks.extend([("crypto", symbol) for symbol in selected])
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
                quote = get_live_quote(m, s)
                result["live_quote"] = quote
                quote_age = quote.get("age_seconds")
                quote_fresh = bool(quote.get("available") and not quote.get("is_proxy")
                                   and quote_age is not None
                                   and quote_age <= (60 if m == "crypto" else 900))
                result["execution_quality"] = {"blocked": not quote_fresh}
                result["economic_event_risk"] = economic_event_risk(m)
                result["paper_trade"] = record_signal(result)
                results.append({
                    "market": m, "symbol": s, "signal": result["signal"],
                    "setup_score": result["setup_score"], "price": result["price"],
                    "analysis_price": result["price"],
                    "live_price": quote.get("price") if quote.get("available") else None,
                    "quote_source": quote.get("source"),
                    "quote_age_seconds": quote.get("age_seconds"),
                    "bias": result["bias"]
                })
            except Exception as exc:
                results.append({"market": m, "symbol": s, "signal": "ERROR", "error": str(exc)[:150]})
    results.sort(key=lambda item: item.get("setup_score", 0), reverse=True)
    return {
        "timeframe": timeframe, "results": results, "updated_at": utc_now(),
        "universe": {
            "market": market, "crypto_pairs_available": universe_count,
            "crypto_pairs_meeting_liquidity_filter": eligible_count,
            "crypto_pairs_analyzed": sum(1 for item in results if item.get("market") == "crypto"),
            "ranking_method": ranking_method,
            "note": "اسکنر همه نمادها را تحلیل هم‌زمان نمی‌کند؛ ابتدا کل فهرست را با حجم ۲۴ساعته فیلتر و سپس حداکثر limit نماد را تحلیل می‌کند."
        }
    }


@app.post("/api/ai-synthesis")
async def ai_synthesis(request: Request):
    if not AI_API_KEY and not GEMINI_API_KEY:
        raise HTTPException(503, "برای هوش مصنوعی مولد رایگان، GEMINI_API_KEY را در Render ثبت کن؛ بدون کلید فقط تحلیل الگوریتمی فعال است.")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "بدنه درخواست JSON معتبر نیست")
    technical = payload.get("technical")
    if isinstance(technical, dict):
        technical = dict(technical)
        technical.pop("chart_candles", None)
        technical.pop("rsi_chart", None)
        advanced = technical.get("advanced_analysis")
        if isinstance(advanced, dict):
            advanced = dict(advanced)
            advanced.pop("rsi_series", None)
            technical["advanced_analysis"] = advanced
        heatmap = technical.get("liquidation_heatmap")
        if isinstance(heatmap, dict):
            heatmap = dict(heatmap)
            heatmap.pop("liquidation_leverage_data", None)
            heatmap.pop("y_axis", None)
            heatmap.pop("price_candlesticks", None)
            heatmap["clusters"] = (heatmap.get("clusters") or [])[:12]
            technical["liquidation_heatmap"] = heatmap
        payload["technical"] = technical
    context = json.dumps(payload, ensure_ascii=False)[:16000]
    prompt = (
        "تو تحلیلگر محتاط بازار هستی. فقط بر اساس داده‌های ورودی تحلیل کن و هیچ عدد یا خبری نساز. "
        "بین داده زنده، داده با تأخیر، تقریب محاسباتی و داده ناموجود تفاوت بگذار. سود قطعی وعده نده. "
        "ساختار چندتایم‌فریمی را با وزن بیشتر به 4h/1d/1w بررسی کن؛ تعارض تایم‌فریم‌ها را پنهان نکن. "
        "ساختار بازار، BOS/CHOCH، sweep، FVG، نواحی Order Block معتبر، حجم، VWAP، Volume Profile، ADX/ATR، RSI/MACD، "
        "مشتقات کریپتو، تأیید مستقل MarginPad در صورت موجود بودن، ریسک تقویم اقتصادی، دفتر سفارش snapshot، COT هفتگی، داده‌های اقتصاد کلان و همبستگی‌ها را جداگانه ارزیابی کن. "
        "COT هفتگی و داده‌های کلان با تأخیرند؛ دفتر سفارش فقط snapshot است؛ Volume Profile و SMC تقریبی‌اند. "
        "ورودی، حدضرر، اهداف، R/R و شرط ابطال را فقط وقتی داده کافی است ارائه کن؛ در غیر این صورت WAIT. "
        "خروجی فارسی و ساختاریافته باشد: خلاصه، عوامل صعودی، عوامل نزولی، تعارض‌ها، رژیم بازار، سناریوی اصلی، "
        "سناریوی جایگزین، شرط ابطال، ریسک‌های رویدادی، کیفیت داده و دلایل عدم معامله. "
        "امتیازها احتمال موفقیت یا نرخ برد نیستند و هیچ احتمال عددی نساز. داده ورودی: " + context
    )
    try:
        if AI_API_KEY:
            openai_model = AI_MODEL if not AI_MODEL.startswith("gemini-") else os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
            body = json.dumps({"model": openai_model, "input": prompt, "max_output_tokens": 1100}).encode("utf-8")
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=body,
                headers={"Authorization": "Bearer " + AI_API_KEY, "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
            chunks = []
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            chunks.append(part.get("text", ""))
            return {"analysis": "\n".join(chunks), "model": data.get("model", openai_model), "provider": "OpenAI", "updated_at": utc_now()}
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        gemini_body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1100}}).encode("utf-8")
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/" + urllib.parse.quote(model, safe="") + ":generateContent?key=" + urllib.parse.quote(GEMINI_API_KEY, safe="")
        req = urllib.request.Request(gemini_url, data=gemini_body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        chunks = []
        for candidate in data.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                if part.get("text"):
                    chunks.append(part["text"])
        if not chunks:
            raise RuntimeError("Gemini پاسخ متنی برنگرداند")
        return {"analysis": "\n".join(chunks), "model": model, "provider": "Google Gemini", "updated_at": utc_now(), "free_tier_note": "سهمیه رایگان و محدودیت‌های Google AI Studio اعمال می‌شود."}
    except Exception as exc:
        raise HTTPException(502, "درخواست هوش مصنوعی انجام نشد؛ کلید یا سهمیه سرویس را بررسی کن. " + str(exc)[:150])


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


# Background monitor: continues tracking simulated positions while this Render
# process is awake. It never submits live orders.
def _paper_monitor_loop():
    while True:
        try:
            monitor_open_trades(get_live_quote)
        except Exception:
            pass
        time.sleep(max(5, int(os.getenv("PAPER_MONITOR_INTERVAL_SECONDS", "10"))))


_paper_monitor_thread = threading.Thread(
    target=_paper_monitor_loop, name="paper-trading-monitor", daemon=True
)
_paper_monitor_thread.start()
