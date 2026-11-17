import os
import json
import sqlite3
import urllib.parse
import urllib.request
import urllib.error
import hmac
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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


def analyze(candles, market, symbol, timeframe):
    if len(candles) < 60:
        raise HTTPException(502, "برای تحلیل معتبرتر حداقل ۶۰ کندل لازم است")
    closes = [c["close"] for c in candles]
    last = candles[-1]
    previous = candles[-2]
    e9, e21, e50 = ema(closes, 9), ema(closes, 21), ema(closes, 50)

    window = candles[-21:-1]
    swing_high = max(c["high"] for c in window)
    swing_low = min(c["low"] for c in window)
    bos_up = last["close"] > swing_high
    bos_down = last["close"] < swing_low
    sweep_high = last["high"] > swing_high and last["close"] < swing_high
    sweep_low = last["low"] < swing_low and last["close"] > swing_low
    bullish_fvg = last["low"] > candles[-3]["high"]
    bearish_fvg = last["high"] < candles[-3]["low"]

    trs = []
    for index in range(1, len(candles)):
        current, prior = candles[index], candles[index - 1]
        trs.append(max(
            current["high"] - current["low"],
            abs(current["high"] - prior["close"]),
            abs(current["low"] - prior["close"])
        ))
    atr = sum(trs[-14:]) / max(1, len(trs[-14:]))

    avg_volume = sum(c["volume"] for c in candles[-21:-1]) / 20
    volume_confirm = last["volume"] > avg_volume if avg_volume > 0 else False

    bull = 0
    bear = 0
    bull += 20 if e9 > e21 else 0
    bear += 20 if e9 < e21 else 0
    bull += 15 if e21 > e50 else 0
    bear += 15 if e21 < e50 else 0
    bull += 10 if last["close"] > e21 else 0
    bear += 10 if last["close"] < e21 else 0
    bull += 25 if bos_up else 0
    bear += 25 if bos_down else 0
    bull += 15 if sweep_low else 0
    bear += 15 if sweep_high else 0
    bull += 10 if bullish_fvg else 0
    bear += 10 if bearish_fvg else 0
    bull += 5 if volume_confirm and last["close"] >= previous["close"] else 0
    bear += 5 if volume_confirm and last["close"] <= previous["close"] else 0

    direction = "BUY" if bull >= 60 and bull - bear >= 15 else (
        "SELL" if bear >= 60 and bear - bull >= 15 else "WAIT"
    )
    setup_score = max(bull, bear)
    price = last["close"]

    if direction == "BUY":
        stop = min(swing_low, price - 1.5 * atr)
        risk = max(price - stop, atr)
        targets = [price + risk * r for r in (1.5, 2.5, 3.5)]
    elif direction == "SELL":
        stop = max(swing_high, price + 1.5 * atr)
        risk = max(stop - price, atr)
        targets = [price - risk * r for r in (1.5, 2.5, 3.5)]
    else:
        stop, targets, risk = None, [None, None, None], None

    prior_trend_up = ema(closes[:-1], 21) > ema(closes[:-1], 50)
    choch = "صعودی" if bos_up and not prior_trend_up else (
        "نزولی" if bos_down and prior_trend_up else "تأیید نشده"
    )

    order_block = None
    if bos_up:
        for candle in reversed(candles[-12:-1]):
            if candle["close"] < candle["open"]:
                order_block = {"type": "تقریبی صعودی", "low": candle["low"], "high": candle["high"]}
                break
    elif bos_down:
        for candle in reversed(candles[-12:-1]):
            if candle["close"] > candle["open"]:
                order_block = {"type": "تقریبی نزولی", "low": candle["low"], "high": candle["high"]}
                break

    notes = [
        "EMA 9/21: " + ("صعودی" if e9 > e21 else "نزولی"),
        "EMA 21/50: " + ("صعودی" if e21 > e50 else "نزولی"),
        "ساختار: " + ("BOS صعودی" if bos_up else "BOS نزولی" if bos_down else "شکست ساختار دیده نشد"),
        "تغییر کاراکتر (CHOCH): " + choch,
        "لیکوییدیتی: " + ("جمع‌آوری نقدینگی زیر کف" if sweep_low else "جمع‌آوری نقدینگی بالای سقف" if sweep_high else "سوییپ واضح دیده نشد"),
        "FVG: " + ("گپ ارزش منصفانه صعودی" if bullish_fvg else "گپ ارزش منصفانه نزولی" if bearish_fvg else "FVG واضح دیده نشد"),
        "حجم: " + ("بالاتر از میانگین ۲۰ کندل" if volume_confirm else "تأیید حجمی ندارد")
    ]
    return {
        "market": market, "symbol": symbol, "timeframe": timeframe,
        "signal": direction, "bias": "صعودی" if direction == "BUY" else "نزولی" if direction == "SELL" else "خنثی",
        "setup_score": setup_score, "bull_score": bull, "bear_score": bear,
        "score_note": "امتیاز تجربی هم‌جهتی شروط است و احتمال موفقیت معامله نیست.",
        "price": price, "atr": atr, "stop_loss": stop,
        "take_profit_1": targets[0], "take_profit_2": targets[1], "take_profit_3": targets[2],
        "risk_per_unit": risk, "swing_high": swing_high, "swing_low": swing_low,
        "bos": "صعودی" if bos_up else "نزولی" if bos_down else "ندارد",
        "choch": choch, "liquidity_sweep": "پایین" if sweep_low else "بالا" if sweep_high else "ندارد",
        "fvg": "صعودی" if bullish_fvg else "نزولی" if bearish_fvg else "ندارد",
        "order_block_approx": order_block, "analysis": notes,
        "candles_count": len(candles), "last_candle_time": last["time"],
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


@app.get("/api/analyze")
def api_analyze(market: str = "crypto", symbol: str = "BTCUSDT", timeframe: str = "15m"):
    candles = get_candles(market, symbol, timeframe)
    return analyze(candles, market, symbol, timeframe)


@app.get("/api/multi-timeframe")
def multi_timeframe(market: str = "crypto", symbol: str = "BTCUSDT"):
    results = []
    for tf in ("1h", "4h", "1d"):
        try:
            candles = get_candles(market, symbol, tf)
            results.append(analyze(candles, market, symbol, tf))
        except HTTPException as exc:
            results.append({"timeframe": tf, "error": exc.detail})
    available = [item for item in results if "signal" in item]
    buys = sum(1 for item in available if item["signal"] == "BUY")
    sells = sum(1 for item in available if item["signal"] == "SELL")
    overall = "BUY" if buys >= 2 else "SELL" if sells >= 2 else "WAIT"
    return {"market": market, "symbol": symbol, "overall": overall, "timeframes": results, "updated_at": utc_now()}


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
    if asset not in ("gold", "silver", "crypto"):
        raise HTTPException(400, "asset باید gold، silver یا crypto باشد")
    now = datetime.now(timezone.utc).timestamp()
    cached = NEWS_CACHE.get(asset)
    if cached and now - cached["at"] < 600:
        return cached["data"]
    params = urllib.parse.urlencode({
        "query": news_query(asset), "mode": "ArtList", "format": "json",
        "maxrecords": 20, "timespan": "3d", "sort": "DateDesc"
    })
    try:
        raw = fetch_json("https://api.gdeltproject.org/api/v2/doc/doc?" + params, timeout=20)
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
            "source": "GDELT DOC 2.0", "updated_at": utc_now(),
            "disclaimer": "برچسب‌ها بر اساس کلمات تیتر هستند؛ تحلیل قطعی اثر قیمت یا جایگزین مطالعه خبر نیستند."
        }
        NEWS_CACHE[asset] = {"at": now, "data": result}
        return result
    except Exception as exc:
        raise HTTPException(502, "دریافت خبر از GDELT ناموفق بود: " + str(exc)[:140])


FRED_SERIES = {
    "DGS10": {"label": "بازده اوراق ۱۰ ساله آمریکا", "unit": "%"},
    "DFII10": {"label": "بازده واقعی اوراق ۱۰ ساله آمریکا", "unit": "%"},
    "FEDFUNDS": {"label": "نرخ مؤثر فدرال رزرو", "unit": "%"},
    "CPIAUCSL": {"label": "شاخص قیمت مصرف‌کننده آمریکا", "unit": "index"},
    "UNRATE": {"label": "نرخ بیکاری آمریکا", "unit": "%"}
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
            "not_connected": ["DXY", "COT positioning", "Gold ETF flows", "Silver ETF flows", "central-bank gold purchases"]
        }
    now = datetime.now(timezone.utc).timestamp()
    if MACRO_CACHE["data"] and now - MACRO_CACHE["at"] < 3600:
        return MACRO_CACHE["data"]
    indicators = []
    errors = []
    for series_id in FRED_SERIES:
        try:
            indicators.append(fred_observation(series_id))
        except Exception as exc:
            errors.append(series_id + ": " + str(exc)[:100])
    result = {
        "available": bool(indicators), "source": "Federal Reserve Economic Data (FRED)",
        "indicators": indicators, "errors": errors,
        "not_connected": ["DXY", "COT positioning", "Gold ETF flows", "Silver ETF flows", "central-bank gold purchases"],
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
        {"factor": "دلار آمریکا (DXY)", "status": "نیازمند اتصال منبع داده اختصاصی", "impact": "معمولاً دلار قوی می‌تواند فشار بر فلزات ایجاد کند؛ رابطه قطعی نیست"},
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
    limit = max(1, min(limit, 10))
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
