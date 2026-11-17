
import os
import json
import sqlite3
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
DB_PATH = os.path.join(ROOT_DIR, "alerts.db")

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")

app = FastAPI(title="Smart Money AI Market Scanner")

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(ROOT_DIR, "static")),
    name="static"
)

CRYPTO = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT",
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"
]

FX = [
    "EUR/USD", "GBP/USD", "USD/JPY",
    "AUD/USD", "USD/CHF", "USD/CAD"
]

TIMEFRAMES = {
    "5m": {"binance": "5m", "twelve": "5min"},
    "15m": {"binance": "15m", "twelve": "15min"},
    "1h": {"binance": "1h", "twelve": "1h"},
    "4h": {"binance": "4h", "twelve": "4h"},
    "1d": {"binance": "1d", "twelve": "1day"},
    "1w": {"binance": "1w", "twelve": "1week"},
}


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


def fetch_json(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SmartMoneyAI/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        raise HTTPException(
            502,
            f"Data provider error {e.code}: {body[:200]}"
        )

    except Exception as e:
        raise HTTPException(
            502,
            f"خطا در دریافت داده بازار: {str(e)[:150]}"
        )


def get_candles(market, symbol, timeframe):

    if timeframe not in TIMEFRAMES:
        raise HTTPException(400, "تایم‌فریم نامعتبر است")

    tf = TIMEFRAMES[timeframe]

    if market == "crypto":

        if symbol not in CRYPTO:
            raise HTTPException(400, "نماد کریپتو نامعتبر است")

        params = urllib.parse.urlencode({
            "symbol": symbol,
            "interval": tf["binance"],
            "limit": 250
        })

        url = (
            "https://data-api.binance.vision/api/v3/klines?"
            + params
        )

        raw = fetch_json(url)

        return [
            {
                "time": int(x[0]),
                "open": float(x[1]),
                "high": float(x[2]),
                "low": float(x[3]),
                "close": float(x[4]),
                "volume": float(x[5])
            }
            for x in raw
        ]

    if market in ("forex", "gold"):

        if not TWELVE_DATA_API_KEY:
            raise HTTPException(
                400,
                "ابتدا TWELVE_DATA_API_KEY را در Render تنظیم کن."
            )

        if market == "gold":
            symbol = "XAU/USD"

        elif symbol not in FX:
            raise HTTPException(400, "نماد فارکس نامعتبر است")

        params = urllib.parse.urlencode({
            "symbol": symbol,
            "interval": tf["twelve"],
            "outputsize": 250,
            "apikey": TWELVE_DATA_API_KEY,
            "format": "JSON"
        })

        data = fetch_json(
            "https://api.twelvedata.com/time_series?" + params
        )

        if "values" not in data:
            raise HTTPException(
                502,
                "ارائه‌دهنده داده این نماد یا تایم‌فریم را در پلن فعلی ارائه نمی‌کند."
            )

        values = list(reversed(data["values"]))
        candles = []

        for x in values:

            try:
                dt = datetime.strptime(
                    x["datetime"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)

            except ValueError:
                dt = datetime.strptime(
                    x["datetime"], "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)

            candles.append({
                "time": int(dt.timestamp() * 1000),
                "open": float(x["open"]),
                "high": float(x["high"]),
                "low": float(x["low"]),
                "close": float(x["close"]),
                "volume": float(x.get("volume") or 0)
            })

        return candles

    raise HTTPException(400, "بازار نامعتبر است")


def ema(values, period):

    if len(values) < period:
        return values[-1]

    k = 2 / (period + 1)
    value = sum(values[:period]) / period

    for v in values[period:]:
        value = v * k + value * (1 - k)

    return value


def analyze(candles):

    if len(candles) < 50:
        raise HTTPException(502, "داده کافی دریافت نشد")

    closes = [c["close"] for c in candles]
    last = candles[-1]

    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)

    recent = candles[-21:-1]

    swing_high = max(c["high"] for c in recent)
    swing_low = min(c["low"] for c in recent)

    trs = []

    for i in range(1, len(candles)):
        c = candles[i]
        p = candles[i - 1]

        trs.append(max(
            c["high"] - c["low"],
            abs(c["high"] - p["close"]),
            abs(c["low"] - p["close"])
        ))

    atr = sum(trs[-14:]) / 14

    bullish_fvg = candles[-1]["low"] > candles[-3]["high"]
    bearish_fvg = candles[-1]["high"] < candles[-3]["low"]

    bos_up = last["close"] > swing_high
    bos_down = last["close"] < swing_low

    score = 0

    score += 1 if e9 > e21 else -1
    score += 1 if e21 > e50 else -1
    score += 1 if last["close"] > e21 else -1

    score += 1 if bos_up else (
        -1 if bos_down else 0
    )

    score += 1 if bullish_fvg else (
        -1 if bearish_fvg else 0
    )

    if score >= 3:
        signal = "BUY"
        bias = "صعودی"

    elif score <= -3:
        signal = "SELL"
        bias = "نزولی"

    else:
        signal = "WAIT"
        bias = "خنثی"

    price = last["close"]

    if signal == "BUY":

        stop = min(swing_low, price - 1.5 * atr)
        risk = max(price - stop, atr)

        tp1 = price + 1.5 * risk
        tp2 = price + 2.5 * risk

    elif signal == "SELL":

        stop = max(swing_high, price + 1.5 * atr)
        risk = max(stop - price, atr)

        tp1 = price - 1.5 * risk
        tp2 = price - 2.5 * risk

    else:
        stop = None
        tp1 = None
        tp2 = None

    notes = [
        "EMA 9/21: " + (
            "صعودی" if e9 > e21 else "نزولی"
        ),
        "EMA 21/50: " + (
            "صعودی" if e21 > e50 else "نزولی"
        ),
        "BOS صعودی" if bos_up else (
            "BOS نزولی" if bos_down else
            "شکست ساختار تأیید نشد"
        ),
        "FVG صعودی" if bullish_fvg else (
            "FVG نزولی" if bearish_fvg else
            "FVG شناسایی نشد"
        )
    ]

    return {
        "signal": signal,
        "bias": bias,
        "score": score,
        "price": price,
        "stop_loss": stop,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "atr": atr,
        "analysis": notes,
        "candles_count": len(candles),
        "last_candle_time": last["time"]
    }


@app.get("/", response_class=HTMLResponse)
def home():

    path = os.path.join(ROOT_DIR, "static", "index.html")

    with open(path, encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():

    return {
        "status": "ok",
        "app": "Smart Money AI Market Scanner"
    }


@app.get("/api/status")
def status():

    return {
        "status": "online",
        "crypto_data": "public Binance market data",
        "twelve_data_key_configured": bool(TWELVE_DATA_API_KEY),
        "supported_timeframes": list(TIMEFRAMES.keys())
    }


@app.get("/api/options")
def options():

    return {
        "crypto": CRYPTO,
        "forex": FX,
        "gold": ["XAU/USD"],
        "timeframes": list(TIMEFRAMES.keys())
    }


@app.get("/api/analyze")
def api_analyze(
    market: str = "crypto",
    symbol: str = "BTCUSDT",
    timeframe: str = "15m"
):

    candles = get_candles(market, symbol, timeframe)
    result = analyze(candles)

    result.update({
        "market": market,
        "symbol": "XAU/USD" if market == "gold" else symbol,
        "timeframe": timeframe,
        "updated_at": datetime.now(timezone.utc).isoformat()
    })

    return result


@app.post("/webhook")
async def webhook(request: Request):

    data = await request.json()

    if WEBHOOK_TOKEN and data.get("token") != WEBHOOK_TOKEN:
        raise HTTPException(401, "Invalid webhook token")

    with db() as con:

        con.execute("""
        INSERT INTO alerts
        (created_at, symbol, timeframe, signal, price, message)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            data.get("symbol", "UNKNOWN"),
            data.get("timeframe", ""),
            data.get("signal", ""),
            data.get("price"),
            data.get("message", "")
        ))

    return {"ok": True}


@app.get("/api/alerts")
def alerts():

    with db() as con:

        rows = con.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT 50"
        ).fetchall()

    return [dict(row) for row in rows]
