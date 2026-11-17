
import os
import json
import sqlite3
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = BASE_DIR / "alerts.db"

app = FastAPI(
    title="Smart Money AI",
    description="TradingView webhook and market dashboard",
    version="1.0.0"
)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                signal TEXT,
                price REAL,
                message TEXT,
                created_at TEXT
            )
        """)
        conn.commit()

init_db()

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse(
            "<h2>Smart Money AI</h2><p>Dashboard file not found.</p>",
            status_code=200
        )
    return HTMLResponse(index_file.read_text(encoding="utf-8"))

@app.get("/health")
async def health():
    return {"status": "ok", "app": "Smart Money AI"}

@app.post("/webhook")
async def tradingview_webhook(request: Request):
    expected_token = os.getenv("WEBHOOK_TOKEN", "")
    data = await request.json()

    received_token = str(data.get("token", ""))

    if expected_token and received_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid webhook token")

    symbol = str(data.get("symbol", "UNKNOWN"))
    timeframe = str(data.get("timeframe", ""))
    signal = str(data.get("signal", ""))
    message = str(data.get("message", ""))
    price = data.get("price")

    try:
        price = float(price) if price is not None else None
    except (TypeError, ValueError):
        price = None

    created_at = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO alerts
            (symbol, timeframe, signal, price, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, timeframe, signal, price, message, created_at))
        alert_id = cursor.lastrowid
        conn.commit()

    return {
        "success": True,
        "alert_id": alert_id,
        "symbol": symbol,
        "signal": signal
    }

@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    limit = max(1, min(limit, 200))

    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM alerts
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]

@app.get("/api/news")
async def get_news():
    feed_url = os.getenv("NEWS_RSS_URL", "")

    if not feed_url:
        return {"items": [], "message": "NEWS_RSS_URL is not configured"}

    try:
        req = urllib.request.Request(
            feed_url,
            headers={"User-Agent": "SmartMoneyAI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()

        root = ET.fromstring(content)
        items = []

        for item in root.findall(".//item")[:20]:
            items.append({
                "title": item.findtext("title", default=""),
                "link": item.findtext("link", default=""),
                "published": item.findtext("pubDate", default="")
            })

        return {"items": items}

    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"items": [], "error": str(exc)}
        )

@app.get("/api/status")
async def status():
    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM alerts"
        ).fetchone()[0]

    return {
        "app": "Smart Money AI",
        "status": "running",
        "alerts_count": count,
        "ai_configured": bool(os.getenv("AI_API_KEY"))
    }
