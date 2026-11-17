"""Gemini-powered primary decision engine for Smart Money AI.

The model interprets supplied market evidence; deterministic validation remains
responsible for rejecting malformed or unsafe trade plans.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import time


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def build_prompt(evidence):
    return (
        "تو مغز تحلیل‌گر اصلی یک سامانه تصمیم‌یار بازار هستی. فقط از شواهد JSON استفاده کن؛ "
        "هیچ قیمت، اندیکاتور، خبر، رویداد یا داده‌ای را حدس نزن. اگر فیلدی null، unavailable یا error است، "
        "آن را داده ناموجود اعلام کن و جایگزین نساز. منابع و زمان/تازگی داده را در کیفیت داده مشخص کن. "
        "تمام تایم‌فریم‌های موجود را بخوان؛ تایم‌فریم بالاتر جهت زمینه‌ای و تایم‌فریم پایین‌تر زمان ورود را می‌دهد. "
        "ساختار SMC/ICT شامل BOS، CHOCH، sweep، FVG، order block، swingها و premium/discount را بررسی کن؛ "
        "همراه با EMA، RSI، MACD، ADX، ATR، VWAP، حجم و Volume Profile. در کریپتو funding، open interest، "
        "long/short، taker ratio، دلتا معاملات و دفتر سفارش را با محدودیت منبعشان تفسیر کن. "
        "برای طلا و نقره خبر، دلار، بازده واقعی، نرخ بهره، تورم، VIX، نفت، COT و همبستگی‌ها را بررسی کن. "
        "خبر یا داده هفتگی/با تأخیر را سیگنال لحظه‌ای جلوه نده. MarginPad فقط تأیید جانبی است. "
        "سه خروجی را با هم قاطی نکن: جهت اصلی بازار، سناریوی جایگزین و تصمیم ورود اکنون. "
        "اگر شواهد متناقض، ناکافی، قدیمی یا قیمت زنده نامعتبر است، decision را WAIT قرار بده. "
        "اگر BUY یا SELL می‌دهی، entry_zone.low/high، entry_trigger، stop_loss و دو target عددی و سازگار بده؛ "
        "برای BUY حدضرر زیر محدوده ورود و اهداف بالای آن، برای SELL برعکس. "
        "قیمت ورود را با قیمت زنده و سطوح ساختاری سازگار کن و هیچ سطحی را از هوا نساز. "
        "خروجی فقط یک JSON معتبر و بدون Markdown با این کلیدها باشد: "
        "{decision:'BUY|SELL|WAIT', summary:string, market_regime:string, "
        "higher_timeframe_bias:string, entry_zone:{low:number,high:number}|null, "
        "entry_trigger:string, stop_loss:number|null, targets:[number], "
        "invalidation:string, bullish_factors:[string], bearish_factors:[string], "
        "conflicts:[string], risk_flags:[string], data_quality:[string], "
        "setup_quality:integer}. setup_quality فقط امتیاز کیفیت شواهد از 0 تا 100 است، نه احتمال برد. "
        "اگر WAIT است، سطوح ورود/حدضرر/اهداف را null یا آرایه خالی قرار بده. "
        "زبان تمام متن‌ها فارسی باشد. شواهد JSON:\\n"
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), default=str)
    )


def _extract_json(text):
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("```")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("Gemini پاسخ JSON معتبر برنگرداند")
    return json.loads(raw[start:end + 1])


def _gemini_request(model, api_key, body):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + urllib.parse.quote(model, safe="")
        + ":generateContent?key=" + urllib.parse.quote(api_key, safe="")
    )
    request = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_with_retry(model, api_key, body, attempts=3):
    """Retry temporary Gemini capacity/server failures without changing evidence."""
    for attempt in range(attempts):
        try:
            return _gemini_request(model, api_key, body)
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 500, 502, 503, 504) or attempt >= attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = min(8.0, max(0.5, float(retry_after)))
            except (TypeError, ValueError):
                delay = min(6.0, 1.5 * (2 ** attempt))
            time.sleep(delay)


def call_gemini(evidence):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY تنظیم نشده است")
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    body = json.dumps({
        "contents": [{"parts": [{"text": build_prompt(evidence)}]}],
        "generationConfig": {
            "temperature": 0.15,
            "maxOutputTokens": 1800,
            "responseMimeType": "application/json"
        }
    }).encode("utf-8")
    try:
        try:
            data = _request_with_retry(model, api_key, body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:500]
            # A stale Render environment variable may still pin an unavailable model.
            if exc.code == 404 and model != DEFAULT_GEMINI_MODEL:
                model = DEFAULT_GEMINI_MODEL
                data = _request_with_retry(model, api_key, body)
            else:
                raise
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:500]
        if exc.code == 429:
            raise RuntimeError("محدودیت سهمیه یا سرعت Gemini فعال است؛ کمی بعد دوباره تلاش کن. " + detail)
        if exc.code == 503:
            raise RuntimeError("سرور Gemini پس از چند تلاش هنوز شلوغ است (503)؛ این خطا به‌تنهایی به معنی تمام‌شدن سهمیه نیست. چند دقیقه بعد دوباره تلاش کن. " + detail)
        if exc.code in (500, 502, 504, 408):
            raise RuntimeError("اختلال موقت در سرور Gemini پس از چند تلاش: HTTP " + str(exc.code) + ". بعداً دوباره تلاش کن. " + detail)
        if exc.code == 404:
            raise RuntimeError("مدل Gemini در دسترس نیست: " + detail)
        raise RuntimeError("خطای Gemini HTTP " + str(exc.code) + ": " + detail)
    candidates = data.get("candidates") or []
    chunks = []
    for candidate in candidates:
        for part in (candidate.get("content") or {}).get("parts", []):
            if part.get("text"):
                chunks.append(part["text"])
    if not chunks:
        reason = (candidates[0].get("finishReason") if candidates else None) or "پاسخ خالی"
        raise RuntimeError("Gemini پاسخ قابل استفاده نداد: " + str(reason))
    result = _extract_json("\\n".join(chunks))
    if not isinstance(result, dict):
        raise RuntimeError("ساختار پاسخ هوش مصنوعی معتبر نیست")
    result["provider"] = "Google Gemini"
    result["model"] = model
    return result


def validate_decision(result, live_quote=None, execution_quality=None):
    """Downgrade unsafe or incomplete directional outputs to WAIT."""
    result = dict(result or {})
    decision = str(result.get("decision", "WAIT")).upper()
    if decision not in ("BUY", "SELL", "WAIT"):
        decision = "WAIT"
    result["decision"] = decision
    try:
        score = int(result.get("setup_quality", 0))
    except (TypeError, ValueError):
        score = 0
    result["setup_quality"] = max(0, min(100, score))
    if decision == "WAIT":
        result.update({"entry_zone": None, "stop_loss": None, "targets": []})
        return result

    def number(value):
        try:
            n = float(value)
            return n if n > 0 else None
        except (TypeError, ValueError):
            return None

    zone = result.get("entry_zone") or {}
    low, high = number(zone.get("low")), number(zone.get("high"))
    stop = number(result.get("stop_loss"))
    targets = [number(x) for x in (result.get("targets") or [])]
    targets = [x for x in targets if x is not None]
    valid = low is not None and high is not None and low <= high and stop is not None and len(targets) >= 2
    if valid and decision == "BUY":
        valid = stop < low and all(x > high for x in targets[:2]) and targets[0] < targets[1]
    if valid and decision == "SELL":
        valid = stop > high and all(x < low for x in targets[:2]) and targets[0] > targets[1]
    if valid:
        midpoint = (low + high) / 2.0
        risk_distance = abs(midpoint - stop)
        reward_distance = abs(targets[0] - midpoint)
        valid = risk_distance > 0 and reward_distance / risk_distance >= 1.2
    if execution_quality and execution_quality.get("blocked"):
        valid = False
        result.setdefault("risk_flags", []).append("ورود به‌دلیل کیفیت قیمت یا اجرا مسدود است.")
    quote_ok = bool((live_quote or {}).get("available") and not (live_quote or {}).get("is_proxy"))
    if not quote_ok:
        valid = False
        result.setdefault("risk_flags", []).append("قیمت زنده معتبر در دسترس نیست؛ فقط WAIT مجاز است.")
    if not valid:
        result["decision"] = "WAIT"
        result["rejected_model_plan"] = {
            "entry_zone": result.get("entry_zone"),
            "stop_loss": result.get("stop_loss"),
            "targets": result.get("targets")
        }
        result.update({
            "entry_zone": None, "stop_loss": None, "targets": [],
            "summary": (str(result.get("summary") or "") + " | پلن عددی مدل از کنترل صحت عبور نکرد؛ ورود صادر نشد.").strip()
        })
        result.setdefault("conflicts", []).append("پلن عددی با قواعد کنترل ریسک و سازگاری قیمت تأیید نشد.")
    else:
        result["entry_zone"] = {"low": low, "high": high}
        result["stop_loss"] = stop
        result["targets"] = targets[:3]
    return result
