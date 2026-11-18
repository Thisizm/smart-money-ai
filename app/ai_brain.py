"""Gemini-powered primary decision engine for Smart Money AI.

The model interprets supplied market evidence; deterministic validation remains
responsible for rejecting malformed or unsafe trade plans.
"""
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import time
import re


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def build_prompt(evidence):
    return (
        "تو تحلیل‌گر اصلی بازار هستی. فقط از JSON شواهد استفاده کن و هیچ داده‌ای نساز. "
        "پاسخ نهایی را کوتاه، دقیق و فقط به شکل یک JSON معتبر تولید کن؛ استدلال طولانی ننویس. "
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
        "زبان تمام متن‌ها فارسی باشد. summary حداکثر ۲ جمله، هر فهرست حداکثر ۴ مورد، "
        "entry_trigger و invalidation هرکدام حداکثر ۱ جمله باشند. فقط کلیدهای خواسته‌شده را برگردان. "
        "شواهد JSON:\\n"
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
        raise RuntimeError("مدل پاسخ JSON معتبر برنگرداند")
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


def _http_error_body(exc):
    raw = exc.read().decode("utf-8", "ignore")[:4000]
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else {}
    error = error if isinstance(error, dict) else {}
    message = str(error.get("message") or raw or exc.reason or "بدون توضیح")
    status = str(error.get("status") or "")
    details = error.get("details") or []
    detail_text = json.dumps(details, ensure_ascii=False, default=str)
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if not retry_after:
        match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', detail_text)
        if match:
            retry_after = match.group(1)
    try:
        retry_seconds = max(0.0, float(retry_after)) if retry_after is not None else None
    except (TypeError, ValueError):
        retry_seconds = None
    quota_text = (message + " " + status + " " + detail_text).lower()
    daily_quota = any(token in quota_text for token in (
        "per_day", "per day", "daily", "requests_per_day",
        "generate_content_free_tier_requests_per_day"
    ))
    return message[:1000], retry_seconds, daily_quota


def _request_with_retry(model, api_key, body, attempts=2):
    """Retry temporary Gemini capacity/server failures and read timeouts."""
    for attempt in range(attempts):
        try:
            return _gemini_request(model, api_key, body)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                detail, retry_seconds, daily_quota = _http_error_body(exc)
                if daily_quota:
                    raise RuntimeError(
                        "سهمیه روزانه Gemini تمام شده یا برای این مدل در پلن رایگان مجاز نیست؛ "
                        "تلاش مجدد خودکار انجام نشد تا درخواست بی‌فایده تکرار نشود. جزئیات: " + detail
                    ) from exc
                delay = retry_seconds if retry_seconds is not None else 2.0
                if attempt < attempts - 1 and delay <= 12:
                    time.sleep(max(0.5, delay))
                    continue
                wait_note = (
                    "حدود " + str(round(delay)) + " ثانیه دیگر"
                    if delay > 0 else "پس از کاهش تعداد درخواست‌ها"
                )
                raise RuntimeError(
                    "محدودیت سرعت/تعداد درخواست Gemini فعال است؛ " + wait_note +
                    " دوباره امتحان کن. درخواست تکراری بی‌دلیل ارسال نشد. جزئیات: " + detail
                ) from exc
            if exc.code not in (408, 500, 502, 503, 504) or attempt >= attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = min(8.0, max(0.5, float(retry_after)))
            except (TypeError, ValueError):
                delay = min(6.0, 1.5 * (2 ** attempt))
            time.sleep(delay)
        except (socket.timeout, TimeoutError) as exc:
            if attempt >= attempts - 1:
                raise RuntimeError(
                    "زمان انتظار پاسخ Gemini تمام شد؛ ارتباط یا سرور مدل موقتاً کند است. "
                    "این خطا به‌تنهایی نشانه تمام‌شدن سهمیه رایگان نیست. کمی بعد دوباره تلاش کن."
                ) from exc
            time.sleep(1.5)
        except urllib.error.URLError as exc:
            if not isinstance(getattr(exc, "reason", None), (socket.timeout, TimeoutError)):
                raise
            if attempt >= attempts - 1:
                raise RuntimeError(
                    "زمان انتظار پاسخ Gemini تمام شد؛ ارتباط یا سرور مدل موقتاً کند است. "
                    "این خطا به‌تنهایی نشانه تمام‌شدن سهمیه رایگان نیست. کمی بعد دوباره تلاش کن."
                ) from exc
            time.sleep(1.5)


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
            message, retry_seconds, daily_quota = _http_error_body(exc)
            if daily_quota:
                raise RuntimeError("سهمیه روزانه Gemini تمام شده یا برای این مدل در پلن رایگان مجاز نیست. " + message)
            raise RuntimeError("محدودیت سرعت/تعداد درخواست Gemini فعال است؛ کمی بعد دوباره تلاش کن. " + message)
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


DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def _openrouter_text(value):
    """Normalize the text shapes returned by OpenRouter chat completions."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks = []
        for part in value:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    chunks.append(text_value)
        return "\n".join(chunks).strip()
    if isinstance(value, dict):
        text_value = value.get("text")
        return text_value.strip() if isinstance(text_value, str) else ""
    return ""


def call_openrouter(evidence):
    """Primary free-tier reasoning model through OpenRouter; deterministic validation stays downstream."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY در تنظیمات Render ثبت نشده است")
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(evidence)}],
        "temperature": 0.1,
        # Keep internal reasoning from consuming the entire completion budget.
        # OpenRouter counts reasoning tokens inside max_tokens for this endpoint.
        "reasoning": {"max_tokens": 1200, "exclude": True},
        "max_tokens": 6000
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://smart-money-ai-t07z.onrender.com/",
            "X-Title": "Smart Money AI"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:1000]
        if exc.code in (401, 403):
            raise RuntimeError("کلید OpenRouter معتبر نیست یا دسترسی ندارد: " + detail)
        if exc.code == 429:
            raise RuntimeError("OpenRouter محدودیت درخواست یا سهمیه را برگرداند (429). جزئیات: " + detail)
        if exc.code in (502, 503, 504):
            raise RuntimeError("مدل رایگان OpenRouter موقتاً در دسترس نیست. جزئیات: " + detail)
        raise RuntimeError("خطای OpenRouter HTTP " + str(exc.code) + ": " + detail)
    except (socket.timeout, TimeoutError) as exc:
        raise RuntimeError("پاسخ OpenRouter بیش از زمان مجاز طول کشید؛ این خطا به‌تنهایی نشانه اتمام سهمیه نیست.") from exc

    if not isinstance(data, dict):
        raise RuntimeError("ساختار پاسخ OpenRouter معتبر نیست")
    if isinstance(data.get("error"), dict):
        err = data["error"]
        message = str(err.get("message") or err.get("code") or "خطای نامشخص")
        raise RuntimeError("OpenRouter خطا را داخل پاسخ برگرداند: " + message[:700])

    choices = data.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = _openrouter_text(message.get("content"))
    if not content:
        # Do not expose hidden reasoning or raw provider payloads to the user.
        finish = choice.get("finish_reason") or choice.get("native_finish_reason") or "نامشخص"
        provider = data.get("provider") or "نامشخص"
        returned_model = data.get("model") or model
        message_keys = ",".join(sorted(str(k) for k in message.keys())) or "ندارد"
        if finish == "length":
            hint = "؛ احتمالاً سقف خروجی مدل پر شده است"
        elif message.get("refusal"):
            hint = "؛ مدل درخواست را رد کرده است"
        elif message.get("tool_calls"):
            hint = "؛ مدل به‌جای متن، درخواست ابزار برگردانده است"
        else:
            hint = ""
        raise RuntimeError(
            "OpenRouter متن نهایی برنگرداند (finish_reason=" + str(finish) +
            ", provider=" + str(provider) + ", model=" + str(returned_model) +
            ", message_fields=" + message_keys + ")" + hint
        )

    try:
        result = _extract_json(content)
    except (ValueError, TypeError) as exc:
        preview = content[:240].replace("\n", " ")
        raise RuntimeError(
            "مدل پاسخ داد اما JSON قابل‌خواندن نبود؛ ابتدای پاسخ: " + preview
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError("ساختار پاسخ هوش مصنوعی معتبر نیست")
    result["provider"] = "OpenRouter"
    result["model"] = data.get("model") or model
    return result
