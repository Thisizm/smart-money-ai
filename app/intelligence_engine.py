"""Hidden multi-source decision layer for Smart Money AI.

This module only scores evidence that is explicitly available. Missing provider
metrics are never fabricated. Scores are confluence diagnostics, not win
probabilities. The layer is intentionally backend-only; it does not add chart
overlays or UI controls.
"""


def _number(value):
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _aligned(value, side, threshold=0.0):
    number = _number(value)
    if number is None:
        return False
    return number > threshold if side == "long" else number < -threshold


def assess_hidden_intelligence(
    market,
    analysis,
    derivatives=None,
    dominance=None,
    market_context=None,
    event_risk=None,
):
    """Evaluate independent evidence groups and decide whether to keep a signal."""
    analysis = analysis or {}
    derivatives = derivatives or {}
    dominance = dominance or {}
    market_context = market_context or {}
    event_risk = event_risk or {}

    base_signal = str(analysis.get("signal") or "WAIT").upper()
    side = "long" if base_signal == "BUY" else "short" if base_signal == "SELL" else None
    advanced = analysis.get("advanced_analysis") or {}
    adx = advanced.get("adx") or {}
    volatility = advanced.get("volatility") or {}
    momentum = advanced.get("momentum") or {}
    profile = advanced.get("volume_profile") or {}
    price = _number(analysis.get("price"))

    groups = {}
    missing = []
    cautions = []
    score_weights = {"structure": 25, "momentum": 20, "auction": 15, "flow": 20, "derivatives": 10}

    # 1) Structure: pivots, BOS/CHOCH, sweep, FVG and validated OB.
    structure = str(analysis.get("market_structure") or "unknown").lower()
    bos = str(analysis.get("bos") or "").lower()
    choch = str(analysis.get("choch") or "").lower()
    sweep = str(analysis.get("liquidity_sweep") or "").lower()
    fvg = str(analysis.get("fvg") or "").lower()
    ob = analysis.get("order_block_approx") or {}
    ob_type = str(ob.get("type") or "").lower()
    if side == "long":
        structure_aligned = (
            "bullish" in structure or "صعودی" in structure or
            "صعودی" in bos or "صعودی" in choch or sweep == "پایین" or
            "صعودی" in fvg or "صعودی" in ob_type
        )
        structure_event = "صعودی" in bos or "صعودی" in choch or sweep == "پایین"
    elif side == "short":
        structure_aligned = (
            "bearish" in structure or "نزولی" in structure or
            "نزولی" in bos or "نزولی" in choch or sweep == "بالا" or
            "نزولی" in fvg or "نزولی" in ob_type
        )
        structure_event = "نزولی" in bos or "نزولی" in choch or sweep == "بالا"
    else:
        structure_aligned = False
        structure_event = False
    structure_available = structure not in ("", "unknown", "none") or any(
        value not in ("", "ندارد", "تأیید نشده", "none")
        for value in (bos, choch, sweep, fvg)
    )
    structure_score = 25 if structure_event else 18 if structure_aligned else 0
    groups["structure"] = {
        "available": structure_available, "aligned": bool(structure_aligned),
        "score": structure_score, "weight": score_weights["structure"],
        "details": "ساختار، شکست، سوییپ، FVG و ناحیه سفارش بررسی شد."
    }
    if not structure_available:
        missing.append("ساختار بازار")

    # 2) Momentum: RSI, MACD and directional movement.
    rsi = _number(analysis.get("rsi", advanced.get("rsi_wilder_14")))
    macd_hist = _number((advanced.get("macd") or {}).get("histogram"))
    plus_di = _number(adx.get("plus_di"))
    minus_di = _number(adx.get("minus_di"))
    momentum_checks = []
    if rsi is not None:
        momentum_checks.append((rsi >= 50 if side == "long" else rsi <= 50) if side else False)
        if side == "long" and rsi >= 78:
            cautions.append("RSI بسیار بالا است؛ احتمال تعقیب قیمت خرید را بررسی کن.")
        if side == "short" and rsi <= 22:
            cautions.append("RSI بسیار پایین است؛ احتمال تعقیب قیمت فروش را بررسی کن.")
    if macd_hist is not None and side:
        momentum_checks.append(macd_hist > 0 if side == "long" else macd_hist < 0)
    if plus_di is not None and minus_di is not None and side:
        momentum_checks.append(plus_di > minus_di if side == "long" else minus_di > plus_di)
    momentum_available = len(momentum_checks) >= 2
    momentum_aligned = momentum_available and sum(1 for item in momentum_checks if item) >= 2
    groups["momentum"] = {
        "available": momentum_available, "aligned": bool(momentum_aligned),
        "score": 20 if momentum_aligned else 0, "weight": score_weights["momentum"],
        "details": "RSI، MACD و جهت DI/ADX؛ حداقل دو معیار لازم است."
    }
    if not momentum_available:
        missing.append("تأیید مومنتوم کامل")

    # 3) Auction/value: VWAP and approximate OHLCV volume profile.
    vwap = _number(momentum.get("vwap_utc_day"))
    poc = _number(profile.get("poc")) if profile.get("available") else None
    auction_checks = []
    if price is not None and vwap is not None and side:
        auction_checks.append(price >= vwap if side == "long" else price <= vwap)
    if price is not None and poc is not None and side:
        auction_checks.append(price >= poc if side == "long" else price <= poc)
    auction_available = bool(auction_checks)
    auction_aligned = auction_available and any(auction_checks)
    groups["auction"] = {
        "available": auction_available, "aligned": bool(auction_aligned),
        "score": (15 if auction_aligned else 0), "weight": score_weights["auction"],
        "details": "جایگاه قیمت نسبت به VWAP روزانه و POC تقریبی پروفایل حجم."
    }
    if not auction_available:
        missing.append("VWAP/Volume Profile قابل استفاده")

    # 4) Observable trade flow: recent Binance aggregate-trade delta and spot depth.
    cvd_delta = _number(derivatives.get("cvd_delta_percent"))
    book_imbalance = _number(derivatives.get("spot_orderbook_imbalance_top20"))
    flow_checks = []
    if cvd_delta is not None and side:
        flow_checks.append(cvd_delta >= 5 if side == "long" else cvd_delta <= -5)
    if book_imbalance is not None and side:
        flow_checks.append(book_imbalance >= 0.05 if side == "long" else book_imbalance <= -0.05)
    flow_available = bool(flow_checks)
    flow_aligned = flow_available and any(flow_checks)
    flow_opposed = bool(
        side == "long" and cvd_delta is not None and cvd_delta <= -10
        and book_imbalance is not None and book_imbalance <= -0.15
    ) or bool(
        side == "short" and cvd_delta is not None and cvd_delta >= 10
        and book_imbalance is not None and book_imbalance >= 0.15
    )
    groups["flow"] = {
        "available": flow_available, "aligned": bool(flow_aligned),
        "score": 20 if flow_aligned else 0, "weight": score_weights["flow"],
        "details": "دلتا معاملات تجمیعی اخیر و عدم‌تعادل دفتر سفارش لحظه‌ای؛ نه جریان سفارش نهادی."
    }
    if not flow_available and market == "crypto":
        missing.append("دلتا معاملات/دفتر سفارش")
    if flow_opposed:
        cautions.append("دلتا معاملات و دفتر سفارش هر دو به‌شدت خلاف جهت سیگنال‌اند.")

    # 5) Derivatives positioning. OI level alone is not treated as directional.
    funding = _number(derivatives.get("funding_rate"))
    taker_ratio = _number(derivatives.get("taker_buy_sell_ratio"))
    oi_change = _number(derivatives.get("open_interest_change_pct_5m"))
    derivative_checks = []
    if funding is not None and side:
        # Crowded positioning is a contrarian risk hint, not a standalone signal.
        if side == "long" and funding <= -0.0003:
            derivative_checks.append(True)
        elif side == "short" and funding >= 0.0003:
            derivative_checks.append(True)
        elif (side == "long" and funding >= 0.001) or (side == "short" and funding <= -0.001):
            cautions.append("Funding به‌شدت در جهت معامله شلوغ است؛ ریسک ازدحام موقعیت بالاست.")
    if taker_ratio is not None and side:
        derivative_checks.append(taker_ratio >= 1.05 if side == "long" else taker_ratio <= 0.95)
    if oi_change is not None and abs(oi_change) >= 0.5:
        if oi_change < -3:
            cautions.append("Open Interest در نمونه‌های اخیر به‌شدت کاهش یافته؛ حرکت ممکن است ناشی از بستن موقعیت‌ها باشد.")
        elif oi_change > 3:
            cautions.append("Open Interest سریع افزایش یافته؛ احتمال ازدحام اهرمی و نوسان ناگهانی را در نظر بگیر.")
    derivative_available = funding is not None or taker_ratio is not None or oi_change is not None
    derivative_aligned = bool(derivative_checks and any(derivative_checks))
    groups["derivatives"] = {
        "available": derivative_available, "aligned": derivative_aligned,
        "score": 10 if derivative_aligned else 0, "weight": score_weights["derivatives"],
        "details": "Funding، نسبت تیکر و تغییر پنج‌دقیقه‌ای OI؛ سطح OI به‌تنهایی جهت‌دار نیست."
    }
    if market == "crypto" and not derivative_available:
        missing.append("داده مشتقات")

    # Regime chooses the strategy family; it does not force one indicator to predict.
    regime = str(adx.get("regime") or "unknown").lower()
    squeeze = bool(volatility.get("squeeze"))
    if regime == "trending":
        strategy = "trend_following_pullback"
        strategy_label = "ادامه روند و ورود در اصلاح"
    elif regime == "ranging":
        strategy = "liquidity_reversal"
        strategy_label = "برگشت از نقدینگی در بازار رنج"
    else:
        strategy = "breakout_retest"
        strategy_label = "شکست و آزمایش مجدد"

    available_groups = [name for name, item in groups.items() if item["available"]]
    aligned_groups = [name for name, item in groups.items() if item["available"] and item["aligned"]]
    available_weight = sum(groups[name]["weight"] for name in available_groups)
    earned_weight = sum(groups[name]["score"] for name in available_groups)
    confluence_score = round(100.0 * earned_weight / available_weight, 1) if available_weight else 0.0
    coverage = round(100.0 * available_weight / sum(score_weights.values()), 1)

    blocked_reasons = []
    gate_passed = False
    if side:
        required_groups = 3 if len(available_groups) >= 4 else 2
        if not structure_available or not structure_aligned:
            blocked_reasons.append("ساختار بازار جهت معامله را تأیید نمی‌کند.")
        if not momentum_aligned:
            blocked_reasons.append("تأیید مستقل مومنتوم کافی نیست.")
        if len(aligned_groups) < required_groups:
            blocked_reasons.append(
                "تعداد گروه‌های مستقل هم‌جهت کافی نیست (" +
                str(len(aligned_groups)) + "/" + str(required_groups) + ")."
            )
        if regime == "ranging" and not structure_event:
            blocked_reasons.append("بازار رنج است؛ بدون Sweep یا شکست/تغییر ساختار ورود روندی مجاز نیست.")
        if squeeze and not structure_event:
            blocked_reasons.append("بازار در فشردگی نوسان است؛ تا شکست تأییدشده ورود نکن.")
        if flow_opposed:
            blocked_reasons.append("جریان معاملات و دفتر سفارش هر دو خلاف جهت‌اند.")
        if event_risk.get("entry_blocked_by_calendar"):
            blocked_reasons.append("رویداد اقتصادی پرریسک نزدیک است.")
        gate_passed = not blocked_reasons
    else:
        blocked_reasons.append("موتور پایه جهت خرید یا فروش صادر نکرده است؛ لایه پنهان جهت مستقلی نمی‌سازد.")

    # External context is used for risk/context only. A single dominance snapshot
    # or headline is not converted into a directional trading signal.
    if event_risk.get("new_entry_caution"):
        cautions.append("رویداد اقتصادی مهم در یک ساعت آینده؛ ورود و اندازه موقعیت با احتیاط.")
    news = market_context.get("news") or {}
    macro = market_context.get("macro") or {}
    cot = market_context.get("cot") or {}
    if not news.get("available"):
        missing.append("اخبار بازار")
    if not macro.get("available"):
        missing.append("داده‌های کلان اقتصادی")
    if market in ("gold", "silver") and not cot.get("available"):
        missing.append("گزارش هفتگی COT")
    if market == "crypto" and not dominance.get("available"):
        missing.append("دامیننس کل بازار کریپتو")
    if market == "crypto" and dominance.get("available"):
        trend = dominance.get("usdt_trend") or {}
        regime = trend.get("regime")
        if regime == "defensive_pressure":
            groups["usdt_dominance"] = {
                "available": True, "aligned": side == "short",
                "score": 12 if side == "short" else 0, "weight": 15,
                "details": "روند افزایشی USDT.D در چند بازه؛ نشانه فشار دفاعی بازار، نه سیگنال مستقل ورود."
            }
            cautions.append("USDT.D در حال افزایش است؛ فشار دفاعی/ریسک‌گریزی محتمل است، اما باید با ارزش کل بازار و نمودار خود دارایی تأیید شود.")
        elif regime == "risk_on_potential":
            groups["usdt_dominance"] = {
                "available": True, "aligned": side == "long",
                "score": 12 if side == "long" else 0, "weight": 15,
                "details": "روند کاهشی USDT.D در چند بازه؛ احتمال بهبود ریسک‌پذیری، نه اثبات ورود پول."
            }
            cautions.append("USDT.D در حال کاهش است؛ احتمال بهبود ریسک‌پذیری وجود دارد، اما کاهش دامیننس به‌تنهایی ورود پول را ثابت نمی‌کند.")
        else:
            cautions.append("تاریخچه USDT.D هنوز برای تعیین جهت کافی نیست یا بازه‌ها با هم تعارض دارند؛ فعلاً فقط مقدار لحظه‌ای نمایش داده می‌شود.")
        if trend.get("total_market_cap_usd") and trend.get("usdt_market_cap_usd_estimate"):
            cautions.append("برای تفسیر USDT.D، ارزش کل بازار و ارزش بازار تخمینی خود تتر نیز کنار آن بررسی شده‌اند.")
    if profile.get("available"):
        cautions.append("Volume Profile از OHLCV تخمین زده شده و پروفایل واقعی تیک‌به‌تیک نیست.")
    if derivatives.get("spot_order_book_available"):
        cautions.append("دفتر سفارش یک عکس لحظه‌ای است و ممکن است سریع تغییر کند.")
    if oi_change is None and derivatives.get("metric_status", {}).get("open_interest"):
        missing.append("تغییرات تاریخی Open Interest؛ فقط مقدار فعلی دریافت شد.")

    return {
        "engine": "Adaptive Hidden Intelligence",
        "version": 1,
        "backend_only": True,
        "base_signal": base_signal,
        "final_signal": base_signal if gate_passed else "WAIT",
        "gate_passed": bool(gate_passed),
        "decision": "PASS" if gate_passed else "WAIT",
        "selected_strategy": strategy,
        "selected_strategy_label": strategy_label,
        "market_regime": regime,
        "volatility_squeeze": squeeze,
        "confluence_score": confluence_score,
        "score_is_probability": False,
        "data_coverage_percent": coverage,
        "available_evidence_groups": available_groups,
        "aligned_evidence_groups": aligned_groups,
        "evidence_groups": groups,
        "missing_data": sorted(set(missing)),
        "cautions": cautions,
        "blocked_reasons": blocked_reasons,
        "context_availability": {
            "news": bool(news.get("available")),
            "macro": bool(macro.get("available")),
            "cot": bool(cot.get("available")),
            "dominance": bool(dominance.get("available")),
            "economic_calendar": bool(event_risk.get("available"))
        },
        "derivatives_snapshot": {
            "funding_rate": funding,
            "open_interest_contracts": _number(derivatives.get("open_interest_contracts")),
            "open_interest_change_pct_5m": oi_change,
            "taker_buy_sell_ratio": taker_ratio,
            "cvd_delta_percent": cvd_delta,
            "spot_orderbook_imbalance_top20": book_imbalance
        },
        "method_note": "امتیاز فقط میزان هم‌جهتی داده‌های موجود است، نه احتمال برد. داده ناموجود امتیاز مثبت یا منفی نمی‌گیرد."
    }


def apply_intelligence_gate(analysis, assessment):
    """Apply a conservative WAIT gate without drawing any extra chart layers."""
    analysis["hidden_intelligence"] = assessment
    if analysis.get("signal") not in ("BUY", "SELL") or assessment.get("gate_passed"):
        return analysis

    reasons = assessment.get("blocked_reasons") or ["تأیید مستقل کافی نیست."]
    analysis["signal"] = "WAIT"
    analysis["entry_price"] = None
    analysis["stop_loss"] = None
    analysis["take_profit_1"] = None
    analysis["take_profit_2"] = None
    analysis["take_profit_3"] = None
    analysis["trade_plan"] = {
        "entry": None, "stop_loss": None, "targets": [], "risk_reward": [],
        "entry_basis": "موتور چندمنبعی تأیید کافی نداده است."
    }
    levels = analysis.get("chart_levels") or {}
    levels.update({"entry": None, "stop_loss": None, "targets": []})
    analysis["chart_levels"] = levels

    strategy = analysis.setdefault("strategy_plan", {})
    strategy["passed"] = False
    strategy["decision"] = "عدم ورود؛ موتور چندمنبعی تأیید کافی نداده است."
    strategy["intelligence_gate_passed"] = False

    risk = analysis.setdefault("risk_management", {})
    risk["risk_gate_passed"] = False
    risk.setdefault("risk_flags", []).extend(
        ["BLOCK: " + reason for reason in reasons]
    )

    smart = analysis.get("smart_entry") or {}
    smart.update({
        "preferred_side": "neutral", "primary_signal": "WAIT",
        "risk_gate_passed": False, "intelligence_gate_passed": False
    })
    smart.setdefault("risk_flags", []).extend(
        ["BLOCK: " + reason for reason in reasons]
    )
    for side in ("long", "short"):
        plan = smart.get(side)
        if isinstance(plan, dict) and plan.get("available"):
            plan["status"] = "BLOCKED_BY_INTELLIGENCE_ENGINE"
            plan["risk_gate_passed"] = False
    analysis["smart_entry"] = smart

    analysis.setdefault("wait_reason", []).extend(reasons)
    analysis.setdefault("analysis", []).append(
        "موتور چندمنبعی: " + "؛ ".join(reasons)
    )
    scenario = analysis.get("scenario_analysis") or {}
    scenario["signal"] = "فعلاً ورود انجام نشود؛ تأیید چندمنبعی کافی نیست."
    analysis["scenario_analysis"] = scenario
    return analysis
