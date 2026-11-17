"""Conservative OHLCV research backtest with chronological holdout and execution costs."""
import random

from .pro_analysis import _adx, _ema_series, _rsi_wilder, _true_ranges


def _metrics(trades, equity_curve, initial_equity):
    ending = equity_curve[-1] if equity_curve else initial_equity
    returns = [t["net_pnl"] for t in trades]
    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    peak = initial_equity
    max_dd = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    r_values = [t["r_multiple"] for t in trades]
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_percent": round(len(wins) / len(trades) * 100.0, 2) if trades else None,
        "net_pnl": round(ending - initial_equity, 2),
        "return_percent": round((ending / initial_equity - 1) * 100.0, 2) if initial_equity else 0.0,
        "ending_equity": round(ending, 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else (None if not wins else "no_losses_in_sample"),
        "expectancy_r": round(sum(r_values) / len(r_values), 3) if r_values else None,
        "average_win": round(sum(wins) / len(wins), 2) if wins else None,
        "average_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "max_drawdown_percent": round(max_dd, 2),
        "trade_log": trades[-20:],
        "r_multiples": r_values
    }


def _simulate(candles, start_index, end_index, initial_equity=10000.0,
              risk_fraction=0.005, fee_rate=0.0004, slippage_rate=0.0002,
              reward_risk=2.0, max_holding_bars=50):
    equity = initial_equity
    equity_curve = [equity]
    trades = []
    i = max(60, start_index)
    end_index = min(end_index, len(candles))
    closes = [c["close"] for c in candles]

    while i < end_index - 1:
        history = candles[:i + 1]
        hist_closes = closes[:i + 1]
        ema9 = _ema_series(hist_closes, 9)
        ema21 = _ema_series(hist_closes, 21)
        rsi = _rsi_wilder(hist_closes, 14)
        adx, plus_di, minus_di = _adx(history, 14)
        trs = _true_ranges(history)
        atr = sum(trs[-14:]) / min(14, len(trs))
        if len(ema9) < 2 or len(ema21) < 2 or atr <= 0:
            i += 1
            continue

        cross_up = ema9[-1] > ema21[-1] and ema9[-2] <= ema21[-2]
        cross_down = ema9[-1] < ema21[-1] and ema9[-2] >= ema21[-2]
        side = 1 if cross_up and rsi >= 50 and adx >= 18 and plus_di > minus_di else (
            -1 if cross_down and rsi <= 50 and adx >= 18 and minus_di > plus_di else 0
        )
        if side == 0:
            i += 1
            continue

        entry_bar = i + 1
        raw_entry = candles[entry_bar]["open"]
        entry = raw_entry * (1 + slippage_rate if side == 1 else 1 - slippage_rate)
        stop_distance = 1.5 * atr
        stop = entry - side * stop_distance
        target = entry + side * stop_distance * reward_risk
        risk_cash = equity * risk_fraction
        quantity = min(risk_cash / stop_distance, equity / max(entry, 1e-12))
        if quantity <= 0:
            i += 1
            continue

        exit_index = min(end_index - 1, entry_bar + max_holding_bars)
        exit_price = candles[exit_index]["close"]
        exit_reason = "time_exit"
        for j in range(entry_bar, min(end_index, entry_bar + max_holding_bars + 1)):
            bar = candles[j]
            if side == 1:
                stop_hit = bar["low"] <= stop
                target_hit = bar["high"] >= target
            else:
                stop_hit = bar["high"] >= stop
                target_hit = bar["low"] <= target
            # Conservative assumption: if both levels trade in one OHLC bar, stop is first.
            if stop_hit:
                exit_index, exit_price, exit_reason = j, stop, "stop_loss"
                break
            if target_hit:
                exit_index, exit_price, exit_reason = j, target, "take_profit"
                break

        executed_exit = exit_price * (1 - slippage_rate if side == 1 else 1 + slippage_rate)
        gross_pnl = side * (executed_exit - entry) * quantity
        fees = (abs(entry * quantity) + abs(executed_exit * quantity)) * fee_rate
        net_pnl = gross_pnl - fees
        equity += net_pnl
        r_multiple = net_pnl / risk_cash if risk_cash > 0 else 0.0
        trades.append({
            "side": "LONG" if side == 1 else "SHORT",
            "entry_time": candles[entry_bar]["time"],
            "exit_time": candles[exit_index]["time"],
            "entry": round(entry, 8), "exit": round(executed_exit, 8),
            "stop": round(stop, 8), "target": round(target, 8),
            "quantity": round(quantity, 8), "net_pnl": round(net_pnl, 4),
            "r_multiple": round(r_multiple, 3), "exit_reason": exit_reason
        })
        equity_curve.append(equity)
        i = exit_index + 1

    return _metrics(trades, equity_curve, initial_equity)


def _monte_carlo(r_multiples, paths=500, risk_fraction=0.005, initial_equity=10000.0):
    if len(r_multiples) < 5:
        return {"available": False, "message": "برای شبیه‌سازی مونت‌کارلو حداقل ۵ معامله خارج از نمونه لازم است.", "trades": len(r_multiples)}
    rng = random.Random(42)
    final_returns, drawdowns = [], []
    for _ in range(paths):
        sample = [rng.choice(r_multiples) for _ in range(len(r_multiples))]
        equity = peak = initial_equity
        max_dd = 0.0
        for r_value in sample:
            equity *= max(0.01, 1.0 + r_value * risk_fraction)
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak * 100.0)
        final_returns.append((equity / initial_equity - 1.0) * 100.0)
        drawdowns.append(max_dd)
    final_returns.sort()
    drawdowns.sort()
    q05 = final_returns[int((len(final_returns) - 1) * 0.05)]
    q50 = final_returns[int((len(final_returns) - 1) * 0.50)]
    q95 = final_returns[int((len(final_returns) - 1) * 0.95)]
    dd95 = drawdowns[int((len(drawdowns) - 1) * 0.95)]
    return {
        "available": True, "paths": paths, "trades_per_path": len(r_multiples),
        "return_percent_p05": round(q05, 2),
        "return_percent_median": round(q50, 2),
        "return_percent_p95": round(q95, 2),
        "probability_of_loss_percent": round(sum(1 for x in final_returns if x < 0) / paths * 100.0, 1),
        "max_drawdown_percent_p95": round(dd95, 2),
        "method": "bootstrap resampling of out-of-sample trade R-multiples with replacement; fixed 0.5% risk"
    }


def _walk_forward(candles, folds=3):
    warmup = 60
    usable = len(candles) - warmup
    if usable < folds * 10:
        return {"available": False, "message": "داده کافی برای پنجره‌های walk-forward وجود ندارد."}
    fold_size = usable // folds
    results = []
    for index in range(folds):
        start = warmup + index * fold_size
        end = len(candles) if index == folds - 1 else warmup + (index + 1) * fold_size
        stats = _simulate(candles, start, end)
        results.append({"fold": index + 1, "start_index": start, "end_index": end, "metrics": stats})
    return {"available": True, "method": "three sequential fixed-rule out-of-sample windows", "folds": results}


def run_research_backtest(candles, timeframe):
    if len(candles) < 100:
        return {
            "available": False,
            "message": "برای بک‌تست قابل بررسی حداقل ۱۰۰ کندل بسته‌شده لازم است.",
            "candles": len(candles)
        }
    split = max(70, int(len(candles) * 0.70))
    if split >= len(candles) - 10:
        return {"available": False, "message": "داده کافی برای بخش خارج از نمونه وجود ندارد."}
    in_sample = _simulate(candles, 60, split)
    out_sample = _simulate(candles, split, len(candles))
    stressed_out_sample = _simulate(
        candles, split, len(candles), fee_rate=0.0008, slippage_rate=0.0004
    )
    monte_carlo = _monte_carlo(out_sample.get("r_multiples", []))
    walk_forward = _walk_forward(candles)
    return {
        "available": True,
        "strategy": "EMA 9/21 crossover + Wilder RSI + ADX directional filter",
        "timeframe": timeframe,
        "candles": len(candles),
        "split": {"in_sample_percent": 70, "out_of_sample_percent": 30, "chronological": True},
        "in_sample": in_sample,
        "out_of_sample": out_sample,
        "out_of_sample_2x_cost_stress": stressed_out_sample,
        "walk_forward": walk_forward,
        "monte_carlo": monte_carlo,
        "sample_warning": "اگر تعداد معاملات خارج از نمونه کمتر از ۳۰ باشد، معیارها بسیار ناپایدارند." if out_sample["trades"] < 30 else None,
        "assumptions": {
            "starting_equity_per_period": 10000,
            "risk_per_trade_percent": 0.5,
            "fee_per_side_percent": fee_rate_to_percent(0.0004),
            "slippage_per_side_percent": fee_rate_to_percent(0.0002),
            "reward_risk": 2.0,
            "max_holding_bars": 50,
            "leverage": 1.0,
            "same_bar_stop_and_target": "stop loss assumed first",
            "entry_timing": "next candle open after signal candle closes"
        },
        "limitations": [
            "این بک‌تست یک استراتژی پایه EMA/RSI/ADX را می‌سنجد، نه تمام منطق سیگنال اصلی برنامه.",
            "داده OHLCV برای بازسازی ترتیب دقیق معاملات داخل کندل کافی نیست؛ برخورد همزمان حدضرر و هدف با فرض محافظه‌کارانه حدضرر حل می‌شود.",
            "کارمزد و اسلیپیج فرضی‌اند؛ هزینه واقعی صرافی، اسپرد، funding، هزینه قرض‌گیری شورت و اثر بازار ممکن است بیشتر باشد.",
            "نمونه تاریخی کوتاه و تعداد معامله کم، نتیجه آماری قابل اتکا نمی‌سازد؛ این گزارش تضمین عملکرد آینده نیست."
        ]
    }


def fee_rate_to_percent(rate):
    return round(rate * 100.0, 4)
