# Smart Money AI

A research dashboard for crypto, gold and silver. The application is a decision-support prototype; it is not a broker, execution engine or profit guarantee.

## Current analysis layers

- Multi-timeframe technical analysis: 5m, 15m, 1h, 4h, 1d and 1w.
- Trend and momentum: EMA, Wilder RSI, MACD histogram, ADX/+DI/-DI, Stochastic, ATR and Bollinger squeeze.
- Structure and liquidity heuristics: swing highs/lows, BOS, CHOCH, liquidity sweeps, equal highs/lows, FVG, displacement-validated approximate order blocks, premium/discount and Fibonacci retracements.
- Volume and price references: relative volume, UTC-day VWAP, approximate OHLCV volume profile (POC/VAH/VAL), UTC candle-date levels.
- Crypto derivatives context: public funding, open interest, global long/short account ratio, taker buy/sell ratio and a top-20 spot order-book snapshot.
- Macro context: FRED yields, real yields, inflation, jobs, broad trade-weighted USD, VIX, oil and credit spreads.
- Precious-metals positioning: CFTC Disaggregated Futures-Only managed-money COT for gold and silver.
- Cross-asset rolling correlations: BTC, ETH, gold and silver daily returns.
- Research backtest: production `analyze()` signals and trade plans replayed chronologically, with next-bar fills, stop/target simulation, fees/slippage, doubled-cost stress, walk-forward windows and Monte Carlo resampling. Historical replay cannot reconstruct live quotes or future economic-calendar state.
- Risk gates for extreme ATR, weak trend without structure, excessive stop distance, and high-impact US calendar events within 15 minutes. Smart Entry excludes the zone-formation candle from retests and requires a fresh quote still inside the zone.

## Data sources and limitations

- Crypto prices and candles: Binance Spot public market data. Historical candles are paginated (default 1,500, configurable up to 3,000 with `HISTORICAL_CANDLE_LIMIT`). Live quotes are separate from closed candles.
- Scanner: ranks eligible Binance Spot USDT pairs by 24-hour quote volume, applies a default 5m USDT quote-volume floor, excludes common leveraged-token suffixes, then analyzes up to the requested limit. It does not claim to deeply analyze every listed pair in one request.
- Gold/silver prices and candles: Twelve Data spot where the account tier permits, with Yahoo spot fallback. Spot and futures are not silently mixed. If spot is unavailable, the analysis reports unavailable; COMEX futures proxies require explicit `ALLOW_METAL_FUTURES_PROXY=true`. Provider feeds may differ from TradingView.
- Crypto derivatives: Binance public USD-M Futures data plus a spot order-book snapshot. The order book is not historical institutional order flow.
- Macro: FRED. Some releases are monthly or delayed. The broad trade-weighted USD index is not the ICE DXY index.
- COT: CFTC weekly positioning. Positions are reported for Tuesday and are generally published Friday; they are not intraday signals.
- News: GDELT headlines with keyword labels. Labels are not a causal price-impact model.
- Volume profile and SMC zones are approximations from OHLCV bars, not tick-level exchange data.
- Backtest results are hypothetical and use assumed costs. The API replays the production technical signal engine on up to 500 recent closed bars by default (`BACKTEST_CANDLE_LIMIT`, max 1,000). Live quote confirmation and forward-looking economic-calendar state cannot be reconstructed historically. Results do not prove future profitability.
- WAIT is a valid output. Scores are confluence scores, not probabilities or win rates.

## Environment variables

| Variable | Purpose |
| --- | --- |
| FRED_API_KEY | Optional key for macro series |
| OPENAI_API_KEY or AI_API_KEY | Optional AI synthesis; API usage may be billed |
| AI_MODEL | Optional OpenAI Responses API model ID; default is gemini-2.5-flash (OpenAI fallback uses gpt-4.1-mini) |
| GEMINI_API_KEY | Google AI Studio key for Gemini free-tier primary AI decision engine and synthesis |\n| GEMINI_MODEL | Optional Gemini model ID; default `gemini-2.5-flash` |
| HISTORICAL_CANDLE_LIMIT | Binance Spot candle history target; default 1,500, max 3,000 |
| BACKTEST_CANDLE_LIMIT | Production-engine backtest replay bars; default 500, max 1,000 |
| SCANNER_MIN_QUOTE_VOLUME_USDT | Scanner minimum 24h quote volume; default 5,000,000 USDT |
| MAX_ENTRY_SPREAD_BPS | Maximum allowed observable Binance Spot spread for a new entry; default 20 bps |
| ALLOW_METAL_FUTURES_PROXY | Explicit opt-in for COMEX futures proxy when spot unavailable; default false |
| COINGECKO_API_KEY | Optional CoinGecko API key |
| WEBHOOK_TOKEN | Optional authenticated alert webhook |
| DATABASE_PATH | Optional persistent SQLite path |

Set secrets in Render Environment Variables. Never commit API keys or paste them into chat. Exchange keys are not currently used for live trading.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Useful endpoints

- /health
- /api/status
- /api/analyze?market=crypto&symbol=BTCUSDT&timeframe=1h
- /api/multi-timeframe?market=crypto&symbol=BTCUSDT
- /api/backtest?market=crypto&symbol=BTCUSDT&timeframe=1h
- /api/scanner?market=all&timeframe=15m&limit=20
- /api/correlations
- /api/cot?asset=gold
- /api/market-context?asset=gold
- /api/ai-synthesis (POST) — legacy text synthesis\n- /api/ai-decision (POST) — Gemini primary decision engine; gathers the selected analysis, all six timeframes, macro/news/COT, crypto dominance, cross-asset correlations and economic calendar, then validates any directional trade plan before returning it.

## Tests

```bash
python -m compileall -q app
python -m unittest discover -s tests -v
```

The Gemini primary decision engine is analysis-only and does not create paper or live trades. It requires `GEMINI_API_KEY`; free-tier quotas are project-specific and can be exhausted. It does not silently fall back to a paid provider.\n\nLive trading is disabled. No real orders are placed.
