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
- Research backtest: EMA 9/21 + Wilder RSI + ADX baseline, chronological 70/30 split, three walk-forward windows, Monte Carlo trade resampling, next-bar entry, fees/slippage and doubled-cost stress.
- Conservative risk gates for extreme ATR, very weak trend without structure, and excessive stop distance.

## Data sources and limitations

- Crypto prices and candles: Binance Spot public market data. The symbol list is loaded from Binance exchange metadata and includes all currently trading USDT-quoted spot pairs; the list is cached for one hour. Live quotes are separate from closed candles.
- Gold/silver prices and candles: Binance USDⓈ-M TradFi perpetual contracts XAUUSDT and XAGUSDT. These are USDT-settled derivatives, not physical spot metal, and can differ from TradingView feeds such as CAPITALCOM:XAUUSD or CAPITALCOM:XAGUSD. No Yahoo/Twelve Data fallback is used for these markets.
- Crypto derivatives: Binance public USD-M Futures data plus a spot order-book snapshot. The order book is not historical institutional order flow.
- Macro: FRED. Some releases are monthly or delayed. The broad trade-weighted USD index is not the ICE DXY index.
- COT: CFTC weekly positioning. Positions are reported for Tuesday and are generally published Friday; they are not intraday signals.
- News: GDELT headlines with keyword labels. Labels are not a causal price-impact model.
- Volume profile and SMC zones are approximations from OHLCV bars, not tick-level exchange data.
- Backtest results are hypothetical, use assumed costs, and test a baseline strategy rather than every rule in the live signal engine. They do not prove future profitability.
- WAIT is a valid output. Scores are confluence scores, not probabilities or win rates.

## Environment variables

| Variable | Purpose |
| --- | --- |
| FRED_API_KEY | Optional key for macro series |
| OPENAI_API_KEY or AI_API_KEY | Optional AI synthesis; API usage may be billed |
| AI_MODEL | Optional Responses API model ID; default is gpt-6-luna |
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
- /api/ai-synthesis (POST)

## Tests

```bash
python -m compileall -q app
python -m unittest discover -s tests -v
```

Live trading is disabled. No real orders are placed.
