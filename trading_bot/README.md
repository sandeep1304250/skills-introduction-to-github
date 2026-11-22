# Trading Bot

A small, modular trading bot in Python. You can backtest it, paper-trade it, and, if you choose to, connect it to a crypto exchange through [ccxt](https://github.com/ccxt/ccxt).

> ⚠️ **Disclaimer.** This is educational software, not financial advice. The bundled strategies are simple
> textbook examples with **no proven edge**. Backtests on synthetic data mean nothing about real markets.
> Never trade money you can't afford to lose.

## Features

- **Strategies**: SMA crossover (`sma_cross`), RSI mean reversion (`rsi`) and Donchian breakout (`breakout`). Add your own by subclassing `Strategy`.
- **Risk management**: sizes each position so that a stop-out loses a fixed share of equity, caps position size, sets stop-loss and take-profit levels, and stops opening new positions after a set drawdown.
- **Backtester**: signals fire on a bar's close and fill at the next bar's open, so there is no look-ahead. It models fees and slippage, fills at the open when price gaps through a stop, and reports total return, CAGR, Sharpe, max drawdown, win rate and profit factor.
- **Paper trading**: replays CSV or synthetic data, or reads live exchange prices while simulating the fills.
- **Live trading** (opt-in): market orders through ccxt. It uses the exchange testnet by default and needs explicit flags before it touches real funds.
- **Restart-safe**: entry price and drawdown peak are saved to a JSON state file.

## Quick start

```bash
cd trading_bot
pip install -r requirements.txt

# Backtest on synthetic data
python -m tradingbot backtest
python -m tradingbot backtest --strategy rsi --param period=14 --param oversold=25 --trades
python -m tradingbot backtest --strategy breakout --param entry=20 --param exit=10

# Backtest on your own data (CSV columns: date/timestamp, open, high, low, close, volume)
python -m tradingbot backtest --source csv --csv prices.csv --plot equity.png

# Backtest on real exchange candles (pip install ccxt)
python -m tradingbot backtest --source ccxt --exchange binance --symbol BTC/USDT --timeframe 1h --bars 1000

# Paper trade: replay synthetic bars quickly, or follow live prices every 60 s
python -m tradingbot paper -v
python -m tradingbot paper --source ccxt --symbol ETH/USDT --timeframe 15m --interval 60 -v
```

### Practising with paper trading on live prices

Paper mode reads real prices from a public exchange feed but simulates every order, so it needs no account and no API keys.

```bash
pip install ccxt
python -m tradingbot paper --source ccxt --exchange kraken --symbol BTC/USD --timeframe 5m --interval 60 -v
```

- Short timeframes (`1m`, `5m`) produce trades sooner, because the bot acts once per closed candle.
- Press Ctrl+C to stop. Cash, position and entry price are saved to `.tradingbot_paper_state.json`, and running the same command again picks up where you left off. Delete that file to start over with fresh cash.
- Binance blocks some countries, including the US. If you get a network error, try `--exchange kraken` or `--exchange coinbase` with a `/USD` pair.

### Live trading

```bash
export EXCHANGE_API_KEY=...  EXCHANGE_API_SECRET=...
python -m tradingbot live --source ccxt --exchange binance --symbol BTC/USDT -v          # testnet
python -m tradingbot live --source ccxt ... --no-sandbox --yes-i-understand              # REAL money
```

The bot is **long-only** and trades one symbol at a time. Use API keys with trading permission only, never withdrawal permission.

## Risk options

| Flag | Default | Meaning |
|---|---|---|
| `--risk-per-trade` | 0.02 | Share of equity lost if the stop is hit |
| `--max-position` | 0.95 | Largest position, as a share of equity |
| `--stop-loss` | 0.05 | Stop distance below entry (0 turns it off) |
| `--take-profit` | 0.10 | Target distance above entry (0 turns it off) |
| `--max-drawdown` | 0.25 | Stop opening positions after this drop from peak equity (0 turns it off) |
| `--fee` / `--slippage` | 0.001 / 0.0005 | Simulated costs |

## Strategies

| Name | Buys when | Sells when | Parameters (defaults) |
|---|---|---|---|
| `sma_cross` | fast SMA crosses above slow SMA | fast crosses below slow | `fast=20 slow=50` |
| `rsi` | RSI rises back above `oversold` | RSI falls back below `overbought` | `period=14 oversold=30 overbought=70` |
| `breakout` | close beats the highest high of the last `entry` bars | close falls under the lowest low of the last `exit` bars | `entry=20 exit=10` |

## Writing a strategy

```python
from tradingbot.strategies import Signal, Strategy, STRATEGIES

class Momentum(Strategy):
    name = "momentum"
    def __init__(self, lookback=10): self.lookback = lookback
    @property
    def warmup(self): return self.lookback + 1
    def generate_signals(self, data):
        change = data["close"].pct_change(self.lookback)
        s = (change > 0.05).astype(int) * Signal.BUY
        s[change < -0.05] = Signal.SELL
        return s

STRATEGIES["momentum"] = Momentum
```

Rule: a bar's signal may use only data up to and including that bar. `tests/test_tradingbot.py` checks this for the bundled strategies.

## Layout

```
tradingbot/
  data.py        CSV / synthetic / ccxt data
  strategies/    Strategy base + SMA crossover, RSI, breakout
  risk.py        position sizing, stops, drawdown halt
  broker.py      PaperBroker (simulated), CCXTBroker (real orders)
  backtest.py    event-driven backtester + metrics
  bot.py         polling trading loop
  cli.py         command-line interface
tests/           pytest suite (`python -m pytest tests`)
```
