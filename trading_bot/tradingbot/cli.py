"""Command-line interface: `python -m tradingbot {backtest,paper,live} ...`"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from . import data as datasrc
from .backtest import run_backtest
from .bot import TradingBot
from .broker import CCXTBroker, PaperBroker
from .risk import RiskManager
from .strategies import STRATEGIES, get_strategy


def parse_params(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        if not _:
            raise SystemExit(f"--param expects key=value, got '{p}'")
        out[k] = float(v) if "." in v else int(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tradingbot", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--strategy", default="sma_cross", choices=list(STRATEGIES))
    common.add_argument("--param", action="append", metavar="KEY=VALUE", help="strategy parameter, e.g. fast=10")
    common.add_argument("--cash", type=float, default=10_000.0)
    common.add_argument("--risk-per-trade", type=float, default=0.02)
    common.add_argument("--max-position", type=float, default=0.95)
    common.add_argument("--stop-loss", type=float, default=0.05, help="fraction, 0 disables")
    common.add_argument("--take-profit", type=float, default=0.10, help="fraction, 0 disables")
    common.add_argument("--max-drawdown", type=float, default=0.25, help="fraction, 0 disables")
    common.add_argument("--fee", type=float, default=0.001)
    common.add_argument("--slippage", type=float, default=0.0005)
    common.add_argument("-v", "--verbose", action="store_true")

    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("--source", choices=["synthetic", "csv", "ccxt"], default="synthetic")
    source.add_argument("--csv", help="path to OHLCV CSV (with --source csv)")
    source.add_argument("--exchange", default="binance", help="ccxt exchange id")
    source.add_argument("--symbol", default="BTC/USDT")
    source.add_argument("--timeframe", default="1h")
    source.add_argument("--bars", type=int, default=1000)
    source.add_argument("--seed", type=int, default=42)

    bt = sub.add_parser("backtest", parents=[common, source], help="test a strategy on historical data")
    bt.add_argument("--trades", action="store_true", help="print the trade list")
    bt.add_argument("--plot", help="save an equity-curve PNG to this path (needs matplotlib)")

    for name, help_ in (("paper", "simulated trading on live or replayed data"),
                        ("live", "REAL orders on an exchange via ccxt")):
        p = sub.add_parser(name, parents=[common, source], help=help_)
        p.add_argument("--interval", type=float, default=60.0, help="seconds between polls")
        p.add_argument("--steps", type=int, help="stop after N polls")
        p.add_argument("--state-file", default=f".tradingbot_{name}_state.json")
        if name == "live":
            p.add_argument("--no-sandbox", action="store_true", help="trade on the real exchange, not its testnet")
            p.add_argument("--yes-i-understand", action="store_true", help="confirm you accept the risk of real losses")
    return ap


def make_risk(a) -> RiskManager:
    return RiskManager(a.risk_per_trade, a.max_position, a.stop_loss or None, a.take_profit or None, a.max_drawdown or None)


def load_history(a):
    if a.source == "csv":
        if not a.csv:
            raise SystemExit("--source csv needs --csv PATH")
        return datasrc.load_csv(a.csv)
    if a.source == "ccxt":
        return datasrc.fetch_ccxt(a.exchange, a.symbol, a.timeframe, a.bars)
    return datasrc.synthetic(a.bars, seed=a.seed)


def cmd_backtest(a) -> int:
    df = load_history(a)
    strategy = get_strategy(a.strategy, **parse_params(a.param))
    res = run_backtest(df, strategy, make_risk(a), a.cash, a.fee, a.slippage)
    first, last = df["close"].iloc[0], df["close"].iloc[-1]
    print(f"Strategy: {strategy.name} {vars(strategy)}  |  {len(df)} bars {df.index[0]} -> {df.index[-1]}")
    print(f"Buy & hold return: {last / first - 1:.2%}\n")
    print(res.summary())
    if a.trades:
        print("\nentry_time           exit_time            entry      exit       pnl        reason")
        for t in res.trades:
            print(f"{str(t.entry_time):20} {str(t.exit_time):20} {t.entry_price:<10.2f} {t.exit_price:<10.2f} "
                  f"{t.pnl:<10.2f} {t.reason}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        res.equity.plot(ax=ax, label="strategy")
        (df["close"] / first * a.cash).plot(ax=ax, label="buy & hold", alpha=0.6)
        ax.legend(); ax.set_ylabel("equity"); fig.tight_layout(); fig.savefig(a.plot)
        print(f"\nSaved equity curve to {a.plot}")
    return 0


def make_feed(a, strategy):
    """Returns a callable producing (closed_bars, last_price)."""
    lookback = max(strategy.warmup * 3, 200)
    if a.source == "ccxt":
        import ccxt
        ex = getattr(ccxt, a.exchange)({"enableRateLimit": True})

        def feed():
            df = datasrc.fetch_ccxt(a.exchange, a.symbol, a.timeframe, lookback + 1, exchange=ex)
            price = float(ex.fetch_ticker(a.symbol)["last"])
            return df.iloc[:-1], price  # drop the still-forming candle
        return feed

    # Replay historical/synthetic bars one at a time — useful for dry runs.
    history = load_history(a)
    state = {"i": strategy.warmup}

    def feed():
        i = state["i"]
        if i >= len(history):
            return None
        state["i"] += 1
        window = history.iloc[max(0, i - lookback):i]
        return window, float(window["close"].iloc[-1])
    return feed


def cmd_trade(a, live: bool) -> int:
    strategy = get_strategy(a.strategy, **parse_params(a.param))
    if live:
        if a.source != "ccxt":
            raise SystemExit("live trading requires --source ccxt")
        key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
        if not key or not secret:
            raise SystemExit("Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET environment variables.")
        if a.no_sandbox and not a.yes_i_understand:
            raise SystemExit("Refusing to trade real funds without --yes-i-understand.")
        broker = CCXTBroker(a.exchange, a.symbol, key, secret, sandbox=not a.no_sandbox)
    else:
        broker = PaperBroker(a.cash, a.fee, a.slippage)
    bot = TradingBot(strategy, broker, make_risk(a), a.state_file)
    feed = make_feed(a, strategy)
    interval = a.interval if a.source == "ccxt" else 0.0
    bot.run(feed, interval=interval, max_steps=a.steps)
    fills = broker.fills
    print(f"\nDone. {len(fills)} fills. Cash={broker.cash:.2f} position={broker.position:.6f}")
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "backtest":
        return cmd_backtest(a)
    return cmd_trade(a, live=a.cmd == "live")


if __name__ == "__main__":
    sys.exit(main())
