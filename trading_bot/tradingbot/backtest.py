"""Event-driven backtester. Signals computed on a bar's close are executed at the next bar's open."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .broker import Fill, PaperBroker
from .risk import RiskManager
from .strategies import Signal, Strategy


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    reason: str

    @property
    def return_pct(self) -> float:
        return self.exit_price / self.entry_price - 1


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    fills: list[Fill]
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        width = max(len(k) for k in self.metrics)
        lines = []
        for k, v in self.metrics.items():
            if isinstance(v, float):
                v = f"{v:,.2%}" if k.endswith("%") else f"{v:,.4f}"
            lines.append(f"{k:<{width}}  {v}")
        return "\n".join(lines)


def periods_per_year(index: pd.Index) -> float:
    if len(index) < 2:
        return 252.0
    step = pd.Series(index).diff().median()
    return pd.Timedelta(days=365) / step if step and step > pd.Timedelta(0) else 252.0


def compute_metrics(equity: pd.Series, trades: list[Trade], starting_cash: float) -> dict:
    returns = equity.pct_change().dropna()
    ppy = periods_per_year(equity.index)
    years = max(len(equity) / ppy, 1e-9)
    final = float(equity.iloc[-1])
    total_return = final / starting_cash - 1
    cagr = (final / starting_cash) ** (1 / years) - 1 if final > 0 else -1.0
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(ppy)) if std and std > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_loss = -sum(t.pnl for t in losses)
    return {
        "Starting equity": float(starting_cash),
        "Final equity": final,
        "Total return %": total_return,
        "CAGR %": float(cagr),
        "Sharpe": sharpe,
        "Max drawdown %": float(drawdown.min()),
        "Trades": len(trades),
        "Win rate %": len(wins) / len(trades) if trades else 0.0,
        "Profit factor": (sum(t.pnl for t in wins) / gross_loss) if gross_loss > 0 else float("inf") if wins else 0.0,
    }


def run_backtest(data: pd.DataFrame, strategy: Strategy, risk: RiskManager | None = None,
                 starting_cash: float = 10_000.0, fee_rate: float = 0.001, slippage: float = 0.0005) -> BacktestResult:
    risk = risk or RiskManager()
    broker = PaperBroker(starting_cash, fee_rate, slippage)
    signals = strategy.generate_signals(data)

    trades: list[Trade] = []
    equity_curve = []
    entry_price = entry_time = entry_cost = None
    peak = starting_cash
    pending = Signal.HOLD

    def close_position(price, time, reason):
        nonlocal entry_price, entry_time, entry_cost
        fill = broker.sell(broker.position, price, time, reason)
        if fill:
            pnl = fill.qty * fill.price - fill.fee - entry_cost
            trades.append(Trade(entry_time, time, entry_price, fill.price, fill.qty, pnl, reason))
        entry_price = entry_time = entry_cost = None

    for time, bar in data.iterrows():
        # 1) Execute yesterday's signal at today's open.
        if pending == Signal.BUY and broker.position == 0 and not risk.halted(broker.equity(bar.open), peak):
            qty = risk.position_size(broker.equity(bar.open), broker.cash, bar.open)
            fill = broker.buy(qty, bar.open, time, "signal")
            if fill:
                entry_price, entry_time, entry_cost = fill.price, time, fill.qty * fill.price + fill.fee
        elif pending == Signal.SELL and broker.position > 0:
            close_position(bar.open, time, "signal")

        # 2) Protective exits within the bar.
        if broker.position > 0:
            hit = risk.exit_reason(entry_price, bar.open, bar.low, bar.high)
            if hit:
                close_position(hit[1], time, hit[0])

        # 3) Mark to market and read this bar's signal for tomorrow.
        eq = broker.equity(bar.close)
        peak = max(peak, eq)
        equity_curve.append(eq)
        pending = Signal(int(signals.loc[time]))

    # Close any open position at the final close so metrics reflect realised P&L.
    if broker.position > 0:
        close_position(data["close"].iloc[-1], data.index[-1], "end_of_data")
        equity_curve[-1] = broker.equity(data["close"].iloc[-1])

    equity = pd.Series(equity_curve, index=data.index, name="equity")
    return BacktestResult(equity, trades, broker.fills, compute_metrics(equity, trades, starting_cash))
