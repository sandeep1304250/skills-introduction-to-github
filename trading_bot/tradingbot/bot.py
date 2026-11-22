"""The trading loop: pull data, check exits, act on the latest closed-bar signal."""
from __future__ import annotations

import json
import logging
import os
import time as _time
from typing import Callable

import pandas as pd

from .broker import Broker
from .risk import RiskManager
from .strategies import Signal, Strategy

log = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, strategy: Strategy, broker: Broker, risk: RiskManager | None = None,
                 state_file: str | None = None):
        self.strategy = strategy
        self.broker = broker
        self.risk = risk or RiskManager()
        self.state_file = state_file
        self.entry_price: float | None = None
        self.peak_equity: float | None = None
        self.last_bar = None
        self._load_state()

    # --- persistence so a restart keeps stop-loss levels ---
    def _load_state(self):
        if self.state_file and os.path.exists(self.state_file):
            with open(self.state_file) as f:
                s = json.load(f)
            self.entry_price, self.peak_equity = s.get("entry_price"), s.get("peak_equity")
            self.last_bar = s.get("last_bar")

    def _save_state(self):
        if self.state_file:
            with open(self.state_file, "w") as f:
                json.dump({"entry_price": self.entry_price, "peak_equity": self.peak_equity,
                           "last_bar": self.last_bar}, f)

    def step(self, data: pd.DataFrame, price: float | None = None) -> str:
        """Process one update. `data` must contain only closed bars; `price` is the
        latest traded price (defaults to the last close). Returns the action taken."""
        if data.empty:
            return "no_data"
        price = float(price if price is not None else data["close"].iloc[-1])
        bar_time = str(data.index[-1])
        equity = self.broker.equity(price)
        self.peak_equity = max(self.peak_equity or equity, equity)
        action = "hold"

        position = self.broker.position
        if position > 0 and self.entry_price:
            hit = self.risk.exit_reason(self.entry_price, price, price, price)
            if hit:
                self.broker.sell(position, price, bar_time, hit[0])
                self.entry_price = None
                action = hit[0]

        # Act on each closed bar's signal once.
        if action == "hold" and bar_time != self.last_bar:
            self.last_bar = bar_time
            signal = self.strategy.latest_signal(data)
            if signal == Signal.BUY and self.broker.position == 0:
                if self.risk.halted(equity, self.peak_equity):
                    action = "halted"
                    log.warning("Max drawdown reached; not opening new positions.")
                else:
                    qty = self.risk.position_size(equity, self.broker.cash, price)
                    fill = self.broker.buy(qty, price, bar_time, "signal")
                    if fill:
                        self.entry_price = fill.price
                        action = "buy"
            elif signal == Signal.SELL and self.broker.position > 0:
                self.broker.sell(self.broker.position, price, bar_time, "signal")
                self.entry_price = None
                action = "sell"

        log.info("%s price=%.4f action=%s position=%.6f equity=%.2f",
                 bar_time, price, action, self.broker.position, self.broker.equity(price))
        self._save_state()
        return action

    def run(self, feed: Callable[[], tuple[pd.DataFrame, float] | None], interval: float = 60.0,
            max_steps: int | None = None):
        """Poll `feed` forever (or max_steps times). feed returns (closed_bars, last_price) or None to stop."""
        steps = 0
        try:
            while max_steps is None or steps < max_steps:
                got = feed()
                if got is None:
                    break
                try:
                    self.step(*got)
                except Exception:  # keep the loop alive on transient errors
                    log.exception("step failed")
                steps += 1
                if interval > 0:
                    _time.sleep(interval)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
