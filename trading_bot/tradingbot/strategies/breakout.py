from __future__ import annotations

import pandas as pd

from .base import Signal, Strategy


class Breakout(Strategy):
    """Donchian-channel breakout.

    Buy when the close breaks above the highest high of the previous `entry` bars;
    sell when it breaks below the lowest low of the previous `exit` bars.
    A shorter exit window than the entry window lets winners run but cuts losers sooner.
    """

    name = "breakout"

    def __init__(self, entry: int = 20, exit: int = 10):
        if entry < 2 or exit < 2:
            raise ValueError("entry and exit windows must be at least 2 bars")
        self.entry, self.exit = int(entry), int(exit)

    @property
    def warmup(self) -> int:
        return max(self.entry, self.exit) + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        # shift(1): compare against the channel *before* this bar, so the bar can't break its own high.
        upper = data["high"].rolling(self.entry).max().shift(1)
        lower = data["low"].rolling(self.exit).min().shift(1)
        signals = pd.Series(Signal.HOLD, index=data.index, dtype=int)
        signals[close > upper] = Signal.BUY
        signals[close < lower] = Signal.SELL
        return signals
