from __future__ import annotations

import pandas as pd

from .base import Signal, Strategy


class SMACrossover(Strategy):
    """Buy when the fast SMA crosses above the slow SMA, sell on the cross below."""

    name = "sma_cross"

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast, self.slow = int(fast), int(slow)

    @property
    def warmup(self) -> int:
        return self.slow + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        fast = close.rolling(self.fast).mean()
        slow = close.rolling(self.slow).mean()
        above = (fast > slow).astype(int)
        cross = above.diff().fillna(0)
        signals = pd.Series(Signal.HOLD, index=data.index, dtype=int)
        signals[cross > 0] = Signal.BUY
        signals[cross < 0] = Signal.SELL
        signals[slow.isna()] = Signal.HOLD
        return signals
