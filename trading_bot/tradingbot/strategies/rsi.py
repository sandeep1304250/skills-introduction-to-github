from __future__ import annotations

import pandas as pd

from .base import Signal, Strategy


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


class RSIMeanReversion(Strategy):
    """Buy when RSI crosses up through `oversold`, sell when it crosses down through `overbought`."""

    name = "rsi"

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        if not 0 < oversold < overbought < 100:
            raise ValueError("need 0 < oversold < overbought < 100")
        self.period, self.oversold, self.overbought = int(period), float(oversold), float(overbought)

    @property
    def warmup(self) -> int:
        return self.period + 2

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        r = rsi(data["close"], self.period)
        prev = r.shift(1)
        signals = pd.Series(Signal.HOLD, index=data.index, dtype=int)
        signals[(prev < self.oversold) & (r >= self.oversold)] = Signal.BUY
        signals[(prev > self.overbought) & (r <= self.overbought)] = Signal.SELL
        return signals
