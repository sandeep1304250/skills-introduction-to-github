from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum

import pandas as pd


class Signal(IntEnum):
    SELL = -1
    HOLD = 0
    BUY = 1


class Strategy(ABC):
    """A strategy maps price history to a target signal per bar.

    `generate_signals` must only use information available up to each bar
    (no look-ahead); the engine acts on a bar's signal at the *next* bar's open.
    """

    name = "base"

    @property
    def warmup(self) -> int:
        """Bars of history needed before signals are meaningful."""
        return 0

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series of Signal values aligned to data.index."""

    def latest_signal(self, data: pd.DataFrame) -> Signal:
        if len(data) < self.warmup:
            return Signal.HOLD
        return Signal(int(self.generate_signals(data).iloc[-1]))
