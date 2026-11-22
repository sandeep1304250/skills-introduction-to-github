"""Position sizing and protective exits."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskManager:
    """
    risk_per_trade: fraction of equity you're willing to lose if the stop is hit.
    max_position_pct: cap on position value as a fraction of equity.
    stop_loss_pct / take_profit_pct: exit thresholds relative to entry (None disables).
    max_drawdown_pct: halt new entries once equity falls this far below its peak.
    """

    risk_per_trade: float = 0.01
    max_position_pct: float = 0.95
    stop_loss_pct: float | None = 0.05
    take_profit_pct: float | None = 0.10
    max_drawdown_pct: float | None = 0.25

    def __post_init__(self):
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1]")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")

    def position_size(self, equity: float, cash: float, price: float) -> float:
        """Units to buy so a stop-out loses ~risk_per_trade of equity, within caps."""
        if price <= 0 or equity <= 0:
            return 0.0
        max_value = min(equity * self.max_position_pct, cash)
        if self.stop_loss_pct:
            value = equity * self.risk_per_trade / self.stop_loss_pct
        else:
            value = equity * self.risk_per_trade
        return max(0.0, min(value, max_value)) / price

    def exit_reason(self, entry_price: float, open_: float, low: float, high: float) -> tuple[str, float] | None:
        """Return (reason, exit_price) if this bar's range hits the stop or target.

        The stop is checked first: when both are hit within one bar we assume the worst.
        If price gaps through a level at the open, the fill is the open price."""
        if self.stop_loss_pct:
            stop = entry_price * (1 - self.stop_loss_pct)
            if low <= stop:
                return "stop_loss", min(stop, open_)
        if self.take_profit_pct:
            target = entry_price * (1 + self.take_profit_pct)
            if high >= target:
                return "take_profit", max(target, open_)
        return None

    def halted(self, equity: float, peak_equity: float) -> bool:
        if not self.max_drawdown_pct or peak_equity <= 0:
            return False
        return (peak_equity - equity) / peak_equity >= self.max_drawdown_pct
