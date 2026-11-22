"""Brokers execute orders. PaperBroker simulates fills; CCXTBroker sends real orders."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class Fill:
    time: datetime | None
    side: str  # "buy" | "sell"
    qty: float
    price: float
    fee: float
    reason: str = ""


class Broker(ABC):
    @property
    @abstractmethod
    def cash(self) -> float: ...

    @property
    @abstractmethod
    def position(self) -> float: ...

    @abstractmethod
    def buy(self, qty: float, price: float, time=None, reason: str = "") -> Fill | None: ...

    @abstractmethod
    def sell(self, qty: float, price: float, time=None, reason: str = "") -> Fill | None: ...

    def equity(self, price: float) -> float:
        return self.cash + self.position * price


@dataclass
class PaperBroker(Broker):
    """Simulated broker with proportional fees and slippage. Long-only, no leverage."""

    starting_cash: float = 10_000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    fills: list[Fill] = field(default_factory=list)

    def __post_init__(self):
        self._cash = float(self.starting_cash)
        self._position = 0.0

    @property
    def cash(self) -> float:
        return self._cash

    def get_state(self) -> dict:
        return {"cash": self._cash, "position": self._position}

    def set_state(self, state: dict) -> None:
        self._cash = float(state["cash"])
        self._position = float(state["position"])

    @property
    def position(self) -> float:
        return self._position

    def buy(self, qty, price, time=None, reason=""):
        px = price * (1 + self.slippage)
        # Never spend more than available cash, including the fee.
        qty = min(qty, self._cash / (px * (1 + self.fee_rate)))
        if qty <= 1e-12:
            return None
        fee = qty * px * self.fee_rate
        self._cash -= qty * px + fee
        self._position += qty
        fill = Fill(time, "buy", qty, px, fee, reason)
        self.fills.append(fill)
        return fill

    def sell(self, qty, price, time=None, reason=""):
        qty = min(qty, self._position)
        if qty <= 1e-12:
            return None
        px = price * (1 - self.slippage)
        fee = qty * px * self.fee_rate
        self._cash += qty * px - fee
        self._position -= qty
        fill = Fill(time, "sell", qty, px, fee, reason)
        self.fills.append(fill)
        return fill


class CCXTBroker(Broker):
    """Places real market orders on a crypto exchange through ccxt.

    REAL MONEY. Only constructed when the CLI is run with --live and API keys are set.
    Use `sandbox=True` (the default from the CLI unless --no-sandbox) to hit the exchange testnet.
    """

    def __init__(self, exchange_id: str, symbol: str, api_key: str, secret: str, sandbox: bool = True, exchange=None):
        if exchange is None:
            import ccxt  # optional dependency
            exchange = getattr(ccxt, exchange_id)({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
            if sandbox:
                exchange.set_sandbox_mode(True)
        self.exchange = exchange
        self.symbol = symbol
        self.base, self.quote = symbol.split("/")
        self.fills: list[Fill] = []

    def _balance(self, asset: str) -> float:
        return float(self.exchange.fetch_balance().get(asset, {}).get("free") or 0.0)

    @property
    def cash(self) -> float:
        return self._balance(self.quote)

    @property
    def position(self) -> float:
        return self._balance(self.base)

    def _order(self, side, qty, time, reason):
        qty = float(self.exchange.amount_to_precision(self.symbol, qty))
        if qty <= 0:
            return None
        log.warning("LIVE %s %s %s (%s)", side.upper(), qty, self.symbol, reason)
        order = self.exchange.create_order(self.symbol, "market", side, qty)
        price = float(order.get("average") or order.get("price") or 0.0)
        fee = float((order.get("fee") or {}).get("cost") or 0.0)
        fill = Fill(time, side, float(order.get("filled") or qty), price, fee, reason)
        self.fills.append(fill)
        return fill

    def buy(self, qty, price, time=None, reason=""):
        return self._order("buy", qty, time, reason)

    def sell(self, qty, price, time=None, reason=""):
        return self._order("sell", min(qty, self.position), time, reason)
