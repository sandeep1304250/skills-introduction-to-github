"""Market data sources. All sources return a DataFrame indexed by timestamp with
columns: open, high, low, close, volume."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV with a timestamp/date column."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in ("timestamp", "date", "datetime", "time") if c in df.columns), None)
    if time_col is None:
        raise ValueError("CSV needs a timestamp/date/datetime/time column")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df[COLUMNS].astype(float)


def synthetic(bars: int = 1000, start_price: float = 100.0, drift: float = 0.0003,
              volatility: float = 0.02, seed: int | None = 42, freq: str = "1h") -> pd.DataFrame:
    """Geometric-Brownian-motion price series, handy for demos and tests."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, bars)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, volatility / 2, bars)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(100, 1000, bars)
    index = pd.date_range("2024-01-01", periods=bars, freq=freq)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def fetch_ccxt(exchange_id: str, symbol: str, timeframe: str = "1h", limit: int = 500, exchange=None) -> pd.DataFrame:
    """Fetch recent candles from a crypto exchange via ccxt (pip install ccxt)."""
    if exchange is None:
        import ccxt  # optional dependency
        exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", *COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp").astype(float)
