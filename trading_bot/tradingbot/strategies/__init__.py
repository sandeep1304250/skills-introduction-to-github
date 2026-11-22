from .base import Signal, Strategy
from .rsi import RSIMeanReversion
from .sma_cross import SMACrossover

STRATEGIES = {
    "sma_cross": SMACrossover,
    "rsi": RSIMeanReversion,
}


def get_strategy(name: str, **params) -> Strategy:
    try:
        return STRATEGIES[name](**params)
    except KeyError:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {', '.join(STRATEGIES)}") from None


__all__ = ["Signal", "Strategy", "SMACrossover", "RSIMeanReversion", "STRATEGIES", "get_strategy"]
