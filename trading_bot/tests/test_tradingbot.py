import numpy as np
import pandas as pd
import pytest

from tradingbot.backtest import run_backtest
from tradingbot.bot import TradingBot
from tradingbot.broker import PaperBroker
from tradingbot.data import synthetic
from tradingbot.risk import RiskManager
from tradingbot.strategies import RSIMeanReversion, Signal, SMACrossover, get_strategy
from tradingbot.strategies.rsi import rsi


def frame(closes):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx)


def test_sma_cross_signals_on_crossovers():
    df = frame([10] * 5 + [20] * 5 + [5] * 5)
    sig = SMACrossover(fast=2, slow=4).generate_signals(df)
    assert (sig == Signal.BUY).sum() == 1 and (sig == Signal.SELL).sum() == 1
    assert sig[sig == Signal.BUY].index[0] < sig[sig == Signal.SELL].index[0]


def test_strategies_have_no_lookahead():
    df = synthetic(400, seed=1)
    for strat in (SMACrossover(5, 20), RSIMeanReversion()):
        full = strat.generate_signals(df)
        for n in (100, 250, 399):
            assert full.iloc[n - 1] == strat.generate_signals(df.iloc[:n]).iloc[-1]


def test_rsi_bounds():
    r = rsi(synthetic(500)["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_paper_broker_accounting():
    b = PaperBroker(1000, fee_rate=0.01, slippage=0.0)
    fill = b.buy(100, 10)  # can't afford 100 units; should cap at cash
    assert b.cash >= -1e-9 and fill.qty < 100
    b.sell(b.position, 10)
    assert b.position == 0
    qty = 1000 / (10 * 1.01)  # all cash, fee included
    assert fill.qty == pytest.approx(qty)
    assert b.cash == pytest.approx(qty * 10 * 0.99)  # proceeds minus 1% sell fee


def test_position_size_respects_risk_and_caps():
    rm = RiskManager(risk_per_trade=0.01, max_position_pct=0.5, stop_loss_pct=0.05)
    assert rm.position_size(10_000, 10_000, 100) * 100 == pytest.approx(2_000)  # 1% / 5%
    rm = RiskManager(risk_per_trade=0.1, max_position_pct=0.5, stop_loss_pct=0.05)
    assert rm.position_size(10_000, 10_000, 100) * 100 == pytest.approx(5_000)  # capped


def test_stop_loss_gap_fills_at_open():
    rm = RiskManager(stop_loss_pct=0.05, take_profit_pct=None)
    assert rm.exit_reason(100, open_=90, low=88, high=91) == ("stop_loss", 90)
    assert rm.exit_reason(100, open_=99, low=94, high=100) == ("stop_loss", 95)
    assert rm.exit_reason(100, open_=99, low=96, high=100) is None


def test_backtest_executes_next_open_and_balances():
    df = frame([10] * 5 + [20] * 5 + [5] * 5)
    res = run_backtest(df, SMACrossover(2, 4), RiskManager(stop_loss_pct=None, take_profit_pct=None,
                                                          risk_per_trade=1, max_position_pct=1),
                       starting_cash=1000, fee_rate=0, slippage=0)
    buy_signal_time = SMACrossover(2, 4).generate_signals(df).eq(Signal.BUY).idxmax()
    assert res.fills[0].time > buy_signal_time
    assert len(res.trades) == 1
    assert res.equity.iloc[-1] == pytest.approx(1000 + res.trades[0].pnl)


def test_drawdown_halt():
    rm = RiskManager(max_drawdown_pct=0.2)
    assert rm.halted(79, 100) and not rm.halted(81, 100)


def test_bot_step_buys_once_per_bar_and_stops_out(tmp_path):
    class AlwaysBuy(SMACrossover):
        def latest_signal(self, data):
            return Signal.BUY

    broker = PaperBroker(1000, 0, 0)
    bot = TradingBot(AlwaysBuy(2, 4), broker, RiskManager(stop_loss_pct=0.05, take_profit_pct=None),
                     state_file=str(tmp_path / "s.json"))
    df = frame([100] * 10)
    assert bot.step(df) == "buy"
    assert bot.step(df) == "hold"  # same bar, no duplicate entry
    assert bot.step(df, price=94) == "stop_loss"
    assert broker.position == 0
    # state persists across restarts
    bot2 = TradingBot(AlwaysBuy(2, 4), broker, state_file=str(tmp_path / "s.json"))
    assert bot2.last_bar == bot.last_bar


def test_get_strategy_errors():
    with pytest.raises(ValueError):
        get_strategy("nope")
    with pytest.raises(ValueError):
        SMACrossover(50, 20)


def test_paper_balances_survive_restart(tmp_path):
    class AlwaysBuy(SMACrossover):
        def latest_signal(self, data):
            return Signal.BUY

    state = str(tmp_path / "s.json")
    broker = PaperBroker(1000, 0, 0)
    TradingBot(AlwaysBuy(2, 4), broker, state_file=state).step(frame([100] * 10))
    restored = PaperBroker(1000, 0, 0)
    bot = TradingBot(AlwaysBuy(2, 4), restored, state_file=state)
    assert broker.position > 0
    assert restored.position == pytest.approx(broker.position)
    assert restored.cash == pytest.approx(broker.cash)
    assert bot.entry_price == pytest.approx(100)


def test_run_survives_feed_errors():
    calls = {"n": 0}

    def flaky_feed():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("network down")
        return frame([100] * 10), 100.0

    bot = TradingBot(SMACrossover(2, 4), PaperBroker(1000))
    bot.run(flaky_feed, interval=0, max_steps=3)
    assert calls["n"] == 3
