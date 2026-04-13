from __future__ import annotations
"""Backtesting module for LSTM strategy vs benchmark."""
import numpy as np
import pandas as pd
from typing import Tuple

from src import config

def calc_max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative returns."""
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(np.min(dd))

def calc_sharpe_ratio(returns: np.ndarray, risk_free: float = 0.03) -> float:
    """Annualised Sharpe ratio (252 trading days)."""
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free / 252
    return float(np.sqrt(252) * excess.mean() / returns.std())

def backtest_strategy(pred_probs: np.ndarray, actual_returns: np.ndarray,
                      threshold: float = 0.55) -> dict:
    """Backtest LSTM strategy with simple threshold rule.

    Params
    ------
    pred_probs    : (N,) predicted up-probability per day
    actual_returns: (N,) actual next-day returns
    threshold     : entry threshold (default 0.55)

    Returns
    -------
    dict with keys: total_return, sharpe, max_drawdown, num_trades, win_rate
    """
    N = min(len(pred_probs), len(actual_returns))

    # Strategy: hold when prob > threshold
    signals = (pred_probs[:N] >= threshold).astype(float)
    strategy_returns = signals * actual_returns[:N]

    total_return = float(np.expm1(np.sum(np.log1p(strategy_returns))))
    sharpe       = calc_sharpe_ratio(strategy_returns)
    mdd          = calc_max_drawdown(strategy_returns)

    num_trades = int(signals.sum())
    if num_trades == 0:
        return {"total_return": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "num_trades": 0, "win_rate": 0.0}

    # Win rate: days where return > 0 among trades
    trade_mask = signals == 1
    win_rate   = float((actual_returns[:N][trade_mask] > 0).mean())

    return {
        "total_return":  round(total_return, 4),
        "sharpe":         round(sharpe, 4),
        "max_drawdown":   round(mdd, 4),
        "num_trades":     num_trades,
        "win_rate":       round(win_rate, 4),
    }

def backtest_vs_benchmark(strategy_returns: np.ndarray,
                          benchmark_returns: np.ndarray) -> dict:
    """Compare LSTM strategy vs buy-and-hold benchmark."""
    strat_total = float(np.expm1(np.sum(np.log1p(strategy_returns))))
    bench_total = float(np.expm1(np.sum(np.log1p(benchmark_returns))))

    return {
        "strategy_total_return": round(strat_total, 4),
        "benchmark_total_return": round(bench_total, 4),
        "alpha":                  round(strat_total - bench_total, 4),
        "strategy_sharpe":        round(calc_sharpe_ratio(strategy_returns), 4),
        "benchmark_sharpe":       round(calc_sharpe_ratio(benchmark_returns), 4),
        "strategy_mdd":          round(calc_max_drawdown(strategy_returns), 4),
        "benchmark_mdd":         round(calc_max_drawdown(benchmark_returns), 4),
    }

if __name__ == "__main__":
    # Mock demo
    np.random.seed(42)
    N = 500
    probs = np.random.uniform(0.3, 0.9, N)
    actual = np.random.normal(0.001, 0.02, N)
    benchmark = np.random.normal(0.0008, 0.015, N)

    result = backtest_strategy(probs, actual)
    print("Strategy:", result)

    cmp = backtest_vs_benchmark(actual, benchmark)
    print("vs Benchmark:", cmp)
