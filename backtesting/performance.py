"""
Portfolio performance analytics for backtesting results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG


def compute_portfolio_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Generate portfolio equity curve and associated statistics.
    """
    net_returns = trades["NetReturn"]
    equity_curve = (1 + net_returns).cumprod() * CONFIG.backtest.initial_investment
    drawdown_series = calculate_drawdown(equity_curve)
    sharpe = sharpe_ratio(net_returns)
    sortino = sortino_ratio(net_returns)

    summary = pd.DataFrame(
        {
            "Equity": equity_curve,
            "Drawdown": drawdown_series,
            "NetReturn": net_returns,
        }
    )
    summary.attrs["metrics"] = {
        "CumulativeReturn": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": drawdown_series.min(),
    }
    return summary


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve / cumulative_max - 1
    return drawdown


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = None) -> float:
    risk_free_rate = risk_free_rate or CONFIG.backtest.risk_free_rate / 252
    excess_returns = returns - risk_free_rate
    std = returns.std()
    if std == 0:
        return np.nan
    return np.sqrt(252) * excess_returns.mean() / std


def sortino_ratio(returns: pd.Series, risk_free_rate: float = None) -> float:
    risk_free_rate = risk_free_rate or CONFIG.backtest.risk_free_rate / 252
    downside = returns[returns < 0]
    downside_std = downside.std()
    if downside_std == 0:
        return np.nan
    return np.sqrt(252) * (returns.mean() - risk_free_rate) / downside_std


def buy_and_hold_benchmark(prices: pd.Series) -> float:
    return prices.iloc[-1] / prices.iloc[0] - 1




