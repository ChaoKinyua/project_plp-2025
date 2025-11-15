"""
Backtesting and walk-forward validation utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import CONFIG, BacktestConfig

LOGGER = logging.getLogger(__name__)


PredictionFn = Callable[[pd.DataFrame], pd.Series]
TrainingFn = Callable[[pd.DataFrame], object]


@dataclass
class WalkForwardResult:
    predictions: pd.Series
    signals: pd.Series
    trades: pd.DataFrame


class WalkForwardBacktester:
    """
    Execute walk-forward validation using a model-specific training and prediction callable.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or CONFIG.backtest

    def run(
        self,
        df: pd.DataFrame,
        training_fn: TrainingFn,
        prediction_fn: PredictionFn,
        window: Optional[int] = None,
        step: Optional[int] = None,
    ) -> WalkForwardResult:
        df = df.copy()
        window = window or self.config.walk_forward_window
        step = step or self.config.rebalance_frequency

        predictions = []
        signals = []

        for start in range(window, len(df) - step, step):
            train_slice = df.iloc[start - window : start]
            test_slice = df.iloc[start : start + step]

            model = training_fn(train_slice)
            horizon_predictions = prediction_fn(model, test_slice)
            horizon_predictions.index = test_slice.index[: len(horizon_predictions)]
            predictions.append(horizon_predictions)
            signals.append(generate_signals(horizon_predictions, test_slice["Close"]))

        if not predictions:
            raise ValueError("Walk-forward produced no predictions. Increase dataset size or adjust window.")

        predictions_series = pd.concat(predictions)
        signals_series = pd.concat(signals)
        trades = simulate_trades(df["Close"], signals_series)
        return WalkForwardResult(predictions_series, signals_series, trades)


def generate_signals(predictions: pd.Series, actuals: pd.Series, threshold: Optional[float] = None) -> pd.Series:
    """
    Create long/short signals based on expected returns.
    """
    threshold = threshold or CONFIG.backtest.signal_threshold
    expected_return = (predictions - actuals) / actuals

    signals = pd.Series(index=predictions.index, dtype=float)
    signals[expected_return > threshold] = 1.0
    signals[expected_return < -threshold] = -1.0
    signals[(expected_return >= -threshold) & (expected_return <= threshold)] = 0.0
    return signals


def simulate_trades(prices: pd.Series, signals: pd.Series) -> pd.DataFrame:
    """
    Simulate trades and compute position returns.
    """
    aligned_prices = prices.loc[signals.index]
    returns = aligned_prices.pct_change().fillna(0)
    position_returns = returns * signals.shift(1).fillna(0)

    transaction_cost = CONFIG.backtest.transaction_cost
    trade_changes = signals.diff().abs()
    costs = trade_changes * transaction_cost
    net_returns = position_returns - costs

    trades = pd.DataFrame(
        {
            "Price": aligned_prices,
            "Signal": signals,
            "Returns": returns,
            "PositionReturn": position_returns,
            "TransactionCost": costs,
            "NetReturn": net_returns,
        }
    )
    return trades




