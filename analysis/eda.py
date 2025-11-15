"""
Exploratory data analysis utilities.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for OHLCV data.
    """
    numeric_cols = df.select_dtypes(include=[np.number])
    summary = numeric_cols.describe().T
    summary["skewness"] = numeric_cols.skew()
    summary["kurtosis"] = numeric_cols.kurtosis()
    return summary


def compute_volatility(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Calculate annualized volatility based on log returns.
    """
    returns = np.log(df["Close"]).diff()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility


def moving_averages(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Append moving averages for provided window sizes.
    """
    ma_df = df.copy()
    for window in windows:
        ma_df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()
    return ma_df


def volume_analysis(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Calculate volume z-scores to identify abnormal activity.
    """
    volume = df["Volume"]
    rolling_mean = volume.rolling(window=window).mean()
    rolling_std = volume.rolling(window=window).std()
    z_score = (volume - rolling_mean) / rolling_std.replace({0: np.nan})
    return z_score


def gap_detection(df: pd.DataFrame, threshold: float = 0.03) -> pd.DataFrame:
    """
    Identify price gaps larger than threshold percentage.
    """
    gap_df = df.copy()
    gap_df["prev_close"] = df["Close"].shift(1)
    gap_df["gap_pct"] = (df["Open"] - gap_df["prev_close"]) / gap_df["prev_close"]
    return gap_df[np.abs(gap_df["gap_pct"]) >= threshold][["Open", "prev_close", "gap_pct"]]


def rolling_correlation(df: pd.DataFrame, benchmark: pd.Series, window: int = 63) -> pd.Series:
    """
    Compute rolling correlation with a benchmark series.
    """
    return df["Close"].pct_change().rolling(window=window).corr(benchmark.pct_change())




