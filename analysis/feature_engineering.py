"""
Feature engineering utilities for time-series models.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.core.window import Window
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import CONFIG, FeatureConfig

LOGGER = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    feature_config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Generate technical indicators, lag features, and rolling statistics.
    """
    feature_config = feature_config or CONFIG.features
    working_df = df.copy()

    working_df = add_technical_indicators(working_df, feature_config)

    if feature_config.use_lag_features:
        working_df = add_lag_features(working_df, feature_config.lag_days, feature_config.target_column)

    working_df = add_rolling_statistics(working_df, feature_config.rolling_windows)

    if feature_config.include_volume_indicators and "Volume" in working_df.columns:
        working_df = add_volume_indicators(working_df, feature_config.rolling_windows)

    if feature_config.seasonal_decompose:
        working_df = add_seasonal_components(
            working_df,
            period=feature_config.seasonal_period,
            target_column=feature_config.target_column,
        )

    working_df = working_df.dropna()
    return working_df


def add_technical_indicators(df: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    """
    Append technical indicator columns based on configuration.
    """
    indicators = set(ind.lower() for ind in feature_config.technical_indicators)
    df = df.copy()
    close = df["Close"]

    if "sma" in indicators:
        df["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
        df["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    if "ema" in indicators:
        df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
        df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    if "rsi" in indicators:
        df["RSI_14"] = relative_strength_index(close, window=14)
    if "macd" in indicators:
        macd_line, signal_line = macd(close)
        df["MACD"] = macd_line
        df["MACD_Signal"] = signal_line
    if "bollinger" in indicators:
        middle, upper, lower = bollinger_bands(close, window=20)
        df["Bollinger_Middle"] = middle
        df["Bollinger_Upper"] = upper
        df["Bollinger_Lower"] = lower
    if "stochastic" in indicators and {"High", "Low"}.issubset(df.columns):
        stoch_k, stoch_d = stochastic_oscillator(df["High"], df["Low"], close)
        df["Stochastic_%K"] = stoch_k
        df["Stochastic_%D"] = stoch_d
    return df


def add_lag_features(df: pd.DataFrame, lags: Iterable[int], target_column: str) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)
    return df


def add_rolling_statistics(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        rolling: Window = df["Close"].rolling(window=window)
        df[f"RollingMean_{window}"] = rolling.mean()
        df[f"RollingStd_{window}"] = rolling.std()
        df[f"RollingMin_{window}"] = rolling.min()
        df[f"RollingMax_{window}"] = rolling.max()
    return df


def add_volume_indicators(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    df = df.copy()
    volume = df["Volume"]
    for window in windows:
        vol_rolling = volume.rolling(window=window)
        df[f"VolumeMean_{window}"] = vol_rolling.mean()
        df[f"VolumeStd_{window}"] = vol_rolling.std()
        df[f"OnBalanceVolume_{window}"] = on_balance_volume(df["Close"], volume).rolling(window=window).mean()
    return df


def add_seasonal_components(df: pd.DataFrame, period: int, target_column: str) -> pd.DataFrame:
    df = df.copy()
    try:
        decomposition = seasonal_decompose(df[target_column], model="additive", period=period, extrapolate_trend="freq")
        df[f"{target_column}_Trend"] = decomposition.trend
        df[f"{target_column}_Seasonal"] = decomposition.seasonal
        df[f"{target_column}_Resid"] = decomposition.resid
    except ValueError as exc:
        LOGGER.warning("Seasonal decomposition failed: %s", exc)
    return df


def handle_multicollinearity(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Iteratively remove features with Variance Inflation Factor above threshold.
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    vif_scores = compute_vif(df[columns])

    while any(score > threshold for score in vif_scores.values()) and len(columns) > 1:
        col_to_remove = max(vif_scores, key=vif_scores.get)
        LOGGER.info("Dropping %s due to VIF=%.2f", col_to_remove, vif_scores[col_to_remove])
        columns.remove(col_to_remove)
        vif_scores = compute_vif(df[columns])

    return df[columns], vif_scores


def compute_vif(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute variance inflation factor for each column.
    """
    df = df.select_dtypes(include=[np.number]).dropna()
    if df.empty:
        return {}

    vif_values = {}
    for i, column in enumerate(df.columns):
        try:
            vif_values[column] = float(variance_inflation_factor(df.values, i))
        except np.linalg.LinAlgError:
            vif_values[column] = float("inf")
    return vif_values


# --- Indicator helpers -------------------------------------------------------------------------

def relative_strength_index(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace({0: np.nan})
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    return rolling_mean, upper_band, lower_band


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace({0: np.nan})
    percent_d = percent_k.rolling(window=3).mean()
    return percent_k, percent_d


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    obv = (volume * direction).cumsum()
    return obv




