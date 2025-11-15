"""
Data fetching, preprocessing, and splitting utilities.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import CONFIG, FeatureConfig

LOGGER = logging.getLogger(__name__)


class PassthroughScaler:
    """Identity transformer that mimics scikit-learn scaler API."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


class StockDataLoader:
    """
    Encapsulates data acquisition, preprocessing, feature scaling, and dataset splitting.
    """

    def __init__(self, feature_config: Optional[FeatureConfig] = None):
        self.config = CONFIG.data
        self.feature_config = feature_config or CONFIG.features
        self.scalers: Dict[str, Tuple[object, List[str]]] = {}
        Path(self.config.raw_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.processed_data_dir).mkdir(parents=True, exist_ok=True)

    def fetch_data(self, tickers: Optional[List[str]] = None, force: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for tickers from Yahoo Finance.
        """
        tickers = tickers or self.config.tickers
        data_frames: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            cache_path = Path(self.config.raw_data_dir) / f"{ticker}.parquet"
            if self.config.cache_data and cache_path.exists() and not force:
                LOGGER.info("Loading %s from cache", ticker)
                data_frames[ticker] = pd.read_parquet(cache_path)
                continue

            LOGGER.info(
                "Downloading %s data from %s to %s",
                ticker,
                self.config.start_date,
                self.config.end_date,
            )
            df = yf.download(
                tickers=ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                interval=self.config.interval,
                auto_adjust=self.config.adjust_dividends,
                progress=False,
                group_by="ticker",
            )

            if df.empty:
                raise ValueError(f"No data returned for ticker {ticker}.")

            df = self._flatten_columns(df)
            df["Ticker"] = ticker
            df.index.name = "Date"
            df = self.clean_data(df)
            years_of_data = (df.index.max() - df.index.min()).days / 365.25
            if years_of_data < self.config.min_history_years:
                raise ValueError(
                    f"{ticker} does not have minimum {self.config.min_history_years} years of data."
                )

            if self.config.cache_data:
                df.to_parquet(cache_path, index=True)
            data_frames[ticker] = df
        return data_frames

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values and ensure the dataframe is sorted chronologically.
        """
        df = df.sort_index()
        if self.config.drop_missing:
            df = df.dropna()
        else:
            if self.config.fill_method == "ffill":
                df = df.ffill().bfill()
            elif self.config.fill_method == "bfill":
                df = df.bfill().ffill()
            elif self.config.fill_method == "interpolate":
                df = df.interpolate(method="time").ffill().bfill()
            else:
                raise ValueError(f"Unsupported fill method {self.config.fill_method}.")
        return df

    def scale_features(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Apply scaling to features based on configured scaling method.
        """
        feature_columns = feature_columns or df.columns.tolist()
        scaler = self._get_scaler(self.feature_config.scaling_method)
        scaled_values = scaler.fit_transform(df[feature_columns])
        scaled_df = df.copy()
        scaled_df[feature_columns] = scaled_values
        key = self._scaler_key(feature_columns)
        self.scalers[key] = (scaler, feature_columns)
        return scaled_df, self.scalers

    def transform_with_fitted_scalers(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Transform a dataframe using previously fitted scalers.
        """
        transformed = df.copy()
        feature_columns = feature_columns or df.columns.tolist()
        key = self._scaler_key(feature_columns)
        scaler_entry = self.scalers.get(key)
        if not scaler_entry:
            LOGGER.warning("Scaler for columns %s not found. Skipping.", feature_columns)
            return transformed
        scaler, columns = scaler_entry
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            LOGGER.warning("Columns %s missing from dataframe. Skipping scaler.", missing_columns)
            return transformed
        transformed[columns] = scaler.transform(df[columns])
        return transformed

    def train_validation_test_split(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test subsets using time-based or random splits.
        """
        target_column = target_column or self.feature_config.target_column
        if self.config.interval.endswith("m"):
            LOGGER.warning(
                "Minute-level data detected. Ensure data is aggregated appropriately before training."
            )

        split_config = CONFIG.split
        if split_config.time_based:
            total_rows = len(df)
            test_rows = int(total_rows * split_config.test_size)
            val_rows = int(total_rows * split_config.validation_size)
            train = df.iloc[: total_rows - (val_rows + test_rows)]
            val = df.iloc[total_rows - (val_rows + test_rows) : total_rows - test_rows]
            test = df.iloc[total_rows - test_rows :]
            return train, val, test

        train_df, test_df = train_test_split(
            df,
            test_size=split_config.test_size,
            shuffle=True,
            random_state=CONFIG.traditional.seed,
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=split_config.validation_size,
            shuffle=True,
            random_state=CONFIG.traditional.seed,
        )
        return train_df.sort_index(), val_df.sort_index(), test_df.sort_index()

    def walk_forward_splits(
        self, df: pd.DataFrame, n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits using TimeSeriesSplit.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(df))

    def save_processed(
        self, df: pd.DataFrame, ticker: str, suffix: str = "processed"
    ) -> Path:
        """
        Store processed dataframe to disk with metadata hash to support reproducibility.
        """
        metadata = {
            "config": asdict(CONFIG.data),
            "feature_config": asdict(self.feature_config),
            "rows": len(df),
        }
        metadata_hash = hashlib.md5(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        filename = f"{ticker}_{suffix}_{metadata_hash[:8]}.parquet"
        output_path = Path(self.config.processed_data_dir) / filename
        df.to_parquet(output_path, index=True)
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return output_path

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten multi-index columns returned by yfinance for single ticker requests.
        """
        if isinstance(df.columns, pd.MultiIndex):
            tickers = df.columns.get_level_values(0).unique()
            if len(tickers) == 1:
                df.columns = [col[1] for col in df.columns]
            else:
                df.columns = ["_".join(filter(None, col)).strip("_") for col in df.columns]
        return df

    def _get_scaler(self, method: str):
        if method == "standard":
            return StandardScaler()
        if method == "minmax":
            return MinMaxScaler()
        if method == "robust":
            return RobustScaler()
        if method in {"none", "identity", "", None}:
            return PassthroughScaler()
        raise ValueError(f"Unknown scaling method {method}.")

    def _scaler_key(self, columns: List[str]) -> str:
        """
        Generate a consistent key for a set of columns.
        """
        return "|".join(columns)


def concatenate_ticker_data(data_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple ticker dataframes into a single panel with multi-index.
    """
    concatenated = pd.concat(data_frames, names=["Ticker", "Date"])
    concatenated = concatenated.reset_index().set_index(["Date", "Ticker"]).sort_index()
    return concatenated

