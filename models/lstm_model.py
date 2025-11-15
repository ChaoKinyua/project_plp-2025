"""
LSTM forecasting pipeline for stock prices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam

from analysis.evaluation import MetricResult, regression_metrics
from config import CONFIG
from models.utils import create_sequences, get_callbacks, set_global_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class LSTMModelArtifacts:
    horizon: int
    model: Model
    history: Dict[str, List[float]] = field(default_factory=dict)
    metrics: Optional[MetricResult] = None


class LSTMForecaster:
    """
    Multi-horizon LSTM forecaster.
    """

    def __init__(
        self,
        lookback: Optional[int] = None,
        horizons: Optional[Iterable[int]] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        lookback = lookback or CONFIG.lstm.lookback_window
        horizons = list(horizons or CONFIG.lstm.forecast_horizons)
        self.lookback = lookback
        self.horizons = horizons
        self.target_column = target_column or CONFIG.features.target_column
        self.feature_columns = feature_columns
        self.models: Dict[int, LSTMModelArtifacts] = {}
        set_global_seed(CONFIG.lstm.seed)

    def _prepare_arrays(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = df[self.feature_columns].values if self.feature_columns else df.drop(columns=[self.target_column]).values
        target = df[self.target_column].values
        return features, target

    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Construct LSTM network according to configuration.
        """
        inputs = Input(shape=input_shape)
        x = inputs

        for units in CONFIG.lstm.units[:-1]:
            x = LSTM(units, return_sequences=True)(x)
            x = Dropout(CONFIG.lstm.dropout)(x)

        x = LSTM(CONFIG.lstm.units[-1], return_sequences=False)(x)
        x = Dropout(CONFIG.lstm.dropout)(x)
        outputs = Dense(1, activation="linear")(x)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=CONFIG.lstm.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[int, LSTMModelArtifacts]:
        """
        Train models for each forecast horizon.
        """
        if feature_columns:
            self.feature_columns = feature_columns
        elif self.feature_columns is None:
            self.feature_columns = [
                col for col in train_df.columns if col != self.target_column
            ]

        train_X_raw, train_y_raw = self._prepare_arrays(train_df)
        val_X_raw, val_y_raw = self._prepare_arrays(val_df)

        for horizon in self.horizons:
            LOGGER.info("Training LSTM for horizon %s days", horizon)
            X_train, y_train = create_sequences(train_X_raw, train_y_raw, self.lookback, horizon)
            X_val, y_val = create_sequences(val_X_raw, val_y_raw, self.lookback, horizon)

            if len(X_train) == 0 or len(X_val) == 0:
                raise ValueError("Insufficient data to create sequences. Adjust lookback or horizon.")

            model = self.build_model(input_shape=X_train.shape[1:])
            callbacks = get_callbacks(f"lstm_h{horizon}")

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=CONFIG.lstm.epochs,
                batch_size=CONFIG.lstm.batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            metrics = regression_metrics(y_val, model.predict(X_val).flatten())
            artifact = LSTMModelArtifacts(
                horizon=horizon,
                model=model,
                history=history.history,
                metrics=metrics,
            )
            self.models[horizon] = artifact

        return self.models

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int,
    ) -> np.ndarray:
        """
        Generate predictions for the latest sequence in dataframe.
        """
        if horizon not in self.models:
            raise KeyError(f"Horizon {horizon} model not trained.")

        features, targets = self._prepare_arrays(df)
        X, _ = create_sequences(features, targets, self.lookback, horizon)
        return self.models[horizon].model.predict(X, verbose=0).flatten()

    def evaluate(
        self,
        test_df: pd.DataFrame,
    ) -> Dict[int, MetricResult]:
        """
        Evaluate trained models on a test dataset.
        """
        results: Dict[int, MetricResult] = {}
        test_X_raw, test_y_raw = self._prepare_arrays(test_df)

        for horizon, artifact in self.models.items():
            X_test, y_test = create_sequences(test_X_raw, test_y_raw, self.lookback, horizon)
            if len(X_test) == 0:
                continue
            y_pred = artifact.model.predict(X_test, verbose=0).flatten()
            metrics = regression_metrics(y_test, y_pred)
            artifact.metrics = metrics
            results[horizon] = metrics

        return results




