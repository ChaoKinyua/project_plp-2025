"""
Main entry point orchestrating the stock prediction workflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from analysis import evaluation as eval_utils
from analysis import feature_engineering
from analysis import eda
from backtesting.performance import buy_and_hold_benchmark, compute_portfolio_stats
from backtesting.strategy import WalkForwardBacktester
from config import CONFIG
from data.data_loader import StockDataLoader
from models.arima_model import ArimaForecaster
from models.ensemble_model import TraditionalRegressors
from models.lstm_model import LSTMForecaster
from visualization import dashboard, plots


def configure_logging() -> None:
    Path(CONFIG.logging.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, CONFIG.logging.level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(CONFIG.logging.log_file) if CONFIG.logging.log_to_file else logging.NullHandler(),
            logging.StreamHandler(),
        ],
    )


def run_pipeline() -> None:
    configure_logging()
    logger = logging.getLogger("pipeline")
    data_loader = StockDataLoader()

    for ticker, df in data_loader.fetch_data().items():
        logger.info("Processing ticker %s", ticker)

        summary = eda.summary_statistics(df)
        summary.to_csv(Path(CONFIG.data.processed_data_dir) / f"{ticker}_summary.csv")

        engineered_df = feature_engineering.engineer_features(df)
        data_loader.save_processed(engineered_df, ticker)

        train_df, val_df, test_df = data_loader.train_validation_test_split(engineered_df)

        feature_columns = [
            col
            for col in engineered_df.columns
            if col != CONFIG.features.target_column and pd.api.types.is_numeric_dtype(engineered_df[col])
        ]

        scaled_train, _ = data_loader.scale_features(train_df[feature_columns], feature_columns)
        scaled_val = data_loader.transform_with_fitted_scalers(val_df[feature_columns])
        scaled_test = data_loader.transform_with_fitted_scalers(test_df[feature_columns])

        train_ready = scaled_train.join(train_df[[CONFIG.features.target_column]])
        val_ready = scaled_val.join(val_df[[CONFIG.features.target_column]])
        test_ready = scaled_test.join(test_df[[CONFIG.features.target_column]])

        metrics: Dict[str, eval_utils.MetricResult] = {}

        # LSTM
        lstm_forecaster = LSTMForecaster(feature_columns=feature_columns)
        lstm_forecaster.fit(train_ready, val_ready)
        lstm_results = lstm_forecaster.evaluate(test_ready)
        for horizon, metric in lstm_results.items():
            metrics[f"LSTM_{horizon}d"] = metric

        # ARIMA
        arima = ArimaForecaster()
        arima.fit(train_ready[CONFIG.features.target_column], val_ready[CONFIG.features.target_column])
        arima_metric = arima.evaluate(test_ready[CONFIG.features.target_column])
        metrics["ARIMA"] = arima_metric

        # Traditional ML
        traditional = TraditionalRegressors()
        traditional.train(train_ready, val_ready, feature_columns=feature_columns)
        traditional_metrics = traditional.evaluate(test_ready, feature_columns=feature_columns)
        metrics.update({name.upper(): metric for name, metric in traditional_metrics.items()})

        metrics_table = eval_utils.metrics_table(metrics)
        metrics_table.to_csv(Path(CONFIG.data.processed_data_dir) / f"{ticker}_metrics.csv")

        # Visualization
        plots.plot_historical_prices(df, ticker, save=True)

        for model_name, artifact in traditional.artifacts.items():
            if artifact.val_predictions is None:
                continue
            predictions_series = pd.Series(
                artifact.val_predictions,
                index=val_ready.index[: len(artifact.val_predictions)],
            )
            actual_series = val_ready[CONFIG.features.target_column].iloc[: len(artifact.val_predictions)]
            plots.plot_predictions(actual_series, predictions_series, ticker, horizon=1, save=True)

        # Backtesting on Random Forest signals as example
        backtester = WalkForwardBacktester()

        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        def training_fn(train_slice: pd.DataFrame):
            target = CONFIG.features.target_column
            X_train = train_slice[feature_columns]
            y_train = train_slice[target]
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            )
            pipeline.fit(X_train, y_train)
            return pipeline

        def prediction_fn(model, test_slice: pd.DataFrame):
            preds = model.predict(test_slice[feature_columns])
            return pd.Series(preds, index=test_slice.index)

        walk_forward_result = backtester.run(
            engineered_df,
            training_fn=training_fn,
            prediction_fn=prediction_fn,
        )
        portfolio = compute_portfolio_stats(walk_forward_result.trades)
        benchmark = buy_and_hold_benchmark(engineered_df[CONFIG.features.target_column])

        logger.info(
            "Backtest metrics: %s | Buy&Hold=%.2f%%",
            portfolio.attrs["metrics"],
            benchmark * 100,
        )

        # Dashboard example
        dash_fig = dashboard.build_dashboard(
            engineered_df,
            predictions={"WalkForward": walk_forward_result.predictions},
            metrics_table=metrics_table,
        )
        dash_fig.write_html(Path(CONFIG.visualization.output_dir) / f"{ticker}_dashboard.html")


if __name__ == "__main__":
    run_pipeline()

