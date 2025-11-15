"""
Visualization utilities for stock prediction analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from analysis.evaluation import MetricResult
from config import CONFIG


def plot_historical_prices(df: pd.DataFrame, ticker: str, save: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    df["Close"].plot(ax=ax, label="Close Price")
    ax.set_title(f"{ticker} Historical Close Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    if save and CONFIG.visualization.save_plots:
        _persist_matplotlib(fig, f"{ticker}_historical")
    return fig


def plot_predictions(
    actual: pd.Series,
    predictions: pd.Series,
    ticker: str,
    horizon: int,
    confidence_intervals: Optional[pd.DataFrame] = None,
    save: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    actual.plot(ax=ax, label="Actual", color="black")
    predictions.plot(ax=ax, label=f"Predicted ({horizon}d)", color="tab:blue")

    if confidence_intervals is not None:
        ax.fill_between(
            confidence_intervals.index,
            confidence_intervals["lower"],
            confidence_intervals["upper"],
            color="tab:blue",
            alpha=0.2,
            label="Confidence Interval",
        )

    ax.set_title(f"{ticker} {horizon}-Day Forecast vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    if save and CONFIG.visualization.save_plots:
        _persist_matplotlib(fig, f"{ticker}_forecast_{horizon}d")
    return fig


def plot_residuals(residuals_df: pd.DataFrame, save: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    residuals_df["Residual"].plot(ax=ax, label="Residual")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Residuals - {residuals_df['Model'].iloc[0]}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.legend()
    if save and CONFIG.visualization.save_plots:
        _persist_matplotlib(fig, f"{residuals_df['Model'].iloc[0]}_residuals")
    return fig


def plot_error_distribution(residuals_df: pd.DataFrame, save: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals_df["Residual"], bins=30, kde=True, ax=ax)
    ax.set_title(f"Error Distribution - {residuals_df['Model'].iloc[0]}")
    ax.set_xlabel("Residual")
    if save and CONFIG.visualization.save_plots:
        _persist_matplotlib(fig, f"{residuals_df['Model'].iloc[0]}_error_distribution")
    return fig


def plot_model_comparison(metrics: Dict[str, MetricResult], metric_name: str = "RMSE", save: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_values = {name: getattr(result, metric_name.lower()) for name, result in metrics.items()}
    sns.barplot(x=list(metric_values.keys()), y=list(metric_values.values()), ax=ax)
    ax.set_title(f"Model Comparison - {metric_name}")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Model")
    if save and CONFIG.visualization.save_plots:
        _persist_matplotlib(fig, f"model_comparison_{metric_name}")
    return fig


def plot_feature_importance(feature_importances: pd.Series, save: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importances.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    if save and CONFIG.visualization.save_plots:
        _persist_matplotlib(fig, "feature_importance")
    return fig


def interactive_prediction_chart(actual: pd.Series, predictions: pd.Series, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines", name="Actual"))
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions.values,
            mode="lines",
            name="Predicted",
        )
    )
    fig.update_layout(
        title=f"{ticker} Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )
    return fig


def _persist_matplotlib(fig: plt.Figure, filename: str) -> None:
    output_dir = Path(CONFIG.visualization.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.png", bbox_inches="tight")




