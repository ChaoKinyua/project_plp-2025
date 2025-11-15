"""
Evaluation utilities for regression and forecasting models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class MetricResult:
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "MSE": self.mse,
            "RMSE": self.rmse,
            "MAE": self.mae,
            "MAPE": self.mape,
            "R2": self.r2,
        }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    """
    Compute regression metrics for predictions.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return MetricResult(mse=mse, rmse=rmse, mae=mae, mape=mape, r2=r2)


def metrics_table(results: Dict[str, MetricResult]) -> pd.DataFrame:
    """
    Convert metrics dictionary to a dataframe for comparison.
    """
    data = {model: metrics.as_dict() for model, metrics in results.items()}
    return pd.DataFrame(data).T.sort_values(by="RMSE")


def residuals_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: Iterable,
    model_name: str,
) -> pd.DataFrame:
    """
    Build a dataframe with residual information for diagnostics.
    """
    residuals = y_true - y_pred
    return pd.DataFrame(
        {
            "Actual": y_true,
            "Predicted": y_pred,
            "Residual": residuals,
            "AbsResidual": np.abs(residuals),
            "SquaredResidual": residuals**2,
            "Model": model_name,
        },
        index=index,
    )


def confidence_intervals(
    predictions: np.ndarray,
    std_dev: np.ndarray,
    z_score: float = 1.96,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute upper and lower confidence bounds assuming normal residuals.
    """
    lower = predictions - z_score * std_dev
    upper = predictions + z_score * std_dev
    return lower, upper


def summarize_cross_validation(
    metric_results: List[MetricResult],
) -> Dict[str, float]:
    """
    Aggregate metrics across cross-validation folds.
    """
    aggregated = {}
    metrics_keys = metric_results[0].as_dict().keys()
    for metric_name in metrics_keys:
        values = [result.as_dict()[metric_name] for result in metric_results]
        aggregated[f"{metric_name}_mean"] = float(np.mean(values))
        aggregated[f"{metric_name}_std"] = float(np.std(values))
    return aggregated




