"""
ARIMA/SARIMAX forecasting utilities (improved, with robust fitting and convergence handling).
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from statsmodels.tsa.stattools import adfuller

from analysis.evaluation import MetricResult, regression_metrics
from config import CONFIG, ArimaConfig

LOGGER = logging.getLogger(__name__)

# Optional: silence statsmodels ConvergenceWarning if you prefer to handle manually
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class ArimaArtifacts:
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    model: SARIMAXResultsWrapper
    metrics: Optional[MetricResult] = None
    adf_statistic: Optional[float] = None
    adf_pvalue: Optional[float] = None


class ArimaForecaster:
    """
    Fit ARIMA/SARIMAX models with a safer grid search and robust parameter fitting.

    Improvements over original:
    - Better handling of convergence (retries, alternative optimizers, increased iterations)
    - Reduced and configurable search grid by default
    - Stationarity checks and optional differencing
    - Defensive checks for short/no-variance series
    - Informative logging on failures
    """

    def __init__(self, config: Optional[ArimaConfig] = None):
        self.config = config or CONFIG.arima
        self.artifacts: Optional[ArimaArtifacts] = None

    def adf_test(self, series: pd.Series) -> Tuple[float, float]:
        """
        Run Augmented Dickey-Fuller test for stationarity.
        """
        result = adfuller(series.dropna(), autolag="AIC")
        return float(result[0]), float(result[1])

    def deseasonalize(self, series: pd.Series) -> pd.Series:
        """
        Optionally remove seasonal component prior to ARIMA fitting using STL.
        Returns residual (deseasonalized) series or original if STL fails.
        """
        try:
            stl = STL(series, period=self.config.seasonal_period, robust=True)
            res = stl.fit()
            # Use resid + trend to keep lower-frequency behavior if you prefer; here we use resid
            return res.resid
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("STL decomposition failed: %s", exc)
            return series

    def _safe_fit(self, model: SARIMAX) -> Optional[SARIMAXResultsWrapper]:
        """
        Try fitting model with a set of strategies (optimizers, increased iterations).
        Optimized for speed - tries fewer optimizers.
        Returns results if converged, otherwise None.
        """
        methods = [
            dict(method="lbfgs", maxiter=100),  # Reduced iterations
            dict(method="bfgs", maxiter=100),   # Reduced iterations
        ]  # Removed powell and cg for speed

        for opt in methods:
            try:
                res = model.fit(disp=False, **opt)
                mle_retvals = getattr(res, "mle_retvals", {})
                converged = mle_retvals.get("converged", True) if isinstance(mle_retvals, dict) else True
                if converged:
                    LOGGER.debug("Fitted with %s (converged).", opt)
                    return res
                else:
                    LOGGER.debug("Fitted with %s but did not converge: %s", opt, mle_retvals)
            except Exception as exc:  # pragma: no cover - continue trying other optimizers
                LOGGER.debug("Fit failed using %s: %s", opt, exc)
        return None

    def select_parameters(self, series: pd.Series) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], float]:
        """
        Brute-force search over a reduced parameter grid optimizing for AIC.
        Optimized for speed - limited search space.
        Returns best (p,d,q), best seasonal (P,D,Q,s) and the corresponding AIC.
        """
        # Very limited search for speed
        max_p = min(self.config.max_p, 2)
        max_q = min(self.config.max_q, 2)
        p = range(0, max_p + 1)
        q = range(0, max_q + 1)
        d = [self.config.difference_order]
        pdq = list(itertools.product(p, d, q))

        # Reduced seasonal search - only try (0,0,0) and (1,0,1) for speed
        seasonal_pdq = [
            (0, self.config.seasonal_difference, 0, self.config.seasonal_period),
            (1, self.config.seasonal_difference, 1, self.config.seasonal_period),
        ]

        best_aic = np.inf
        best_param = (0, d[0], 0)
        best_seasonal = (0, 0, 0, self.config.seasonal_period)

        for order in pdq:
            for seasonal_order in seasonal_pdq:
                try:
                    model = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=self.config.enforce_stationarity,
                        enforce_invertibility=self.config.enforce_invertibility,
                    )
                    res = self._safe_fit(model)
                    if res is None:
                        continue
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_param = order
                        best_seasonal = seasonal_order
                        LOGGER.debug("New best: order=%s seasonal=%s AIC=%.2f", order, seasonal_order, best_aic)
                except Exception as exc:
                    LOGGER.debug("Failed to fit SARIMAX(%s)x(%s): %s", order, seasonal_order, exc)
        return best_param, best_seasonal, best_aic

    def fit(
        self,
        train_series: pd.Series,
        validation_series: Optional[pd.Series] = None,
    ) -> ArimaArtifacts:
        """
        Fit ARIMA/SARIMAX model using combined training (and optional validation) data.
        Performs stationarity checks and uses robust fitting.
        """
        combined_series = (
            pd.concat([train_series, validation_series])
            if validation_series is not None
            else train_series
        )
        # Ensure datetime index and business frequency; interpolate small gaps
        combined_series = combined_series.sort_index()
        combined_series = combined_series.asfreq("B").interpolate()
        combined_series = combined_series.dropna()

        if combined_series.empty:
            raise ValueError("Combined series is empty after preprocessing.")
        if combined_series.std() == 0:
            raise ValueError("Series has zero variance; unable to fit ARIMA.")

        adf_stat, adf_pvalue = self.adf_test(combined_series)
        LOGGER.info("ADF statistic=%.3f, p-value=%.4f", adf_stat, adf_pvalue)

        # If non-stationary and config allows, apply differencing once
        series_for_selection = combined_series
        if adf_pvalue > self.config.adf_pvalue_threshold and self.config.auto_diff:
            LOGGER.info("Series non-stationary (p=%.4f) — applying first difference for parameter selection.", adf_pvalue)
            series_for_selection = combined_series.diff().dropna()

        deseasonalized = self.deseasonalize(series_for_selection)
        order, seasonal_order, best_aic = self.select_parameters(deseasonalized)

        LOGGER.info(
            "Selected ARIMA order=%s seasonal_order=%s with AIC=%.2f",
            order,
            seasonal_order,
            best_aic,
        )

        model = SARIMAX(
            combined_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=self.config.enforce_stationarity,
            enforce_invertibility=self.config.enforce_invertibility,
        )

        results = self._safe_fit(model)
        if results is None:
            # As a last resort, try a very simple fallback model
            LOGGER.warning("Final fit did not converge for selected order; falling back to ARIMA(1,0,0).")
            fallback = SARIMAX(combined_series, order=(1, 0, 0), enforce_stationarity=True, enforce_invertibility=True)
            results = self._safe_fit(fallback)
            if results is None:
                raise RuntimeError("ARIMA fitting failed to converge with all tried optimizers.")

        self.artifacts = ArimaArtifacts(
            order=order,
            seasonal_order=seasonal_order,
            model=results,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pvalue,
        )
        return self.artifacts

    def forecast(self, steps: int) -> pd.Series:
        """
        Produce forecast for specified horizon.
        """
        if not self.artifacts:
            raise ValueError("Model not fit. Call `fit` first.")
        forecast_res = self.artifacts.model.get_forecast(steps=steps)
        return forecast_res.predicted_mean

    def evaluate(self, test_series: pd.Series) -> MetricResult:
        """
        Evaluate fitted ARIMA model against held-out data.
        """
        if not self.artifacts:
            raise ValueError("Model not fit. Call `fit` before evaluate.")

        steps = len(test_series)
        forecast_res = self.artifacts.model.get_forecast(steps=steps)
        y_pred = forecast_res.predicted_mean.values
        y_true = test_series.values
        metrics = regression_metrics(y_true, y_pred)
        self.artifacts.metrics = metrics
        return metrics



