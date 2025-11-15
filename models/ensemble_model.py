"""
Traditional machine learning regressors and ensemble techniques.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from analysis.evaluation import MetricResult, regression_metrics
from config import CONFIG
from models.utils import save_sklearn_model

LOGGER = logging.getLogger(__name__)


@dataclass
class SklearnModelArtifact:
    name: str
    estimator: BaseEstimator
    metrics: Optional[MetricResult] = None
    best_params: Optional[Dict] = None
    val_predictions: Optional[np.ndarray] = None
    test_predictions: Optional[np.ndarray] = None


class TraditionalRegressors:
    """
    Train and evaluate several classical machine learning regressors.
    """

    def __init__(self):
        self.config = CONFIG.traditional
        self.artifacts: Dict[str, SklearnModelArtifact] = {}

    def _get_models(self) -> Dict[str, object]:
        models = {
            "random_forest": RandomForestRegressor(random_state=self.config.seed),
            "gradient_boosting": GradientBoostingRegressor(random_state=self.config.seed),
            "svr": SVR(),
            "linear_regression": LinearRegression(),
            "lasso": Lasso(random_state=self.config.seed),
        }
        return models

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        perform_grid_search: bool = True,
    ) -> Dict[str, SklearnModelArtifact]:
        target_column = target_column or CONFIG.features.target_column
        feature_columns = feature_columns or [
            col for col in train_df.columns if col != target_column
        ]

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_val = val_df[feature_columns]
        y_val = val_df[target_column]

        for model_name, estimator in self._get_models().items():
            LOGGER.info("Training %s", model_name)
            params_grid = self._get_param_grid(model_name) if perform_grid_search else None

            # Skip grid search if all parameters have only one value (no search needed)
            if params_grid:
                total_combinations = 1
                for param_values in params_grid.values():
                    total_combinations *= len(param_values)
                
                if total_combinations == 1:
                    # Single combination - set params directly and skip grid search
                    estimator.set_params(**{k: v[0] for k, v in params_grid.items()})
                    estimator.fit(X_train, y_train)
                    best_estimator = estimator
                    best_params = {k: v[0] for k, v in params_grid.items()}
                else:
                    grid = GridSearchCV(
                        estimator,
                        params_grid,
                        scoring="neg_mean_squared_error",
                        cv=3,
                        n_jobs=-1,
                    )
                    grid.fit(X_train, y_train)
                    best_estimator = grid.best_estimator_
                    best_params = grid.best_params_
            else:
                estimator.fit(X_train, y_train)
                best_estimator = estimator
                best_params = None

            predictions = best_estimator.predict(X_val)
            metrics = regression_metrics(y_val, predictions)
            artifact = SklearnModelArtifact(
                name=model_name,
                estimator=best_estimator,
                metrics=metrics,
                best_params=best_params,
                val_predictions=predictions,
            )
            self.artifacts[model_name] = artifact

        return self.artifacts

    def build_ensemble(self) -> Optional[SklearnModelArtifact]:
        """
        Create a voting regressor ensemble from trained base models.
        """
        if not self.artifacts:
            LOGGER.warning("No trained models to ensemble.")
            return None

        estimators = [
            (name, artifact.estimator)
            for name, artifact in self.artifacts.items()
            if name in {"random_forest", "gradient_boosting", "svr", "linear_regression"}
        ]
        if not estimators:
            return None

        ensemble = VotingRegressor(estimators=estimators)
        # Fit ensemble using stored training data if available
        # We'll reconstruct dataset from artifacts (assuming predictions computed on validation set only).
        # For robustness, return ensemble without additional fitting; user should call `fit_ensemble`.
        artifact = SklearnModelArtifact(
            name="ensemble",
            estimator=ensemble,
        )
        self.artifacts["ensemble"] = artifact
        return artifact

    def fit_ensemble(
        self,
        train_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> SklearnModelArtifact:
        target_column = target_column or CONFIG.features.target_column
        feature_columns = feature_columns or [
            col for col in train_df.columns if col != target_column
        ]

        ensemble_artifact = self.artifacts.get("ensemble") or self.build_ensemble()
        if not ensemble_artifact:
            raise ValueError("Unable to construct ensemble. Train base models first.")

        ensemble_artifact.estimator.fit(train_df[feature_columns], train_df[target_column])
        return ensemble_artifact

    def evaluate(
        self,
        test_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> Dict[str, MetricResult]:
        target_column = target_column or CONFIG.features.target_column
        feature_columns = feature_columns or [
            col for col in test_df.columns if col != target_column
        ]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        results: Dict[str, MetricResult] = {}
        for name, artifact in self.artifacts.items():
            try:
                y_pred = artifact.estimator.predict(X_test)
            except Exception as exc:
                LOGGER.error("Failed to evaluate %s: %s", name, exc)
                continue
            metrics = regression_metrics(y_test, y_pred)
            artifact.metrics = metrics
            artifact.test_predictions = y_pred
            results[name] = metrics
        return results

    def persist_model(
        self,
        model_name: str,
        path: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        artifact = self.artifacts.get(model_name)
        if not artifact:
            raise KeyError(f"Model {model_name} not trained.")
        metadata = metadata or {}
        metadata.setdefault("model_name", model_name)
        metadata.setdefault("best_params", artifact.best_params)
        save_sklearn_model(artifact.estimator, path, metadata)

    def _get_param_grid(self, model_name: str) -> Optional[Dict]:
        grids = {
            "random_forest": self.config.random_forest_params,
            "gradient_boosting": self.config.gradient_boosting_params,
            "svr": self.config.svr_params,
        }
        return grids.get(model_name)

