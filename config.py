"""
Project configuration settings for the stock prediction system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def default_start_date(years: int = 5) -> str:
    """Return an ISO formatted start date that is `years` in the past."""
    return (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")


@dataclass
class DataConfig:
    tickers: List[str] = field(default_factory=lambda: ["AAPL"])  # Single ticker for faster runs
    start_date: str = field(default_factory=lambda: default_start_date(5))  # Reduced from 7 to 5 years
    end_date: str = field(default_factory=lambda: datetime.today().strftime("%Y-%m-%d"))
    interval: str = "1d"
    min_history_years: int = 3  # Reduced requirement
    cache_data: bool = True
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    adjust_dividends: bool = True
    drop_missing: bool = False
    fill_method: str = "ffill"


@dataclass
class SplitConfig:
    test_size: float = 0.2
    validation_size: float = 0.1
    time_based: bool = True
    rolling_window: Optional[int] = None


@dataclass
class FeatureConfig:
    technical_indicators: List[str] = field(
        default_factory=lambda: [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bollinger",
        ]  # Removed stochastic for speed
    )
    use_lag_features: bool = True
    lag_days: List[int] = field(default_factory=lambda: [1, 5, 10])  # Reduced from 4 to 3
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 21])  # Reduced from 4 to 3
    include_volume_indicators: bool = True
    seasonal_decompose: bool = False
    seasonal_period: int = 252  # roughly one trading year
    target_column: str = "Close"
    scaling_method: str = "standard"  # options: standard, minmax, robust


@dataclass
class LSTMConfig:
    lookback_window: int = 30  # Reduced from 60 to 30
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 5])  # Removed 30-day for speed
    units: List[int] = field(default_factory=lambda: [32, 32])  # Reduced from [64, 64]
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    batch_size: int = 64  # Increased batch size for faster training
    epochs: int = 10  # Reduced from 50 to 10
    learning_rate: float = 1e-3
    patience: int = 3  # Reduced from 5 to 3
    seed: int = 42
    checkpoint_dir: str = "models/checkpoints"


@dataclass
class TraditionalModelConfig:
    random_forest_params: Dict[str, List] = field(
        default_factory=lambda: {
            "n_estimators": [100],  # Single value, no grid search
            "max_depth": [10],
            "min_samples_leaf": [2],
        }
    )
    gradient_boosting_params: Dict[str, List] = field(
        default_factory=lambda: {
            "n_estimators": [100],  # Single value
            "learning_rate": [0.1],
            "max_depth": [3],
        }
    )
    svr_params: Dict[str, List] = field(
        default_factory=lambda: {
            "kernel": ["rbf"],  # Single kernel
            "C": [10.0],  # Single value
            "gamma": ["scale"],
            "epsilon": [0.1],
        }
    )
    baseline_models: List[str] = field(
        default_factory=lambda: ["linear_regression", "lasso"]
    )
    seed: int = 42


@dataclass
class ArimaConfig:
    difference_order: int = 1
    seasonal_difference: int = 0
    seasonal_period: int = 5
    max_p: int = 3  # Reduced from 5 to 3
    max_q: int = 3  # Reduced from 5 to 3
    information_criterion: str = "aic"
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    adf_pvalue_threshold: float = 0.05
    auto_diff: bool = True


@dataclass
class BacktestConfig:
    initial_investment: float = 100_000.0
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.02
    walk_forward_window: int = 126  # Reduced from 252 (half year instead of full year)
    rebalance_frequency: int = 10  # Increased from 5 to reduce number of backtests
    signal_threshold: float = 0.5
    use_confidence_intervals: bool = True


@dataclass
class VisualizationConfig:
    show_interactive: bool = True
    save_plots: bool = True
    output_dir: str = "visualization/outputs"


@dataclass
class LoggingConfig:
    level: str = "INFO"  # Changed from DEBUG to reduce log noise
    log_to_file: bool = True
    log_file: str = "logs/project.log"


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    traditional: TraditionalModelConfig = field(default_factory=TraditionalModelConfig)
    arima: ArimaConfig = field(default_factory=ArimaConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


CONFIG = ProjectConfig()

