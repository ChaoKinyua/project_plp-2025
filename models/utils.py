"""
Shared utilities for model training and persistence.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import CONFIG

LOGGER = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Ensure reproducibility across numpy, random, and tensorflow.
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate input/output sequences for recurrent models using sliding window.
    """
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i])
        y.append(targets[i + horizon - 1])
    return np.array(X), np.array(y)


def get_callbacks(model_name: str) -> List[tf.keras.callbacks.Callback]:
    """
    Compose default keras callbacks for training.
    """
    checkpoint_path = Path(CONFIG.lstm.checkpoint_dir) / f"{model_name}.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=CONFIG.lstm.patience,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, CONFIG.lstm.patience // 2),
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    return callbacks


def save_sklearn_model(model, path: str, metadata: Dict) -> None:
    """
    Persist scikit-learn models along with metadata.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta_path = Path(path).with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))


def load_sklearn_model(path: str):
    """
    Load scikit-learn model from disk.
    """
    return joblib.load(path)




