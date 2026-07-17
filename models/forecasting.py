"""
Energy consumption forecasting using XGBoost, LightGBM, and Random Forest.
Provides a unified interface for training, prediction, evaluation, and model comparison.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

from config import (
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    RANDOM_FOREST_PARAMS,
    SAVED_MODELS_DIR,
    TARGET_COLUMN,
    TEST_DAYS,
)
from utils.metrics import evaluate_all

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "xgboost": (xgb.XGBRegressor, XGBOOST_PARAMS),
    "lightgbm": (lgb.LGBMRegressor, LIGHTGBM_PARAMS),
    "random_forest": (RandomForestRegressor, RANDOM_FOREST_PARAMS),
}


class EnergyForecaster:
    """Train and serve energy consumption forecasts with a chosen algorithm."""

    def __init__(self, model_type: str = "xgboost"):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(MODEL_REGISTRY)}")
        self.model_type = model_type
        cls, params = MODEL_REGISTRY[model_type]
        self.model = cls(**params)
        self.is_trained = False
        self.feature_names: List[str] = []
        logger.info("EnergyForecaster initialized with %s", model_type)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model on training data."""
        self.feature_names = list(X_train.columns)
        logger.info("Training %s on %d samples with %d features", self.model_type, len(X_train), len(self.feature_names))
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")
        return self.model.predict(X[self.feature_names])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Predict on test data and compute MAE, RMSE, MAPE, R²."""
        predictions = self.predict(X_test)
        metrics = evaluate_all(y_test.values, predictions)
        logger.info("%s evaluation: %s", self.model_type, metrics)
        return metrics

    def get_feature_importance(self) -> Tuple[List[str], np.ndarray]:
        """Return feature names and their importance scores."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet")
        importances = self.model.feature_importances_
        return self.feature_names, importances

    def save(self, path: Optional[str] = None) -> str:
        """Serialize the model to disk."""
        if path is None:
            path = str(SAVED_MODELS_DIR / f"forecaster_{self.model_type}.joblib")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }, path)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> "EnergyForecaster":
        """Load a previously saved model."""
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.model_type = data["model_type"]
        instance.feature_names = data["feature_names"]
        instance.is_trained = data["is_trained"]
        logger.info("Model loaded from %s (%s)", path, instance.model_type)
        return instance


def time_based_split(
    df: pd.DataFrame,
    test_days: int = TEST_DAYS,
    date_col: str = "day",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically: the last `test_days` calendar days form the test set.
    """
    max_date = df[date_col].max()
    cutoff = max_date - pd.Timedelta(days=test_days)
    train = df[df[date_col] <= cutoff].copy()
    test = df[df[date_col] > cutoff].copy()
    logger.info(
        "Time-based split: train=%d rows (up to %s), test=%d rows (after %s)",
        len(train), cutoff.date(), len(test), cutoff.date(),
    )
    return train, test


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, EnergyForecaster]]:
    """
    Train all three model types and return a comparison DataFrame plus the fitted models.
    """
    results = []
    models = {}
    for model_type in MODEL_REGISTRY:
        logger.info("Comparing model: %s", model_type)
        forecaster = EnergyForecaster(model_type=model_type)
        forecaster.train(X_train, y_train)
        metrics = forecaster.evaluate(X_test, y_test)
        metrics["Model"] = model_type
        results.append(metrics)
        models[model_type] = forecaster

    metrics_df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "MAPE", "R2"]]
    logger.info("Model comparison:\n%s", metrics_df.to_string(index=False))
    return metrics_df, models
