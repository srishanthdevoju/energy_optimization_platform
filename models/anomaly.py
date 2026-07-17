"""
Anomaly detection using Isolation Forest on daily energy consumption.
Detects sudden spikes, unusual patterns, and potential appliance faults.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import ANOMALY_CONTAMINATION, SAVED_MODELS_DIR, TARGET_COLUMN

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Isolation Forest-based anomaly detector for energy consumption data."""

    FEATURE_COLS = [
        "energy_sum", "energy_mean", "energy_max", "energy_std", "energy_min",
    ]
    TEMPORAL_COLS = [
        "day_of_week", "month", "is_weekend",
    ]

    def __init__(self, contamination: float = ANOMALY_CONTAMINATION):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.used_features: List[str] = []

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and prepare feature matrix for the Isolation Forest."""
        df = df.copy()

        if "day_of_week" not in df.columns and "day" in df.columns:
            df["day_of_week"] = df["day"].dt.dayofweek
        if "month" not in df.columns and "day" in df.columns:
            df["month"] = df["day"].dt.month
        if "is_weekend" not in df.columns and "day_of_week" in df.columns:
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        available = [c for c in self.FEATURE_COLS + self.TEMPORAL_COLS if c in df.columns]
        self.used_features = available
        return df[available].fillna(0)

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the Isolation Forest on the provided data."""
        X = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        logger.info("Fitting Isolation Forest on %d samples, %d features", *X_scaled.shape)
        self.model.fit(X_scaled)
        self.is_trained = True
        logger.info("Isolation Forest training complete")

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run anomaly detection and return the DataFrame augmented with:
        - is_anomaly: boolean flag
        - anomaly_score: continuous score (lower = more anomalous)
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        result = df.copy()
        result["is_anomaly"] = predictions == -1
        result["anomaly_score"] = scores

        n_anomalies = result["is_anomaly"].sum()
        pct = n_anomalies / len(result) * 100
        logger.info("Detected %d anomalies (%.1f%%) out of %d records", n_anomalies, pct, len(result))
        return result

    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics about detected anomalies."""
        if "is_anomaly" not in df.columns:
            df = self.detect(df)

        anomalies = df[df["is_anomaly"]]
        normal = df[~df["is_anomaly"]]

        summary = {
            "total_records": len(df),
            "total_anomalies": int(anomalies["is_anomaly"].sum()),
            "anomaly_percentage": float(anomalies["is_anomaly"].sum() / len(df) * 100),
            "avg_anomaly_energy": float(anomalies[TARGET_COLUMN].mean()) if len(anomalies) > 0 else 0,
            "avg_normal_energy": float(normal[TARGET_COLUMN].mean()) if len(normal) > 0 else 0,
            "max_anomaly_energy": float(anomalies[TARGET_COLUMN].max()) if len(anomalies) > 0 else 0,
        }

        if "LCLid" in df.columns and len(anomalies) > 0:
            anomalies_per_hh = anomalies.groupby("LCLid").size()
            summary["households_with_anomalies"] = int(anomalies_per_hh.count())
            summary["max_anomalies_single_household"] = int(anomalies_per_hh.max())
            summary["top_anomaly_household"] = str(anomalies_per_hh.idxmax())

        if "day" in df.columns and len(anomalies) > 0:
            month_dist = anomalies["day"].dt.month.value_counts().to_dict()
            summary["anomalies_by_month"] = {int(k): int(v) for k, v in month_dist.items()}

            dow_dist = anomalies["day"].dt.day_name().value_counts().to_dict()
            summary["anomalies_by_day_of_week"] = dow_dist

        logger.info("Anomaly summary: %d anomalies (%.1f%%)", summary["total_anomalies"], summary["anomaly_percentage"])
        return summary

    def save(self, path: Optional[str] = None) -> str:
        """Serialize the detector to disk."""
        if path is None:
            path = str(SAVED_MODELS_DIR / "anomaly_detector.joblib")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "contamination": self.contamination,
            "used_features": self.used_features,
            "is_trained": self.is_trained,
        }, path)
        logger.info("Anomaly detector saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        """Load a previously saved detector."""
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.contamination = data["contamination"]
        instance.used_features = data["used_features"]
        instance.is_trained = data["is_trained"]
        logger.info("Anomaly detector loaded from %s", path)
        return instance
