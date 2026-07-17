"""
Household clustering using K-Means on aggregated consumption profiles.
Identifies distinct usage patterns: high consumers, low consumers, weekend-heavy, etc.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_N_CLUSTERS,
    CLUSTER_K_RANGE,
    SAVED_MODELS_DIR,
    TARGET_COLUMN,
    SEASON_MAP,
)

logger = logging.getLogger(__name__)


class HouseholdClusterer:
    """K-Means clustering on per-household consumption profiles."""

    def __init__(self, n_clusters: int = DEFAULT_N_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.profiles: Optional[pd.DataFrame] = None
        self.labels: Optional[np.ndarray] = None
        self.cluster_summary: Optional[pd.DataFrame] = None
        self.profile_columns: List[str] = []

    def build_household_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-household statistics into a profile vector:
        - mean, std, max, min of daily energy
        - weekend_ratio: mean weekend consumption / mean weekday consumption
        - peak_month_energy: max monthly average
        - seasonal_variation: std across seasonal averages
        - reading_count: total number of daily records
        """
        df = df.copy()
        df["is_weekend"] = (df["day"].dt.dayofweek >= 5).astype(int)
        df["month"] = df["day"].dt.month
        df["season"] = df["month"].map(SEASON_MAP)

        agg = df.groupby("LCLid").agg(
            mean_energy=(TARGET_COLUMN, "mean"),
            std_energy=(TARGET_COLUMN, "std"),
            max_energy=(TARGET_COLUMN, "max"),
            min_energy=(TARGET_COLUMN, "min"),
            median_energy=(TARGET_COLUMN, "median"),
            reading_count=(TARGET_COLUMN, "count"),
        ).reset_index()

        weekend_avg = (
            df[df["is_weekend"] == 1]
            .groupby("LCLid")[TARGET_COLUMN]
            .mean()
            .rename("weekend_avg")
        )
        weekday_avg = (
            df[df["is_weekend"] == 0]
            .groupby("LCLid")[TARGET_COLUMN]
            .mean()
            .rename("weekday_avg")
        )
        agg = agg.merge(weekend_avg, on="LCLid", how="left")
        agg = agg.merge(weekday_avg, on="LCLid", how="left")
        agg["weekend_ratio"] = agg["weekend_avg"] / agg["weekday_avg"].replace(0, np.nan)
        agg["weekend_ratio"] = agg["weekend_ratio"].fillna(1.0)

        seasonal_avg = (
            df.groupby(["LCLid", "season"])[TARGET_COLUMN]
            .mean()
            .unstack(fill_value=0)
        )
        agg["seasonal_variation"] = seasonal_avg.std(axis=1).values
        agg["peak_season_energy"] = seasonal_avg.max(axis=1).values

        monthly_avg = (
            df.groupby(["LCLid", "month"])[TARGET_COLUMN]
            .mean()
            .unstack(fill_value=0)
        )
        agg["peak_month_energy"] = monthly_avg.max(axis=1).values

        agg = agg.drop(columns=["weekend_avg", "weekday_avg"], errors="ignore")
        agg = agg.fillna(0)

        self.profiles = agg
        self.profile_columns = [
            c for c in agg.columns if c not in ("LCLid", "reading_count")
        ]
        logger.info("Built profiles for %d households with %d features", len(agg), len(self.profile_columns))
        return agg

    def find_optimal_k(
        self, k_range: Optional[range] = None
    ) -> Tuple[List[int], List[float], List[float]]:
        """Compute inertia and silhouette scores for a range of K values."""
        if self.profiles is None:
            raise RuntimeError("Call build_household_profiles() first")

        k_range = k_range or CLUSTER_K_RANGE
        X = self.scaler.fit_transform(self.profiles[self.profile_columns])

        k_values, inertias, silhouettes = [], [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            k_values.append(k)
            inertias.append(km.inertia_)
            sil = silhouette_score(X, km.labels_) if k > 1 else 0
            silhouettes.append(sil)
            logger.info("K=%d, inertia=%.2f, silhouette=%.4f", k, km.inertia_, sil)

        return k_values, inertias, silhouettes

    def fit(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """Fit K-Means on the household profiles."""
        if self.profiles is None:
            raise RuntimeError("Call build_household_profiles() first")

        if n_clusters is not None:
            self.n_clusters = n_clusters

        X = self.scaler.fit_transform(self.profiles[self.profile_columns])
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(X)
        self.profiles["cluster"] = self.labels

        sil = silhouette_score(X, self.labels)
        logger.info("K-Means fitted with K=%d, silhouette=%.4f", self.n_clusters, sil)

        self._label_clusters()
        return self.labels

    def _label_clusters(self) -> None:
        """Heuristically assign descriptive labels to clusters."""
        if self.profiles is None or "cluster" not in self.profiles.columns:
            return

        summary = self.profiles.groupby("cluster")[self.profile_columns].mean()

        sorted_clusters = summary["mean_energy"].sort_values()

        label_map: Dict[int, str] = {}
        cluster_list = sorted_clusters.index.tolist()

        label_map[cluster_list[-1]] = "High Consumers"
        label_map[cluster_list[0]] = "Low Consumers"

        remaining = [c for c in cluster_list if c not in label_map]
        if remaining:
            weekend_ratios = summary.loc[remaining, "weekend_ratio"]
            top_weekend = weekend_ratios.idxmax()
            label_map[top_weekend] = "Weekend-Heavy"
            remaining.remove(top_weekend)

        for c in remaining:
            label_map[c] = "Balanced"

        self.profiles["cluster_label"] = self.profiles["cluster"].map(label_map)
        logger.info("Cluster labels assigned: %s", label_map)

    def get_cluster_summary(self) -> pd.DataFrame:
        """Return per-cluster mean statistics with labels."""
        if self.profiles is None or "cluster" not in self.profiles.columns:
            raise RuntimeError("Call fit() first")

        summary = self.profiles.groupby("cluster")[self.profile_columns].mean().reset_index()
        counts = self.profiles["cluster"].value_counts().reset_index()
        counts.columns = ["cluster", "count"]
        summary = summary.merge(counts, on="cluster")

        label_map = dict(
            zip(self.profiles["cluster"], self.profiles["cluster_label"])
        )
        summary["cluster_label"] = summary["cluster"].map(label_map)
        self.cluster_summary = summary
        return summary

    def save(self, path: Optional[str] = None) -> str:
        """Serialize the clusterer to disk."""
        if path is None:
            path = str(SAVED_MODELS_DIR / "clusterer.joblib")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "profiles": self.profiles,
            "labels": self.labels,
            "n_clusters": self.n_clusters,
            "profile_columns": self.profile_columns,
            "cluster_summary": self.cluster_summary,
        }, path)
        logger.info("Clusterer saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> "HouseholdClusterer":
        """Load a previously saved clusterer."""
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.kmeans = data["kmeans"]
        instance.scaler = data["scaler"]
        instance.profiles = data["profiles"]
        instance.labels = data["labels"]
        instance.n_clusters = data["n_clusters"]
        instance.profile_columns = data["profile_columns"]
        instance.cluster_summary = data.get("cluster_summary")
        logger.info("Clusterer loaded from %s", path)
        return instance
