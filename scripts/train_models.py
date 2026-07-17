"""
One-shot training script — trains all ML models and saves artifacts.

Run with:  python scripts/train_models.py
"""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SAVED_MODELS_DIR, TARGET_COLUMN
from preprocessing.data_loader import load_and_merge, load_holidays
from preprocessing.preprocessing import preprocess_pipeline
from preprocessing.feature_engineering import build_features, get_feature_columns
from models.forecasting import EnergyForecaster, time_based_split, compare_models
from models.clustering import HouseholdClusterer
from models.anomaly import AnomalyDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_models")


def main() -> None:
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preprocessing data")
    logger.info("=" * 60)

    holidays = load_holidays()
    df = load_and_merge(sample=True, n_households=500)
    df = preprocess_pipeline(df)
    df = build_features(df, holidays)

    logger.info("Final dataset: %d rows, %d columns", *df.shape)
    logger.info("Columns: %s", list(df.columns))

    feature_cols = get_feature_columns(df)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    logger.info("=" * 60)
    logger.info("STEP 2: Training forecasting models")
    logger.info("=" * 60)

    train_df, test_df = time_based_split(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COLUMN]

    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

    metrics_df, models = compare_models(X_train, y_train, X_test, y_test)
    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(metrics_df.to_string(index=False))
    print("=" * 50 + "\n")

    for name, model in models.items():
        model.save()

    metrics_df.to_csv(str(SAVED_MODELS_DIR / "model_comparison.csv"), index=False)

    best_model_name = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
    logger.info("Best model by RMSE: %s", best_model_name)

    with open(str(SAVED_MODELS_DIR / "best_model.txt"), "w") as f:
        f.write(best_model_name)

    logger.info("=" * 60)
    logger.info("STEP 3: Household clustering")
    logger.info("=" * 60)

    clusterer = HouseholdClusterer()
    profiles = clusterer.build_household_profiles(df)
    logger.info("Profiles built: %d households", len(profiles))

    k_values, inertias, silhouettes = clusterer.find_optimal_k()
    logger.info("Optimal K analysis complete")

    clusterer.fit(n_clusters=4)
    summary = clusterer.get_cluster_summary()
    print("\n" + "=" * 50)
    print("CLUSTER SUMMARY")
    print("=" * 50)
    print(summary.to_string(index=False))
    print("=" * 50 + "\n")

    clusterer.save()

    import pandas as pd
    elbow_df = pd.DataFrame({"k": k_values, "inertia": inertias, "silhouette": silhouettes})
    elbow_df.to_csv(str(SAVED_MODELS_DIR / "elbow_data.csv"), index=False)

    logger.info("=" * 60)
    logger.info("STEP 4: Anomaly detection")
    logger.info("=" * 60)

    detector = AnomalyDetector()
    detector.fit(df)

    df_with_anomalies = detector.detect(df)
    anomaly_summary = detector.get_anomaly_summary(df_with_anomalies)

    print("\n" + "=" * 50)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 50)
    for k, v in anomaly_summary.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print("=" * 50 + "\n")

    detector.save()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("ALL TRAINING COMPLETE in %.1f seconds", elapsed)
    logger.info("Models saved to: %s", SAVED_MODELS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
