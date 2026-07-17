import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TARGET_COLUMN
from preprocessing.data_loader import load_and_merge, load_holidays
from preprocessing.preprocessing import preprocess_pipeline
from preprocessing.feature_engineering import build_features, get_feature_columns
from models.forecasting import time_based_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tune_models")

def main():
    start_time = time.time()
    logger.info("Loading and preprocessing data...")
    holidays = load_holidays()
    df = load_and_merge(sample=True, n_households=300)
    df = preprocess_pipeline(df)
    df = build_features(df, holidays)
    
    feature_cols = get_feature_columns(df)
    train_df, test_df = time_based_split(df)
    
    train_df = train_df.sample(frac=0.1, random_state=42)
    test_df = test_df.sample(frac=0.1, random_state=42)
    
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COLUMN].values
    X_val = test_df[feature_cols].values
    y_val = test_df[TARGET_COLUMN].values
    
    X = np.vstack((X_train, X_val))
    y = np.concatenate((y_train, y_val))
    
    split_index = np.zeros(X.shape[0])
    split_index[:len(X_train)] = -1
    split_index[len(X_train):] = 0
    pds = PredefinedSplit(test_fold=split_index)
    
    xgb_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.07, 0.15],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9]
    }
    
    lgb_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.07, 0.15],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "verbose": [-1]
    }
    
    rf_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [8, 12, 16],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 5]
    }
    
    models = {
        "xgboost": (xgb.XGBRegressor(random_state=42, n_jobs=-1), xgb_grid),
        "lightgbm": (lgb.LGBMRegressor(random_state=42, n_jobs=-1), lgb_grid),
        "random_forest": (RandomForestRegressor(random_state=42, n_jobs=-1), rf_grid)
    }
    
    for name, (model, grid) in models.items():
        logger.info(f"Tuning {name}...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=grid,
            n_iter=3,
            cv=pds,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=1
        )
        search.fit(X, y)
        logger.info(f"Best parameters for {name}: {search.best_params_}")
        logger.info(f"Best validation MAE: {-search.best_score_:.4f}")
        
    logger.info(f"Tuning completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
