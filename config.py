"""
Central configuration for the AI-Powered Energy Analytics System.
All paths, hyperparameters, feature lists, and constants are defined here.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
SCRIPTS_DIR = BASE_DIR / "scripts"

SAVED_MODELS_DIR.mkdir(exist_ok=True)

DAILY_DATASET = DATA_DIR / "daily_dataset.csv"
DAILY_DATASET_SAMPLE = DATA_DIR / "daily_dataset_sample.csv"
WEATHER_DATASET = DATA_DIR / "weather_daily_darksky.csv"
HOUSEHOLDS_DATASET = DATA_DIR / "informations_households.csv"
HOLIDAYS_DATASET = DATA_DIR / "uk_bank_holidays.csv"
ACORN_DATASET = DATA_DIR / "acorn_details.csv"

TOP_N_HOUSEHOLDS = 500

TEMPORAL_FEATURES = [
    "day_of_week", "month", "day_of_month", "week_of_year",
    "is_weekend", "season", "quarter",
]

WEATHER_FEATURES = [
    "temperatureMax", "temperatureMin", "windSpeed",
    "humidity", "cloudCover", "pressure", "visibility", "dewPoint",
]

LAG_PERIODS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [7, 14, 30]

TARGET_COLUMN = "energy_sum"

SEASON_MAP = {
    12: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3,
}

SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 42,
    "n_jobs": -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_N_CLUSTERS = 4
CLUSTER_K_RANGE = range(2, 9)

ANOMALY_CONTAMINATION = 0.05

TEST_DAYS = 60
FORECAST_HORIZON_DEFAULT = 30

APP_TITLE = "⚡ AI-Powered Energy Analytics"
APP_ICON = "⚡"
PAGE_LAYOUT = "wide"

THEME_COLORS = {
    "primary": "#00D4AA",
    "secondary": "#7C3AED",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "success": "#10B981",
    "info": "#3B82F6",
    "background": "#0E1117",
    "surface": "#1E293B",
    "text": "#F1F5F9",
}

PLOTLY_TEMPLATE = "plotly_dark"
