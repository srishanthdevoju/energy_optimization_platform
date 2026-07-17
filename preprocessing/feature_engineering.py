"""
Feature engineering — temporal, lag, rolling, and calendar features
for energy consumption modelling.
"""

import logging
from typing import List, Optional, Set

import pandas as pd
import numpy as np

from config import (
    LAG_PERIODS,
    ROLLING_WINDOWS,
    SEASON_MAP,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the 'day' column."""
    df = df.copy()
    df["day_of_week"] = df["day"].dt.dayofweek
    df["day_of_month"] = df["day"].dt.day
    df["month"] = df["day"].dt.month
    df["week_of_year"] = df["day"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["day"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["season"] = df["month"].map(SEASON_MAP)
    logger.info("Temporal features added")
    return df


def add_holiday_features(
    df: pd.DataFrame, holidays: Set
) -> pd.DataFrame:
    """Add binary flag for UK bank holidays."""
    df = df.copy()
    df["is_bank_holiday"] = df["day"].dt.date.isin(holidays).astype(int)
    n_holiday_rows = df["is_bank_holiday"].sum()
    logger.info("Holiday feature added: %d rows flagged as bank holiday", n_holiday_rows)
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    target: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Create lagged versions of the target variable per household."""
    lags = lags or LAG_PERIODS
    df = df.copy()
    for lag in lags:
        col_name = f"{target}_lag_{lag}"
        df[col_name] = df.groupby("LCLid")[target].shift(lag)
    logger.info("Lag features added: %s", [f"lag_{l}" for l in lags])
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    target: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Create rolling mean and rolling std per household."""
    windows = windows or ROLLING_WINDOWS
    df = df.copy()
    for window in windows:
        mean_col = f"{target}_rolling_mean_{window}"
        std_col = f"{target}_rolling_std_{window}"
        grouped = df.groupby("LCLid")[target]
        df[mean_col] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[std_col] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    logger.info("Rolling features added: windows=%s", windows)
    return df


def build_features(
    df: pd.DataFrame,
    holidays: Optional[Set] = None,
) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps.
    The DataFrame must already have 'day' and 'LCLid' columns.
    """
    logger.info("Starting feature engineering on %d rows", len(df))
    df = add_temporal_features(df)

    if holidays is not None:
        df = add_holiday_features(df, holidays)
    else:
        df["is_bank_holiday"] = 0

    df = add_lag_features(df)
    df = add_rolling_features(df)

    n_nulls_before = df.isna().sum().sum()
    df = df.dropna(subset=[f"{TARGET_COLUMN}_lag_{max(LAG_PERIODS)}"])
    n_nulls_after = df.isna().sum().sum()
    logger.info(
        "Feature engineering complete: %d rows, dropped %d rows with insufficient history",
        len(df),
        n_nulls_before - n_nulls_after,
    )

    float_cols = df.select_dtypes(include=[np.number]).columns
    df[float_cols] = df[float_cols].fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of feature column names used for modelling."""
    exclude = {
        "LCLid", "day", TARGET_COLUMN,
        "stdorToU", "Acorn", "Acorn_grouped", "file",
        "energy_median", "energy_mean", "energy_max",
        "energy_count", "energy_std", "energy_min",
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.int32, np.float32]]
