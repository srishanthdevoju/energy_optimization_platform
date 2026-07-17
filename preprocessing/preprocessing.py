"""
Data preprocessing — missing value handling, duplicate removal, and type coercion.
"""

import logging
from typing import List, Optional

import pandas as pd
import numpy as np

from config import TARGET_COLUMN

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Drop rows where the target column is NaN
    - Forward-fill numeric columns per household, then fill remaining with median
    - Fill categorical NaNs with 'Unknown'
    """
    initial_len = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    dropped = initial_len - len(df)
    if dropped:
        logger.info("Dropped %d rows with missing target values", dropped)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if "LCLid" in df.columns:
        df[numeric_cols] = df.groupby("LCLid")[numeric_cols].transform(
            lambda g: g.ffill().bfill()
        )

    for col in numeric_cols:
        remaining_nulls = df[col].isna().sum()
        if remaining_nulls > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Filled %d NaNs in '%s' with median %.4f", remaining_nulls, col, median_val)

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    total_nulls = df.isna().sum().sum()
    logger.info("Missing value handling complete. Remaining NaNs: %d", total_nulls)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (LCLid, day) pairs, keeping the first occurrence."""
    if "LCLid" in df.columns and "day" in df.columns:
        initial_len = len(df)
        df = df.drop_duplicates(subset=["LCLid", "day"], keep="first")
        removed = initial_len - len(df)
        if removed:
            logger.info("Removed %d duplicate rows", removed)
    else:
        initial_len = len(df)
        df = df.drop_duplicates()
        removed = initial_len - len(df)
        if removed:
            logger.info("Removed %d duplicate rows (full-row dedup)", removed)
    return df.reset_index(drop=True)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes for key columns."""
    if "day" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["day"]):
        df["day"] = pd.to_datetime(df["day"], format="mixed", dayfirst=False)

    float_cols = [
        "energy_median", "energy_mean", "energy_max", "energy_std",
        "energy_sum", "energy_min",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "energy_count" in df.columns:
        df["energy_count"] = pd.to_numeric(df["energy_count"], errors="coerce")

    return df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all preprocessing steps in sequence."""
    logger.info("Starting preprocessing pipeline on %d rows", len(df))
    df = coerce_types(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    logger.info("Preprocessing complete: %d rows remaining", len(df))
    return df
