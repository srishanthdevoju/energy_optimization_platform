"""
Data loader module — loads, parses, and merges all CSV data sources
into a single analysis-ready DataFrame.
"""

import logging
from typing import Optional, Set

import pandas as pd

from config import (
    DAILY_DATASET,
    WEATHER_DATASET,
    HOUSEHOLDS_DATASET,
    HOLIDAYS_DATASET,
    ACORN_DATASET,
    WEATHER_FEATURES,
    TOP_N_HOUSEHOLDS,
)

logger = logging.getLogger(__name__)


def load_daily_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load the daily smart meter dataset and parse dates."""
    path = path or str(DAILY_DATASET)
    logger.info("Loading daily dataset from %s", path)
    df = pd.read_csv(path)
    df["day"] = pd.to_datetime(df["day"], format="mixed", dayfirst=False)
    df["energy_sum"] = pd.to_numeric(df["energy_sum"], errors="coerce")
    logger.info("Daily dataset loaded: %s rows, %s columns", *df.shape)
    return df


def load_weather(path: Optional[str] = None) -> pd.DataFrame:
    """Load weather data with date extraction and numeric column selection."""
    path = path or str(WEATHER_DATASET)
    logger.info("Loading weather data from %s", path)
    df = pd.read_csv(path)
    df["day"] = pd.to_datetime(
        df["time"].astype(str).str.split(" ").str[0],
        format="mixed",
        dayfirst=False,
    )

    keep_cols = ["day"] + [c for c in WEATHER_FEATURES if c in df.columns]
    df = df[keep_cols].copy()

    for col in WEATHER_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["day"]).sort_values("day").reset_index(drop=True)
    logger.info("Weather data loaded: %s rows", len(df))
    return df


def load_households(path: Optional[str] = None) -> pd.DataFrame:
    """Load household information with ACORN classification."""
    path = path or str(HOUSEHOLDS_DATASET)
    logger.info("Loading household data from %s", path)
    df = pd.read_csv(path)
    logger.info("Household data loaded: %s households", len(df))
    return df


def load_holidays(path: Optional[str] = None) -> Set[pd.Timestamp]:
    """Load UK bank holidays and return a set of date objects."""
    path = path or str(HOLIDAYS_DATASET)
    logger.info("Loading bank holidays from %s", path)
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["Bank holidays"], format="mixed", dayfirst=False)
    holiday_set = set(dates.dt.date)
    logger.info("Bank holidays loaded: %s dates", len(holiday_set))
    return holiday_set


def load_acorn_details(path: Optional[str] = None) -> pd.DataFrame:
    """Load ACORN demographic details (latin1 encoded)."""
    path = path or str(ACORN_DATASET)
    logger.info("Loading ACORN details from %s", path)
    df = pd.read_csv(path, encoding="latin1")
    logger.info("ACORN details loaded: %s rows", len(df))
    return df


def sample_top_households(
    df: pd.DataFrame, n: int = TOP_N_HOUSEHOLDS
) -> pd.DataFrame:
    """Keep only the top-N households ranked by number of daily records."""
    counts = df.groupby("LCLid").size().nlargest(n)
    top_ids = set(counts.index)
    sampled = df[df["LCLid"].isin(top_ids)].copy()
    logger.info(
        "Sampled top %d households: %s rows (from %s)", n, len(sampled), len(df)
    )
    return sampled


def load_and_merge(
    sample: bool = True,
    n_households: int = TOP_N_HOUSEHOLDS,
) -> pd.DataFrame:
    """
    Load all data sources, merge them, and optionally sample households.

    Returns a merged DataFrame with daily energy, household info, and weather.
    """
    import os
    from config import DAILY_DATASET_SAMPLE

    if sample and os.path.exists(DAILY_DATASET_SAMPLE):
        logger.info("Loading pre-sampled daily dataset from %s", DAILY_DATASET_SAMPLE)
        df_daily = load_daily_dataset(str(DAILY_DATASET_SAMPLE))
        # Already pre-sampled to 500 households, no need to sample again
    else:
        df_daily = load_daily_dataset()
        if sample:
            df_daily = sample_top_households(df_daily, n=n_households)

    df_weather = load_weather()
    df_households = load_households()

    logger.info("Merging daily data with household info...")
    df = pd.merge(df_daily, df_households, on="LCLid", how="left")

    logger.info("Merging with weather data...")
    df = pd.merge(df, df_weather, on="day", how="left")

    df = df.sort_values(["LCLid", "day"]).reset_index(drop=True)
    logger.info("Final merged dataset: %s rows, %s columns", *df.shape)
    return df
