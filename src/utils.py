import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd


def get_logger(name: str = "fast_track") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_history_window(window: str) -> Tuple[str, int]:
    """
    Parse history window like "days:7" or "hours:168".
    Returns unit and amount.
    """
    if not window or ":" not in window:
        return "days", 7
    unit, val = window.split(":", 1)
    try:
        amount = int(val)
    except ValueError:
        amount = 7
    return unit.lower(), amount


def compute_train_range(end_ts: pd.Timestamp, unit: str, amount: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if unit == "hours":
        start = end_ts - pd.Timedelta(hours=amount)
    else:
        start = end_ts - pd.Timedelta(days=amount)
    return start, end_ts


def to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Kolkata")
    return ts.tz_convert("Asia/Kolkata")


def ts_now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Kolkata")


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)


CITY_COORDS = {
    # Approximate coordinates for Indian cities
    "Bareilly": (28.367, 79.430),
}


def get_city_coords(city: str) -> Tuple[float, float]:
    if city in CITY_COORDS:
        return CITY_COORDS[city]
    # Default to Bareilly if unknown
    return CITY_COORDS["Bareilly"]
