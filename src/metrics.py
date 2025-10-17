from typing import Dict, Optional
import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom))


def mae_by_horizon(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    # y_true and y_pred must be aligned by timestamp
    diffs = (y_true - y_pred).abs()
    # Map horizon as 1..24 based on order after alignment
    horizons = range(1, len(diffs) + 1)
    return pd.Series(diffs.values, index=horizons)
