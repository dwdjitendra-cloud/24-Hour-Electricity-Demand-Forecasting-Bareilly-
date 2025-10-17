from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, y_col: str = "hourly_kwh") -> pd.DataFrame:
    X = df.copy()
    X["hour"] = X.index.hour
    X["dow"] = X.index.dayofweek
    # Sin/Cos for hour of day
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    # One-hot for day of week
    dummies = pd.get_dummies(X["dow"], prefix="dow", drop_first=False)
    X = pd.concat([X, dummies], axis=1)
    return X


def add_lags_and_rolling(
    df: pd.DataFrame,
    y_col: str = "hourly_kwh",
    lags: List[int] = [1, 2, 3],
    rolling_window: int = 24,
) -> pd.DataFrame:
    X = df.copy()
    for l in lags:
        X[f"lag_{l}"] = X[y_col].shift(l)
    X[f"roll_mean_{rolling_window}"] = X[y_col].rolling(rolling_window, min_periods=3).mean()
    return X


def make_feature_frame(
    df: pd.DataFrame,
    y_col: str = "hourly_kwh",
    temp_col: Optional[str] = None,
    lags: List[int] = [1, 2, 3],
    rolling_window: int = 24,
) -> Tuple[pd.DataFrame, pd.Series]:
    X = add_time_features(df, y_col)
    X = add_lags_and_rolling(X, y_col, lags, rolling_window)
    feat_cols = [
        "hour_sin",
        "hour_cos",
        *[c for c in X.columns if c.startswith("dow_")],
        *[f"lag_{l}" for l in lags],
        f"roll_mean_{rolling_window}",
    ]
    if temp_col and temp_col in X.columns:
        feat_cols.append(temp_col)
    Xy = X.dropna(subset=[*feat_cols, y_col]).copy()
    y = Xy[y_col]
    Xf = Xy[feat_cols]
    return Xf, y


def prepare_forecast_rows(
    hist_df: pd.DataFrame,
    horizons: int = 24,
    y_col: str = "hourly_kwh",
    temp_df: Optional[pd.DataFrame] = None,
    temp_col: str = "temperature_2m",
    lags: List[int] = [1, 2, 3],
    rolling_window: int = 24,
) -> pd.DataFrame:
    """
    Create a DataFrame skeleton for next 'horizons' hours with time features and optional temp.
    hist_df is expected to be hourly and continuous.
    """
    last_ts = hist_df.index[-1]
    future_idx = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizons, freq="h", tz=hist_df.index.tz)
    fut = pd.DataFrame(index=future_idx)
    fut["hourly_kwh"] = np.nan

    # Merge temperature if provided
    if temp_df is not None and temp_col in temp_df.columns:
        fut = fut.merge(temp_df[[temp_col]], left_index=True, right_index=True, how="left")

    # Add time features and placeholders for lags/rolling (filled during recursion)
    fut = add_time_features(fut, y_col)
    return fut
