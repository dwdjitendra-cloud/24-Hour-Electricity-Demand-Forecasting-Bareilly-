from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import make_feature_frame, prepare_forecast_rows, add_lags_and_rolling, add_time_features


def train_ridge(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    temp_col: Optional[str] = None,
    lags: List[int] = [1, 2, 3],
    rolling_window: int = 24,
    y_col: str = "hourly_kwh",
) -> Tuple[Ridge, List[str]]:
    train_df = df.loc[train_start:train_end].copy()
    X_train, y_train = make_feature_frame(train_df, y_col=y_col, temp_col=temp_col, lags=lags, rolling_window=rolling_window)
    model = Ridge()
    model.fit(X_train, y_train)
    feat_cols = list(X_train.columns)
    return model, feat_cols


def recursive_forecast(
    model: Ridge,
    df_hist: pd.DataFrame,
    horizons: int = 24,
    temp_df: Optional[pd.DataFrame] = None,
    temp_col: str = "temperature_2m",
    lags: List[int] = [1, 2, 3],
    rolling_window: int = 24,
    y_col: str = "hourly_kwh",
    feat_cols: Optional[List[str]] = None,
) -> pd.Series:
    # Build future frame
    fut = prepare_forecast_rows(df_hist, horizons=horizons, y_col=y_col, temp_df=temp_df, temp_col=temp_col, lags=lags, rolling_window=rolling_window)

    # Start with a copy of history to update lags/rolling during recursion
    sim = df_hist[[y_col]].copy()

    preds = []
    for ts in fut.index:
        # For each step, construct feature row from current sim state and any exogenous vars
        row = pd.DataFrame(index=[ts])
        row["hourly_kwh"] = np.nan
        if temp_df is not None and temp_col in temp_df.columns and ts in temp_df.index:
            row[temp_col] = temp_df.loc[ts, temp_col]
        row = add_time_features(row, y_col)
        # Append row to sim to compute lags/rolling
        tmp = pd.concat([sim, row[[c for c in row.columns if c != y_col]]], axis=1)
        tmp = add_lags_and_rolling(tmp, y_col=y_col, lags=lags, rolling_window=rolling_window)
        # Take the last row as feature
        X_row = tmp.iloc[[-1]].copy()
        # Select known feature columns from training
        if feat_cols is None:
            feat_cols = [
                "hour_sin",
                "hour_cos",
                *[c for c in X_row.columns if c.startswith("dow_")],
                *[f"lag_{l}" for l in lags],
                f"roll_mean_{rolling_window}",
            ]
            if temp_df is not None and temp_col in X_row.columns:
                feat_cols.append(temp_col)
        # Align columns strictly to training feature set; add any missing with zeros
        for c in feat_cols:
            if c not in X_row.columns:
                X_row[c] = 0.0
        X_row = X_row[feat_cols]
        # For first few horizons, some lags may be NaN if insufficient history; fall back to last known value
        if X_row.isna().any(axis=None):
            X_row = X_row.fillna(method="ffill", axis=1).fillna(0.0)
        yhat = float(model.predict(X_row)[0])
        preds.append(yhat)
        # Update sim with predicted value at ts
        sim.loc[ts, y_col] = yhat

    return pd.Series(preds, index=fut.index, name="yhat")
