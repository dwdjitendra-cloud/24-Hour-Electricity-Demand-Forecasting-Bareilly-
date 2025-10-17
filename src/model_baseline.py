import pandas as pd


def forecast_seasonal_naive(hourly_df: pd.DataFrame, horizons: int = 24, y_col: str = "hourly_kwh") -> pd.Series:
    """
    Seasonal naive: y_hat_{t+h} = y_{t+h-24}
    Requires at least 24 hours of history.
    Returns a pd.Series indexed by future timestamps of length 'horizons'.
    """
    if len(hourly_df) < 24:
        raise ValueError("Need at least 24 hours of history for seasonal naive model.")
    last_ts = hourly_df.index[-1]
    future_idx = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizons, freq="h", tz=hourly_df.index.tz)
    # Align past last 24 hours to future
    past_24 = hourly_df[y_col].iloc[-24:]
    yhat = pd.Series(past_24.values[:horizons], index=future_idx)
    return yhat
