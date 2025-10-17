from typing import Optional
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import ensure_dir

sns.set(style="whitegrid")


def plot_forecast_overlay(
    hourly_df: pd.DataFrame,
    forecast: pd.Series,
    save_path: str,
    y_col: str = "hourly_kwh",
    lookback_hours: int = 72,
) -> None:
    ensure_dir(os.path.dirname(save_path))
    last_ts = hourly_df.index[-1]
    start = last_ts - pd.Timedelta(hours=lookback_hours - 1)
    recent = hourly_df.loc[start:]
    plt.figure(figsize=(12, 4))
    plt.plot(recent.index, recent[y_col], label="Actual (last 3 days)")
    plt.plot(forecast.index, forecast.values, label="Forecast (next 24h)", linestyle="--")
    plt.title("Forecast Overlay: Last 3 days actuals and next 24h forecast")
    plt.xlabel("Time (IST)")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_mae_by_horizon(h_mae: pd.Series, save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(10, 4))
    h_mae.plot(kind="bar", color="#1f77b4")
    plt.title("MAE by Forecast Horizon (hours 1–24)")
    plt.xlabel("Horizon (hours ahead)")
    plt.ylabel("MAE (kWh)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
