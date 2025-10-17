import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils import get_logger, ensure_dir, parse_history_window, compute_train_range, save_csv
from src.preprocess import load_and_resample
from src.model_baseline import forecast_seasonal_naive
from src.model_ridge import train_ridge, recursive_forecast
from src.features import make_feature_frame
from src.metrics import mae, wape, smape, mae_by_horizon
from src.plots import plot_forecast_overlay, plot_mae_by_horizon
from src.report import generate_pdf_report
from src.weather import fetch_open_meteo_temperature


ARTIFACT_DIR = os.path.join("artifacts", "fast_track")
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
FORECAST_PATH = os.path.join(ARTIFACT_DIR, "forecast_T_plus_24.csv")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.csv")
REPORT_PATH = os.path.join("reports", "fast_track_report.pdf")
DATA_PATH = os.path.join("data", "SM Cleaned Data BR Aggregated.csv")


logger = get_logger("fast_track_cli")


def evaluate_backtest(
    hourly: pd.DataFrame,
    anchor_ts: pd.Timestamp,
    with_weather: bool,
    city: str,
    history_window: str,
) -> List[Dict]:
    unit, amount = parse_history_window(history_window)
    train_start, train_end = compute_train_range(anchor_ts - pd.Timedelta(hours=1), unit, amount)

    # Train Ridge
    temp_df = None
    if with_weather:
        temp_df = fetch_open_meteo_temperature(city, start_ts=anchor_ts - pd.Timedelta(hours=1), hours=24)

    model, feat_cols = train_ridge(hourly, train_start, train_end, temp_col=("temperature_2m" if temp_df is not None else None))
    ridge_forecast = recursive_forecast(model, hourly.loc[:anchor_ts], horizons=24, temp_df=temp_df, temp_col="temperature_2m", feat_cols=feat_cols)

    # Baseline
    baseline_forecast = forecast_seasonal_naive(hourly.loc[:anchor_ts], horizons=24)

    # Ground truth
    y_true = hourly.loc[anchor_ts + pd.Timedelta(hours=1): anchor_ts + pd.Timedelta(hours=24), "hourly_kwh"].copy()

    # Align
    ridge_pred = ridge_forecast.loc[y_true.index]
    base_pred = baseline_forecast.loc[y_true.index]

    results = []
    for name, pred in [("ridge", ridge_pred), ("seasonal_naive", base_pred)]:
        m_mae = mae(y_true.values, pred.values)
        m_wape = wape(y_true.values, pred.values)
        m_smape = smape(y_true.values, pred.values)
        results.append({
            "anchor_time": anchor_ts.isoformat(),
            "model": name,
            "MAE": m_mae,
            "WAPE": m_wape,
            "sMAPE": m_smape,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="24-Hour Electricity Demand Forecasting")
    parser.add_argument("--with_weather", type=str, default="true", help="Use Open-Meteo temperature (true/false)")
    parser.add_argument("--city", type=str, default="Bareilly")
    parser.add_argument("--make_plots", type=str, default="true")
    parser.add_argument("--history_window", type=str, default="days:7")
    parser.add_argument("--save_report", type=str, default="true")
    args = parser.parse_args()

    with_weather = str(args.with_weather).lower() in ("1", "true", "yes", "y")
    make_plots = str(args.make_plots).lower() in ("1", "true", "yes", "y")
    save_report = str(args.save_report).lower() in ("1", "true", "yes", "y")
    city = args.city

    logger.info("Loading and preprocessing data…")
    hourly = load_and_resample(DATA_PATH)

    # Forecast next 24 hours (beyond last timestamp)
    logger.info("Building next-day forecasts…")
    temp_df_future = None
    if with_weather:
        temp_df_future = fetch_open_meteo_temperature(city, start_ts=hourly.index[-1], hours=24)
        if temp_df_future is None:
            logger.warning("Weather unavailable or misaligned — proceeding without temperature features.")

    # Train ridge on last 7 days up to T-1
    unit, amount = parse_history_window(args.history_window)
    train_start, train_end = compute_train_range(hourly.index[-1] - pd.Timedelta(hours=1), unit, amount)
    model, feat_cols = train_ridge(hourly, train_start, train_end, temp_col=("temperature_2m" if temp_df_future is not None else None))
    ridge_forecast = recursive_forecast(model, hourly, horizons=24, temp_df=temp_df_future, temp_col="temperature_2m", feat_cols=feat_cols)

    baseline_forecast = forecast_seasonal_naive(hourly, horizons=24)

    # Save forecast file (use ridge as primary; include baseline as column too for reference)
    ensure_dir(ARTIFACT_DIR)
    fc_df = pd.DataFrame({
        "timestamp": ridge_forecast.index,
        "yhat_ridge": ridge_forecast.values,
        "yhat_baseline": baseline_forecast.reindex(ridge_forecast.index).values,
    })
    save_csv(fc_df, FORECAST_PATH, index=False)

    # Backtests (T-1 and T-2 days)
    logger.info("Running light backtests (T-1 and T-2)…")
    anchors = [hourly.index[-25], hourly.index[-49]] if len(hourly) >= 49 else [hourly.index[-25]]
    metrics_rows = []
    horizon_mae_all = []
    for anc in anchors:
        results = evaluate_backtest(hourly, anc, with_weather, city, args.history_window)
        metrics_rows.extend(results)
        # Compute horizon-wise MAE using ridge for plot
        # Reproduce ridge forecast at anchor anc
        temp_df_bt = fetch_open_meteo_temperature(city, start_ts=anc - pd.Timedelta(hours=1), hours=24) if with_weather else None
        model_bt, feat_cols_bt = train_ridge(hourly, *compute_train_range(anc - pd.Timedelta(hours=1), unit, amount), temp_col=("temperature_2m" if temp_df_bt is not None else None))
        ridge_bt = recursive_forecast(model_bt, hourly.loc[:anc], horizons=24, temp_df=temp_df_bt, temp_col="temperature_2m", feat_cols=feat_cols_bt)
        y_true_bt = hourly.loc[anc + pd.Timedelta(hours=1): anc + pd.Timedelta(hours=24), "hourly_kwh"].copy()
        ridge_bt = ridge_bt.loc[y_true_bt.index]
        h_mae = mae_by_horizon(y_true_bt, ridge_bt)
        horizon_mae_all.append(h_mae)

    metrics_df = pd.DataFrame(metrics_rows)
    save_csv(metrics_df, METRICS_PATH, index=False)

    # Plots
    if make_plots:
        logger.info("Saving plots…")
        ensure_dir(PLOTS_DIR)
        plot_forecast_overlay(hourly, ridge_forecast, os.path.join(PLOTS_DIR, "forecast_overlay.png"))
        if horizon_mae_all:
            h_mae_avg = pd.concat(horizon_mae_all, axis=1).mean(axis=1)
            plot_mae_by_horizon(h_mae_avg, os.path.join(PLOTS_DIR, "mae_by_horizon.png"))

    # Report
    if save_report:
        logger.info("Generating PDF report…")
        ensure_dir(os.path.dirname(REPORT_PATH))
        context = {
            "title": f"24-Hour Electricity Demand Forecasting — {city}",
            "data_text": "Primary dataset: SM Cleaned Data BR Aggregated.csv at 3-minute resolution. Resampled to hourly (sum). Small gaps (<=2h) forward-filled. Outliers clipped at 1st–99th percentiles.",
            "weather_used": temp_df_future is not None,
            "metrics_rows": [
                [m["anchor_time"], m["model"], f"{m['MAE']:.3f}", f"{m['WAPE']:.3f}", f"{m['sMAPE']:.3f}"] for m in metrics_rows
            ],
        }
        generate_pdf_report(REPORT_PATH, context)

    logger.info("Done. Artifacts saved under artifacts/fast_track and reports/.")


if __name__ == "__main__":
    main()
