import pandas as pd
import numpy as np
from datetime import timedelta
import os

from src.utils import get_logger, ensure_dir, parse_history_window, compute_train_range
from src.preprocess import load_and_resample
from src.model_baseline import forecast_seasonal_naive
from src.model_ridge import train_ridge, recursive_forecast
from src.metrics import mae, wape, smape
from src.weather import fetch_open_meteo_temperature

logger = get_logger("backtesting")

def run_rolling_backtest(
    hourly_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    city: str = "Bareilly",
    with_weather: bool = True,
    history_window: str = "days:7",
    horizons: int = 24,
) -> pd.DataFrame:
    """Run rolling backtest between start_date and end_date"""
    results = []
    current = start_date
    
    while current <= end_date:
        # Train window
        unit, amount = parse_history_window(history_window)
        train_start, train_end = compute_train_range(current - pd.Timedelta(hours=1), unit, amount)
        
        # Get weather if enabled
        temp_df = None
        if with_weather:
            temp_df = fetch_open_meteo_temperature(city, start_ts=current - pd.Timedelta(hours=1), hours=horizons)
        
        # Train Ridge
        model, feat_cols = train_ridge(
            hourly_df, 
            train_start, 
            train_end, 
            temp_col=("temperature_2m" if temp_df is not None else None)
        )
        ridge_forecast = recursive_forecast(
            model,
            hourly_df.loc[:current],
            horizons=horizons,
            temp_df=temp_df,
            temp_col="temperature_2m",
            feat_cols=feat_cols
        )
        
        # Baseline
        baseline_forecast = forecast_seasonal_naive(hourly_df.loc[:current], horizons=horizons)
        
        # Actual values for evaluation
        actuals = hourly_df.loc[
            current + pd.Timedelta(hours=1):
            current + pd.Timedelta(hours=horizons),
            "hourly_kwh"
        ]
        
        if len(actuals) == horizons:  # Only evaluate complete windows
            # Calculate metrics for each horizon
            for h in range(1, horizons + 1):
                h_actual = actuals.iloc[h-1]
                h_ridge = ridge_forecast.iloc[h-1]
                h_baseline = baseline_forecast.iloc[h-1]
                
                results.append({
                    "forecast_origin": current,
                    "horizon": h,
                    "actual": h_actual,
                    "ridge": h_ridge,
                    "baseline": h_baseline,
                    "ridge_ae": abs(h_actual - h_ridge),
                    "baseline_ae": abs(h_actual - h_baseline),
                })
        
        current += pd.Timedelta(days=1)
    
    return pd.DataFrame(results)

def main():
    # Load data
    logger.info("Loading data...")
    hourly = load_and_resample("data/SM Cleaned Data BR Aggregated.csv")
    
    # Define backtest period (last 30 days)
    end_date = hourly.index.max() - pd.Timedelta(days=1)  # Leave last day for final forecast
    start_date = end_date - pd.Timedelta(days=30)
    
    # Run backtest
    logger.info(f"Running rolling backtest from {start_date} to {end_date}...")
    results_df = run_rolling_backtest(
        hourly,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Save detailed results
    ensure_dir("results")
    results_df.to_csv("results/rolling_backtest_detailed.csv", index=False)
    
    # Calculate and save summary metrics
    summary = []
    
    # Overall metrics
    overall_metrics = {
        "scope": "overall",
        "horizon": "all",
        "ridge_mae": results_df["ridge_ae"].mean(),
        "baseline_mae": results_df["baseline_ae"].mean(),
        "ridge_wape": wape(results_df["actual"].values, results_df["ridge"].values),
        "baseline_wape": wape(results_df["actual"].values, results_df["baseline"].values),
        "ridge_smape": smape(results_df["actual"].values, results_df["ridge"].values),
        "baseline_smape": smape(results_df["actual"].values, results_df["baseline"].values),
    }
    summary.append(overall_metrics)
    
    # By horizon metrics
    for h in range(1, 25):
        h_data = results_df[results_df["horizon"] == h]
        horizon_metrics = {
            "scope": "by_horizon",
            "horizon": h,
            "ridge_mae": h_data["ridge_ae"].mean(),
            "baseline_mae": h_data["baseline_ae"].mean(),
            "ridge_wape": wape(h_data["actual"].values, h_data["ridge"].values),
            "baseline_wape": wape(h_data["actual"].values, h_data["baseline"].values),
            "ridge_smape": smape(h_data["actual"].values, h_data["ridge"].values),
            "baseline_smape": smape(h_data["actual"].values, h_data["baseline"].values),
        }
        summary.append(horizon_metrics)
    
    # By weekday metrics
    for weekday in range(7):
        weekday_data = results_df[results_df["forecast_origin"].dt.dayofweek == weekday]
        weekday_metrics = {
            "scope": "by_weekday",
            "horizon": f"weekday_{weekday}",
            "ridge_mae": weekday_data["ridge_ae"].mean(),
            "baseline_mae": weekday_data["baseline_ae"].mean(),
            "ridge_wape": wape(weekday_data["actual"].values, weekday_data["ridge"].values),
            "baseline_wape": wape(weekday_data["actual"].values, weekday_data["baseline"].values),
            "ridge_smape": smape(weekday_data["actual"].values, weekday_data["ridge"].values),
            "baseline_smape": smape(weekday_data["actual"].values, weekday_data["baseline"].values),
        }
        summary.append(weekday_metrics)
    
    pd.DataFrame(summary).to_csv("results/rolling_backtest_summary.csv", index=False)
    logger.info("Backtest results saved to results/")

if __name__ == "__main__":
    main()