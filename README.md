# 24-Hour Electricity Demand Forecasting (Bareilly)

End-to-end, compact Python project to process daily smart-meter readings into synthetic hourly demand profiles and forecast the next 24 hours (H=24) for Bareilly. Includes baseline and Ridge regression models, optional weather integration from Open-Meteo API, evaluation metrics, visualization plots, and a 2-page PDF report.

Features:
- Intelligent daily → hourly disaggregation using typical load patterns
- Day-of-week specific hourly profiles (e.g., different patterns for weekdays vs weekends)
- Automatic handling of missing days and data quality issues
- Two complementary models with automated hyperparameter selection
- Professional reporting with key metrics and visualizations

## Quick Start

1. Clone the repository:
```powershell
git clone https://github.com/dwdjitendra-cloud/24-Hour-Electricity-Demand-Forecasting-Bareilly-.git
cd 24-Hour-Electricity-Demand-Forecasting-Bareilly-
```

2. Create/activate a Python 3.10+ environment

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

4. Download and prepare data:
   - Download `SM Cleaned Data BR Aggregated.csv` from the [v1.0-data release](https://github.com/dwdjitendra-cloud/24-Hour-Electricity-Demand-Forecasting-Bareilly-/releases/tag/v1.0-data)
   - Create a `data` directory in the project root if it doesn't exist
   - Place the downloaded file in the `data/` directory

## Project Structure

fast-track-electricity-forecast/
├── data/
│   └── SM Cleaned Data BR Aggregated.csv
├── src/
│   ├── preprocess.py — Load, clean, resample, handle gaps & outliers
│   ├── features.py — Create lag features, sin/cos, day-of-week, rolling stats
│   ├── model_baseline.py — Seasonal naive baseline
│   ├── model_ridge.py — Ridge regression with recursive multi-step
│   ├── metrics.py — MAE, WAPE/WMAPE, sMAPE
│   ├── plots.py — Overlay and horizon-wise MAE plots
│   ├── report.py — Auto-generate 2-page PDF report
│   └── utils.py — Helpers for logging, IO, CLI parsing
├── artifacts/
│   └── fast_track/
│       ├── forecast_T_plus_24.csv
│       ├── metrics.csv
│       └── plots/
│           ├── forecast_overlay.png
│           └── mae_by_horizon.png
├── reports/
│   └── fast_track_report.pdf
├── run_forecast.py — CLI entrypoint
├── requirements.txt
├── README.md
└── .gitignore

## Install Dependencies

Create/activate a Python 3.10+ environment, then install:

```powershell
pip install -r requirements.txt
```

## Run End-to-End Pipeline

Reproduce the full pipeline with one command:

```powershell
python run_forecast.py --with_weather true --city Bareilly --make_plots true --history_window days:7 --save_report true
```

This will:
- Load and clean data from `data/SM Cleaned Data BR Aggregated.csv` (daily kWh readings)
- Generate synthetic hourly profiles using typical residential load patterns
- Handle missing days with intelligent gap-filling (2-day forward fill + weekday averages)
- Train two models: Seasonal Naive and Ridge Regression
- Forecast next 24 hours (T+1…T+24)
- Light backtest at T−1 and T−2 days, save metrics CSV
- Save plots under `artifacts/fast_track/plots/`
- Generate a 2-page PDF report under `reports/`

## Technical Notes
- Hourly disaggregation uses typical residential load patterns with ±10% random variation
- Different hourly patterns for weekdays vs weekends to capture behavioral differences
- Weather integration via Open-Meteo API (gracefully skips if unavailable)
- All timestamps handled in IST (Asia/Kolkata) timezone
- Seasonal naive baseline uses last 24 hours
- Ridge model uses 7-day window by default (configurable via --history_window)
- Automatic outlier capping at 1st-99th percentiles

## Extended Analysis
For deeper model evaluation, run the rolling backtest script:

```powershell
python run_backtest.py
```

This performs a 30-day rolling backtest and saves detailed results under `results/`:
- `rolling_backtest_detailed.csv`: Full forecast vs actuals for each timestamp/horizon
- `rolling_backtest_summary.csv`: Aggregated metrics by:
  - Overall performance
  - Each forecast horizon (1-24 hours ahead)
  - Day of week patterns

## Example Outputs
- `artifacts/fast_track/forecast_T_plus_24.csv`: 24 rows with `timestamp, yhat_ridge, yhat_baseline`.
- `artifacts/fast_track/metrics.csv`: Backtest metrics (MAE, WAPE, sMAPE) for both models.
- `artifacts/fast_track/plots/*.png`: Overlay and horizon-wise MAE plots.
- `reports/fast_track_report.pdf`: Professional 2-page summary report.

## Data Sources

- **Dataset**: Smart-Meter Data from CEEW (Council on Energy, Environment and Water) - Contains daily electricity consumption readings from smart meters in Bareilly, India for the years 2020-2021.
- **Weather Data**: [Open-Meteo API](https://open-meteo.com/) - Provides historical and forecast weather data (optional integration).
