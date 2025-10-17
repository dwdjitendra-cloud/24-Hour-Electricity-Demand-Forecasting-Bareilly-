from typing import Optional, Tuple
import pandas as pd
import numpy as np

from .utils import get_logger

logger = get_logger(__name__)


POSSIBLE_TS_COLS = [
    "timestamp",
    "time",
    "datetime",
    "date",
    "Date",
    "interval_start",
]

POSSIBLE_VAL_COLS = [
    "value",
    "kwh",
    "energy_kwh",
    "consumption_kwh",
    "reading",
    "t_kWh",  # Added for this dataset
]


def _detect_columns(df: pd.DataFrame, ts_col: Optional[str], val_col: Optional[str]) -> Tuple[str, str]:
    if ts_col is None:
        for c in POSSIBLE_TS_COLS:
            if c in df.columns:
                ts_col = c
                break
    if ts_col is None:
        # Fallback to first column
        ts_col = df.columns[0]
        logger.warning(f"Timestamp column not found; defaulting to first column '{ts_col}'.")

    if val_col is None:
        for c in POSSIBLE_VAL_COLS:
            if c in df.columns:
                val_col = c
                break
    if val_col is None:
        # Fallback to last column
        val_col = df.columns[-1]
        logger.warning(f"Value column not found; defaulting to last column '{val_col}'.")

    return ts_col, val_col


def _synthetic_hourly_pattern(n_days: int = 1) -> pd.Series:
    """Generate typical hourly load curve for residential consumption"""
    # Patterns by weekday (0=Monday to 6=Sunday)
    weekday_patterns = {
        0: np.array([  # Monday
            0.5, 0.4, 0.3, 0.3, 0.3, 0.4,  # 00-06
            0.8, 1.4, 1.6, 1.4, 1.2, 1.1,  # 06-12
            1.0, 0.9, 0.9, 1.0, 1.1, 1.4,  # 12-18
            1.7, 1.9, 1.6, 1.3, 0.9, 0.7,  # 18-24
        ]),
        6: np.array([  # Sunday
            0.7, 0.6, 0.5, 0.4, 0.4, 0.5,  # 00-06
            0.6, 0.8, 1.2, 1.4, 1.3, 1.2,  # 06-12
            1.1, 1.0, 1.0, 1.1, 1.2, 1.4,  # 12-18
            1.8, 2.0, 1.7, 1.4, 1.0, 0.8,  # 18-24
        ])
    }
    # Default weekday pattern
    default_pattern = np.array([
        0.6, 0.5, 0.4, 0.4, 0.4, 0.5,  # 00-06
        0.7, 1.2, 1.5, 1.3, 1.1, 1.0,  # 06-12
        0.9, 0.8, 0.8, 0.9, 1.0, 1.3,  # 12-18
        1.6, 1.8, 1.5, 1.2, 0.9, 0.7,  # 18-24
    ])
    
    # Select pattern based on weekday
    weekday = pd.Timestamp.now().weekday()
    hourly_factors = weekday_patterns.get(weekday, default_pattern)
    
    if n_days == 1:
        return pd.Series(hourly_factors)
    
    # Repeat the pattern with some random variation
    patterns = []
    for _ in range(n_days):
        # Add ±10% random variation
        day_pattern = hourly_factors * (1 + np.random.uniform(-0.1, 0.1, size=24))
        patterns.append(day_pattern)
    return pd.Series(np.concatenate(patterns))


def load_and_resample(
    path: str,
    ts_col: Optional[str] = None,
    val_col: Optional[str] = None,
    tz: str = "Asia/Kolkata",
    gap_ffill_days: int = 2,
    clip_lower_q: float = 0.01,
    clip_upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Load daily data and generate synthetic hourly patterns while preserving daily totals.
    Returns DataFrame with column 'hourly_kwh' at hourly frequency.
    """
    df = pd.read_csv(path)
    ts_col, val_col = _detect_columns(df, ts_col, val_col)

    # Convert timestamps
    df["date"] = pd.to_datetime(df[ts_col]).dt.normalize()
    df = df.sort_values("date")
    
    # Group by date and sum in case of multiple meters
    daily = df.groupby("date")[val_col].sum().sort_index()
    
    # Get date range and handle gaps
    full_dates = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D", normalize=True)
    daily = daily.reindex(full_dates)
    
    # Forward fill gaps up to gap_ffill_days
    daily = daily.ffill(limit=gap_ffill_days)
    
    # For remaining gaps, use moving average of same weekday
    daily = daily.fillna(daily.groupby(daily.index.dayofweek).transform("mean"))
    
    # Generate hourly index
    hourly_idx = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max() + pd.Timedelta(days=1),
        freq="h",
        inclusive="left"
    )
    
    # Create hourly series
    hourly = pd.Series(index=hourly_idx, dtype=float)
    
    # Fill each day with synthetic pattern scaled to match daily total
    for day, total_kwh in daily.items():
        start_hr = day
        end_hr = day + pd.Timedelta(days=1)
        day_pattern = _synthetic_hourly_pattern()
        # Scale pattern to match daily total
        scaled_pattern = day_pattern * (total_kwh / day_pattern.sum())
        hourly.loc[start_hr:end_hr - pd.Timedelta(seconds=1)] = scaled_pattern.values
    
    # Handle timezone
    if hourly.index.tz is None:
        hourly.index = hourly.index.tz_localize(tz)
    else:
        hourly.index = hourly.index.tz_convert(tz)
    
    # Convert to DataFrame
    hourly_df = pd.DataFrame({"hourly_kwh": hourly})
    
    # Cap outliers
    q_low = hourly_df["hourly_kwh"].quantile(clip_lower_q)
    q_high = hourly_df["hourly_kwh"].quantile(clip_upper_q)
    hourly_df["hourly_kwh"] = hourly_df["hourly_kwh"].clip(lower=q_low, upper=q_high)
    
    return hourly_df
