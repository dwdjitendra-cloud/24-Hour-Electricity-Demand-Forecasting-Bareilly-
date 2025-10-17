from typing import Optional
import pandas as pd
import requests

from .utils import get_city_coords


def fetch_open_meteo_temperature(
    city: str,
    start_ts: pd.Timestamp,
    hours: int = 24,
    timezone: str = "Asia/Kolkata",
) -> Optional[pd.DataFrame]:
    """
    Fetch hourly temperature forecast for next 'hours' starting at ceil to next hour after 'start_ts'.
    Returns DataFrame indexed by timestamp with column 'temperature_2m'. If fails, returns None.
    """
    lat, lon = get_city_coords(city)
    # Open-Meteo returns from 'now'; we'll request next 3 days to be safe and then align.
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m&forecast_days=3&timezone={timezone}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        times = data.get("hourly", {}).get("time", [])
        temps = data.get("hourly", {}).get("temperature_2m", [])
        if not times or not temps or len(times) != len(temps):
            return None
        idx = pd.to_datetime(times)
        # Ensure timezone localized
        if getattr(idx, 'tz', None) is None:
            idx = idx.tz_localize(timezone)  # type: ignore
        else:
            idx = idx.tz_convert(timezone)  # type: ignore
        df = pd.DataFrame({"temperature_2m": temps}, index=idx)
        # Select only hours > start_ts and within range
        s1 = (start_ts.ceil("h") + pd.Timedelta(hours=1)).tz_convert(timezone)
        s2 = s1 + pd.Timedelta(hours=hours - 1)
        df = df.loc[s1:s2]
        if df.empty:
            return None
        return df
    except Exception:
        return None
