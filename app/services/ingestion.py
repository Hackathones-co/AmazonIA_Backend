"""
Data ingestion — fetches live weather data from Open-Meteo API
and transforms it into the same format as the training station data.

Open-Meteo is free, no API key required, and provides 15-min resolution data
for any lat/lon including Galápagos.
"""
import logging
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

# Open-Meteo variables → our harmonized column names
# The notebook uses 4 stations; Open-Meteo gives us a single grid point.
# We replicate data across all station prefixes for model compatibility.
OPENMETEO_TO_STATION = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "rh_avg",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_dir",
    "precipitation": "rain_mm",
    "shortwave_radiation": "solar_kw",       # W/m² → we'll convert
    "soil_moisture_0_to_7cm": "soil_moisture_1",
    "soil_moisture_7_to_28cm": "soil_moisture_2",
    "soil_moisture_28_to_100cm": "soil_moisture_3",
}


async def fetch_openmeteo_history(hours_back: int = 48) -> pd.DataFrame:
    """Fetch recent weather data from Open-Meteo for San Cristóbal.

    Returns a wide DataFrame in the same format as the training data,
    with columns like {stn}_temp_c, {stn}_rain_mm, etc.
    """
    now = datetime.utcnow()
    start = (now - timedelta(hours=hours_back)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    params = {
        "latitude": settings.ISLAND_LAT,
        "longitude": settings.ISLAND_LON,
        "hourly": ",".join(OPENMETEO_TO_STATION.keys()),
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{settings.OPENMETEO_BASE}/forecast", params=params)
        resp.raise_for_status()
        data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    df = df.set_index("time")

    for om_var, our_var in OPENMETEO_TO_STATION.items():
        if om_var in hourly:
            df[our_var] = hourly[om_var]

    # Convert solar radiation W/m² → kW/m² (to match SlrkW_Avg)
    if "solar_kw" in df.columns:
        df["solar_kw"] = df["solar_kw"] / 1000.0

    # Upsample from hourly to 15-min (to match training resolution)
    df = df.resample("15min").interpolate(method="time")

    # Replicate across all station prefixes (Open-Meteo = single grid point)
    wide = pd.DataFrame(index=df.index)
    for stn in settings.STATIONS:
        for col in df.columns:
            wide[f"{stn}_{col}"] = df[col].values

        # Add rain_missing indicator (always 0 for API data)
        wide[f"{stn}_rain_missing"] = 0.0

        # Fill columns the model expects but Open-Meteo doesn't provide
        for missing_col in ["rh_max", "rh_min", "net_rad_wm2",
                            "leaf_wetness", "leaf_wet_minutes"]:
            full_col = f"{stn}_{missing_col}"
            if full_col not in wide.columns:
                if missing_col == "rh_max":
                    wide[full_col] = wide.get(f"{stn}_rh_avg", 70.0)
                elif missing_col == "rh_min":
                    wide[full_col] = wide.get(f"{stn}_rh_avg", 70.0) * 0.85
                else:
                    wide[full_col] = 0.0

    wide = wide.fillna(0.0)
    return wide


async def fetch_openmeteo_marine() -> dict:
    """Fetch marine data (wave height, period) for PescaSegura/MANGLE.

    Returns dict with current ocean conditions.
    """
    params = {
        "latitude": settings.ISLAND_LAT,
        "longitude": settings.ISLAND_LON,
        "hourly": "wave_height,wave_period,wave_direction,ocean_current_velocity",
        "timezone": "UTC",
        "forecast_days": 2,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://marine-api.open-meteo.com/v1/marine", params=params
            )
            resp.raise_for_status()
            data = resp.json()

        hourly = data["hourly"]
        times = pd.to_datetime(hourly["time"]).tz_localize(None)
        now_naive = pd.Timestamp.utcnow().replace(tzinfo=None)

        # Find the closest time index to now
        idx = (times - now_naive).abs().argmin()

        return {
            "wave_height_m": hourly.get("wave_height", [None])[idx],
            "wave_period_s": hourly.get("wave_period", [None])[idx],
            "wave_direction_deg": hourly.get("wave_direction", [None])[idx],
            "current_velocity_ms": hourly.get("ocean_current_velocity", [None])[idx],
            "timestamp": str(times[idx]),
        }
    except Exception as e:
        logger.warning(f"Marine API failed: {e}")
        return {
            "wave_height_m": None,
            "wave_period_s": None,
            "wave_direction_deg": None,
            "current_velocity_ms": None,
            "error": str(e),
        }
