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

# Coordenadas específicas para cada estación en San Cristóbal
STATION_COORDS = {
    "jun": {"lat": -0.8833, "lon": -89.5167, "elevation": 548},  # El Junco (laguna, zona alta)
    "cer": {"lat": -0.8667, "lon": -89.5333, "elevation": 517},  # Cerro Alto (zona alta)
    "mira": {"lat": -0.8950, "lon": -89.6050, "elevation": 387}, # El Mirador (zona media)
    "merc": {"lat": -0.9000, "lon": -89.6167, "elevation": 100}, # Merceditas (zona baja/costera)
}

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


async def fetch_openmeteo_history(hours_back: int = 48, station: str | None = None) -> pd.DataFrame:
    """Fetch recent weather data from Open-Meteo for San Cristóbal.

    If station is provided, fetches data for that specific station's coordinates.
    Otherwise, fetches for all stations and returns a wide DataFrame.
    """
    now = datetime.utcnow()
    start = (now - timedelta(hours=hours_back)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    # If specific station requested, fetch only that station
    if station and station in STATION_COORDS:
        coords = STATION_COORDS[station]
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "hourly": ",".join(OPENMETEO_TO_STATION.keys()),
            "start_date": start,
            "end_date": end,
            "timezone": "UTC",
            "elevation": coords["elevation"],
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

        # Convert solar radiation W/m² → kW/m²
        if "solar_kw" in df.columns:
            df["solar_kw"] = df["solar_kw"] / 1000.0

        # Upsample from hourly to 15-min
        df = df.resample("15min").interpolate(method="time")

        # Create wide format with station prefix
        wide = pd.DataFrame(index=df.index)
        for col in df.columns:
            wide[f"{station}_{col}"] = df[col].values

        wide[f"{station}_rain_missing"] = 0.0

        # Fill missing columns
        for missing_col in ["rh_max", "rh_min", "net_rad_wm2", "leaf_wetness", "leaf_wet_minutes"]:
            full_col = f"{station}_{missing_col}"
            if full_col not in wide.columns:
                if missing_col == "rh_max":
                    wide[full_col] = wide.get(f"{station}_rh_avg", 70.0)
                elif missing_col == "rh_min":
                    wide[full_col] = wide.get(f"{station}_rh_avg", 70.0) * 0.85
                else:
                    wide[full_col] = 0.0

        wide = wide.fillna(0.0)
        return wide

    # Fetch data for all stations with their specific coordinates
    all_data = pd.DataFrame()
    
    for stn, coords in STATION_COORDS.items():
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "hourly": ",".join(OPENMETEO_TO_STATION.keys()),
            "start_date": start,
            "end_date": end,
            "timezone": "UTC",
            "elevation": coords["elevation"],
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

        # Convert solar radiation W/m² → kW/m²
        if "solar_kw" in df.columns:
            df["solar_kw"] = df["solar_kw"] / 1000.0

        # Upsample from hourly to 15-min
        df = df.resample("15min").interpolate(method="time")

        # Add to wide format
        if all_data.empty:
            all_data = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            all_data[f"{stn}_{col}"] = df[col].values

        all_data[f"{stn}_rain_missing"] = 0.0

        # Fill missing columns
        for missing_col in ["rh_max", "rh_min", "net_rad_wm2", "leaf_wetness", "leaf_wet_minutes"]:
            full_col = f"{stn}_{missing_col}"
            if full_col not in all_data.columns:
                if missing_col == "rh_max":
                    all_data[full_col] = all_data.get(f"{stn}_rh_avg", 70.0)
                elif missing_col == "rh_min":
                    all_data[full_col] = all_data.get(f"{stn}_rh_avg", 70.0) * 0.85
                else:
                    all_data[full_col] = 0.0

    all_data = all_data.fillna(0.0)
    return all_data


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
