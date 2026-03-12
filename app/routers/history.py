"""
GET /api/v1/history/{station}?hours=24

Returns historical weather data from OpenMeteo for a specific station.
"""
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.services.ingestion import fetch_openmeteo_history

router = APIRouter()

STATIONS = {
    "cer": "Cerro Alto",
    "jun": "El Junco",
    "merc": "Merceditas",
    "mira": "El Mirador",
}


@router.get("/history/{station}")
async def get_station_history(
    station: str,
    hours: int = Query(default=24, ge=1, le=168, description="Hours of historical data to fetch"),
):
    """Get historical weather data for a station.
    
    Returns time series data for temperature, humidity, wind, precipitation, etc.
    """
    if station not in STATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Station '{station}' not found. Valid stations: {list(STATIONS.keys())}",
        )
    
    try:
        # Fetch raw data from OpenMeteo for this specific station
        raw_df = await fetch_openmeteo_history(hours_back=hours, station=station)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch weather data: {exc}")
    
    if raw_df.empty:
        raise HTTPException(status_code=422, detail="Weather API returned empty dataset")
    
    # Extract station-specific columns
    station_cols = [col for col in raw_df.columns if col.startswith(f"{station}_")]
    station_df = raw_df[station_cols].copy()
    
    # Rename columns to remove station prefix
    station_df.columns = [col.replace(f"{station}_", "") for col in station_df.columns]
    
    # Convert to records for JSON response
    records = []
    for timestamp, row in station_df.iterrows():
        records.append({
            "timestamp": timestamp.isoformat(),
            "temp_c": round(float(row.get("temp_c", 0.0) or 0.0), 2),
            "rh_avg": round(float(row.get("rh_avg", 0.0) or 0.0), 2),
            "wind_speed_ms": round(float(row.get("wind_speed_ms", 0.0) or 0.0), 2),
            "wind_dir": round(float(row.get("wind_dir", 0.0) or 0.0), 2),
            "rain_mm": round(float(row.get("rain_mm", 0.0) or 0.0), 3),
            "solar_kw": round(float(row.get("solar_kw", 0.0) or 0.0), 3),
        })
    
    return {
        "station": station,
        "station_name": STATIONS[station],
        "hours": hours,
        "data_points": len(records),
        "data": records,
    }
