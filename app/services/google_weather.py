"""
Google Weather API service — fetches current conditions and 24h rain accumulation
for the 4 Galápagos weather stations.

Endpoints used (GET with query params — not POST):
  GET currentConditions:lookup?location.latitude=&location.longitude=&key=
  GET history/hours:lookup?location.latitude=&location.longitude=&hours=24&key=

Cache: in-memory, 15-minute TTL per station.
  4 stations × 2 endpoints × calls_per_15min = ~23,040/month at max frequency.
"""
import asyncio
import logging
import time
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_BASE = "https://weather.googleapis.com/v1"

STATIONS = {
    "cer":  {"name": "Cerro Alto",  "lat": -0.887048868, "lon": -89.53098555, "zone": "highland"},
    "jun":  {"name": "El Junco",    "lat": -0.896537076, "lon": -89.48162446, "zone": "highland"},
    "merc": {"name": "Merceditas",  "lat": -0.889712315, "lon": -89.44202039, "zone": "coastal"},
    "mira": {"name": "El Mirador",  "lat": -0.886247558, "lon": -89.53958685, "zone": "coastal"},
}

_cache: dict[str, dict] = {}
_CACHE_TTL = 900  # 15 minutes


def _is_fresh(station_id: str) -> bool:
    entry = _cache.get(station_id)
    return entry is not None and (time.time() - entry["ts"]) < _CACHE_TTL


def _base_params(lat: float, lon: float) -> dict:
    return {
        "key": settings.GOOGLE_WEATHER_API_KEY,
        "location.latitude": lat,
        "location.longitude": lon,
        "unitsSystem": "METRIC",
    }


async def _fetch_current(client: httpx.AsyncClient, lat: float, lon: float) -> dict[str, Any]:
    resp = await client.get(
        f"{_BASE}/currentConditions:lookup",
        params=_base_params(lat, lon),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


async def _fetch_history(client: httpx.AsyncClient, lat: float, lon: float, hours: int = 24) -> dict[str, Any]:
    params = {**_base_params(lat, lon), "hours": hours}
    resp = await client.get(
        f"{_BASE}/history/hours:lookup",
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_current(data: dict) -> dict:
    wind = data.get("wind", {})
    wind_dir = wind.get("direction", {})
    temp = data.get("temperature", {})
    condition = data.get("weatherCondition", {})
    precip = data.get("precipitation", {})
    precip_prob = precip.get("probability", {})

    return {
        "temp_c":          round(float(temp.get("degrees", 0.0)), 1),
        "rh_pct":          int(data.get("relativeHumidity", 0)),
        "wind_ms":         round(float(wind.get("speed", {}).get("value", 0.0)), 1),
        "wind_dir_deg":    int(wind_dir.get("degrees", 0)),
        "wind_cardinal":   wind_dir.get("cardinal", "N"),
        "condition_type":  condition.get("type", "UNKNOWN"),
        "condition_desc":  condition.get("description", {}).get("text", ""),
        "precip_prob_pct": int(precip_prob.get("percent", 0)),
        "uv_index":        int(data.get("uvIndex", 0)),
    }


def _parse_rain_24h(history_data: dict) -> float:
    """Sum QPF (mm) over the last 24 hours. Key is 'historyHours' in the real API."""
    total = 0.0
    for hour in history_data.get("historyHours", []):
        qpf = hour.get("precipitation", {}).get("qpf", {})
        quantity = qpf.get("quantity", 0.0)
        if quantity:
            total += float(quantity)
    return round(total, 2)


async def fetch_station(station_id: str) -> dict:
    if _is_fresh(station_id):
        return _cache[station_id]["data"]

    stn = STATIONS[station_id]
    lat, lon = stn["lat"], stn["lon"]

    try:
        async with httpx.AsyncClient() as client:
            current_raw = await _fetch_current(client, lat, lon)
            history_raw = None
            try:
                history_raw = await _fetch_history(client, lat, lon, hours=24)
            except Exception as e:
                logger.warning(f"[google_weather] history failed for {station_id}: {e}")

        result = _parse_current(current_raw)
        result["rain_24h_mm"] = _parse_rain_24h(history_raw) if history_raw else 0.0
        result["station_id"]   = station_id
        result["station_name"] = stn["name"]
        result["zone"]         = stn["zone"]

        _cache[station_id] = {"data": result, "ts": time.time()}
        return result

    except Exception as e:
        logger.error(f"[google_weather] fetch failed for {station_id}: {e}")
        return {
            "station_id":    station_id,
            "station_name":  stn["name"],
            "zone":          stn["zone"],
            "temp_c":        0.0,
            "rh_pct":        0,
            "wind_ms":       0.0,
            "wind_dir_deg":  0,
            "wind_cardinal": "N",
            "condition_type": "UNKNOWN",
            "condition_desc": "",
            "precip_prob_pct": 0,
            "uv_index":      0,
            "rain_24h_mm":   0.0,
            "error":         str(e),
        }


async def fetch_all_stations() -> dict[str, dict]:
    results = await asyncio.gather(*[fetch_station(s) for s in STATIONS])
    return {s: r for s, r in zip(STATIONS, results)}
