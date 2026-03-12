import math
import requests
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Boundaries and Reference Coordinates ───────────────────────────────────────
GALAPAGOS_BOUNDS = {
    "lat_min": -1.50,   # Sur  (Española)
    "lat_max":  1.70,   # Norte (Darwin Island)
    "lon_min": -92.10,  # Oeste (Darwin/Wolf)
    "lon_max": -89.20,  # Este  (San Cristóbal)
}

GALAPAGOS_ZONES = {
    "Santa Cruz":      {"lat": -0.7393, "lon": -90.3423, "desc": "Isla principal, Puerto Ayora"},
    "Isabela":         {"lat": -0.9758, "lon": -90.9667, "desc": "Isla más grande del archipiélago"},
    "San Cristóbal":   {"lat": -0.9020, "lon": -89.6100, "desc": "Capital del archipiélago"},
    "Fernandina":      {"lat": -0.3700, "lon": -91.5500, "desc": "Isla más joven, volcán activo"},
    "Santiago":        {"lat": -0.2300, "lon": -90.7600, "desc": "Isla Santiago / San Salvador"},
    "Española":        {"lat": -1.3600, "lon": -89.6800, "desc": "Isla más al sur"},
    "Genovesa":        {"lat":  0.3200, "lon": -89.9500, "desc": "Isla al norte, 'Isla del Pájaro'"},
    "Marchena":        {"lat":  0.3300, "lon": -90.4700, "desc": "Isla norte"},
    "Floreana":        {"lat": -1.2800, "lon": -90.4300, "desc": "Isla Charles, historia misteriosa"},
    "Darwin Island":   {"lat":  1.6800, "lon": -92.0000, "desc": "Isla más al norte"},
}

WMO_CODES = {
    0: "Despejado", 1: "Mayormente despejado", 2: "Parcialmente nublado",
    3: "Nublado", 45: "Niebla", 48: "Niebla con escarcha",
    51: "Llovizna leve", 53: "Llovizna moderada", 55: "Llovizna intensa",
    61: "Lluvia leve", 63: "Lluvia moderada", 65: "Lluvia intensa",
    71: "Nieve leve", 80: "Chubascos leves", 81: "Chubascos moderados",
    82: "Chubascos intensos", 95: "Tormenta", 99: "Tormenta con granizo"
}

# ── Math Utilities ────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two coordinates on earth."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def points_in_radius(lat: float, lon: float, radius_km: float, num_points: int = 8) -> list:
    """Generates `num_points` coordinates distributed in a circle around (lat, lon)."""
    R = 6371.0
    points = [{"lat": lat, "lon": lon, "label": "Centro"}]
    for i in range(num_points):
        angle = math.radians(360 / num_points * i)
        delta_lat = (radius_km / R) * math.cos(angle)
        delta_lon = (radius_km / R) * math.sin(angle) / math.cos(math.radians(lat))
        points.append({
            "lat": round(lat + math.degrees(delta_lat), 5),
            "lon": round(lon + math.degrees(delta_lon), 5),
            "label": f"Punto {i+1} ({int(math.degrees(angle))}°)"
        })
    return points

def create_grid(center_lat: float, center_lon: float, radius_km: float, cell_size_km: float = 20.0) -> list:
    """Creates a rectangular grid centered on (lat, lon) covering the radius."""
    R = 6371.0
    delta_lat = math.degrees(cell_size_km / R)
    # Use center_lat for longitude scaling approximation
    delta_lon = math.degrees(cell_size_km / (R * math.cos(math.radians(center_lat))))

    # Bounding box big enough to cover the radius
    lat_min = center_lat - math.degrees(radius_km / R)
    lat_max = center_lat + math.degrees(radius_km / R)
    lon_min = center_lon - math.degrees(radius_km / (R * math.cos(math.radians(center_lat))))
    lon_max = center_lon + math.degrees(radius_km / (R * math.cos(math.radians(center_lat))))

    grid = []
    row = 0
    lat = lat_min + delta_lat / 2
    while lat <= lat_max:
        col = 0
        lon = lon_min + delta_lon / 2
        while lon <= lon_max:
            grid.append({
                "cell_id": f"R{row:02d}C{col:02d}",
                "row": row, "col": col,
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "cell_size_km": cell_size_km
            })
            col += 1
            lon += delta_lon
        row += 1
        lat += delta_lat
    return grid

def get_cells_in_radius(lat: float, lon: float, radius_km: float, grid: list) -> list:
    """Returns grid cells whose center is within radius_km from (lat, lon)."""
    # Allowed globally, bounds check removed.
    cells = []
    for cell in grid:
        dist = haversine(lat, lon, cell["lat"], cell["lon"])
        if dist <= radius_km:
            cells.append({**cell, "distance_km": round(dist, 2)})

    return sorted(cells, key=lambda x: x["distance_km"])


# ── Weather Ingestion ─────────────────────────────────────────────────────────

def _get_google_weather_at(lat: float, lon: float) -> dict | None:
    """Fetch current weather from Google Weather API. Returns None if unavailable."""
    api_key = settings.GOOGLE_WEATHER_API_KEY
    if not api_key:
        return None
    try:
        resp = requests.get(
            "https://weather.googleapis.com/v1/currentConditions:lookup",
            params={
                "key": api_key,
                "location.latitude": lat,
                "location.longitude": lon,
                "languageCode": "es",
                "unitsSystem": "METRIC",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        temp_c = data.get("temperature", {}).get("degrees")
        feels_like_c = data.get("feelsLikeTemperature", {}).get("degrees")
        humidity = data.get("humidity")
        wind_kmh = data.get("wind", {}).get("speed", {}).get("value")
        precip_prob = data.get("precipitation", {}).get("probability", {}).get("percent")
        condition_txt = data.get("weatherCondition", {}).get("description", {}).get("text", "Desconocido")
        is_day = data.get("isDaytime", True)

        return {
            "temperature_c": temp_c,
            "feels_like_c": feels_like_c,
            "windspeed_kmh": round(wind_kmh, 1) if wind_kmh is not None else None,
            "humidity_pct": humidity,
            "precip_prob_pct": precip_prob,
            "condition": condition_txt,
            "is_day": is_day,
            "source": "google",
        }
    except Exception as e:
        logger.warning(f"Google Weather API failed for ({lat}, {lon}): {e}")
        return None


def _get_openmeteo_weather_at(lat: float, lon: float) -> dict:
    """Fetch current weather from OpenMeteo API (free, no key required)."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})

        temp_c = current.get("temperature_2m")
        feels_like_c = current.get("apparent_temperature")
        wind_ms = current.get("wind_speed_10m")
        wind_kmh = wind_ms * 3.6 if wind_ms else None
        humidity = current.get("relative_humidity_2m")
        precip_mm = current.get("precipitation")
        weather_code = current.get("weather_code", 0)
        condition_txt = WMO_CODES.get(weather_code, "Desconocido")
        is_day = bool(temp_c and temp_c > 20)

        return {
            "temperature_c": temp_c,
            "feels_like_c": feels_like_c,
            "windspeed_kmh": round(wind_kmh, 1) if wind_kmh else None,
            "humidity_pct": humidity,
            "precipitation_mm": precip_mm,
            "precip_prob_pct": None,
            "condition": condition_txt,
            "is_day": is_day,
            "source": "openmeteo",
        }
    except Exception as e:
        logger.warning(f"OpenMeteo failed for ({lat}, {lon}): {e}")
        return {
            "temperature_c": None, "feels_like_c": None,
            "windspeed_kmh": None, "humidity_pct": None,
            "precip_prob_pct": None, "condition": f"Error: {e}",
            "is_day": False, "source": "error",
        }


def get_weather_at(lat: float, lon: float) -> dict:
    """Fetch current weather. Uses Google Weather API if key is configured, falls back to OpenMeteo."""
    result = _get_google_weather_at(lat, lon)
    if result is not None:
        return result
    return _get_openmeteo_weather_at(lat, lon)

def get_weather_zone(zone_name: str, radius_km: float = 10, num_points: int = 4) -> dict:
    """Returns weather for center and radial points around a Galapagos island."""
    if zone_name not in GALAPAGOS_ZONES:
        return {"error": f"Zone '{zone_name}' does not exist. Options: {list(GALAPAGOS_ZONES.keys())}"}

    zone = GALAPAGOS_ZONES[zone_name]
    points = points_in_radius(zone["lat"], zone["lon"], radius_km, num_points)

    results = []
    for p in points:
        weather = get_weather_at(p["lat"], p["lon"])
        weather["label"] = p["label"]
        weather["lat"] = p["lat"]
        weather["lon"] = p["lon"]
        results.append(weather)

    return {
        "zone": zone_name,
        "description": zone["desc"],
        "center": {"lat": zone["lat"], "lon": zone["lon"]},
        "radius_km": radius_km,
        "num_sample_points": len(results),
        "weather_points": results
    }

def get_grid_weather(lat: float, lon: float, radius_km: float, cell_size_km: float = 20.0) -> dict:
    """Generates a grid, filters cells within radius, and queries weather per cell."""
    g = create_grid(center_lat=lat, center_lon=lon, radius_km=radius_km, cell_size_km=cell_size_km)
    cells = get_cells_in_radius(lat, lon, radius_km, g)

    if cells and "error" in cells[0]:
        return {"error": cells[0]["error"]}

    nearest_island = min(
        GALAPAGOS_ZONES.items(),
        key=lambda x: haversine(lat, lon, x[1]["lat"], x[1]["lon"])
    )

    weather_cells = []
    for cell in cells:
        w = get_weather_at(cell["lat"], cell["lon"])
        weather_cells.append({**cell, **w})

    return {
        "query": {"lat": lat, "lon": lon, "radius_km": radius_km, "cell_size_km": cell_size_km},
        "nearest_galapagos_zone": nearest_island[0],
        "nearest_galapagos_zone_dist_km": round(haversine(lat, lon, nearest_island[1]["lat"], nearest_island[1]["lon"]), 2),
        "cells_found": len(weather_cells),
        "grid": weather_cells
    }
