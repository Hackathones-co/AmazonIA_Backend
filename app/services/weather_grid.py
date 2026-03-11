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

def get_weather_at(lat: float, lon: float) -> dict:
    """Fetch current weather from Google Weather API for a specific coordinate."""
    api_key = settings.GOOGLE_WEATHER_API_KEY
    if not api_key:
        return {
            "temperature_c": None, "feels_like_c": None,
            "windspeed_kmh": None, "humidity_pct": None,
            "precip_prob_pct": None, "condition": "Error: GOOGLE_WEATHER_API_KEY no configurada",
            "is_day": False
        }
        
    try:
        resp = requests.get(
            "https://weather.googleapis.com/v1/currentConditions:lookup",
            params={
                "key": api_key,
                "location.latitude": lat,
                "location.longitude": lon,
            },
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Temperature mapping
        t_data = data.get("temperature", {})
        temp_c = t_data.get("degrees")
        
        fl_data = data.get("feelsLikeTemperature", {})
        feels_like_c = fl_data.get("degrees")
        
        # Wind mapping (converting km/h)
        wind_data = data.get("wind", {}).get("speed", {})
        wind_kmh = wind_data.get("value")
        
        # Humidity
        humidity = data.get("relativeHumidity")
        
        # Precipitation probability
        precip_data = data.get("precipitation", {}).get("probability", {})
        precip_prob = precip_data.get("percent")
        
        # Condition
        weather_cond = data.get("weatherCondition", {})
        condition_txt = weather_cond.get("description", {}).get("text", "Desconocido")
        
        # Day/Night
        is_day = data.get("isDaytime", True)
        
        return {
            "temperature_c": temp_c,
            "feels_like_c": feels_like_c,
            "windspeed_kmh": wind_kmh,
            "humidity_pct": humidity,
            "precip_prob_pct": precip_prob,
            "condition": condition_txt,
            "is_day": is_day
        }
    except Exception as e:
        logger.warning(f"Failed to fetch weather for {lat}, {lon}: {e}")
        return {
            "temperature_c": None, "feels_like_c": None,
            "windspeed_kmh": None, "humidity_pct": None,
            "precip_prob_pct": None, "condition": f"Error: {e}",
            "is_day": False
        }

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
