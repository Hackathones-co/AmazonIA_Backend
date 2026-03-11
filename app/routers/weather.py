from fastapi import APIRouter, HTTPException, Query
from app.services.weather_grid import get_grid_weather, get_weather_zone, GALAPAGOS_ZONES

router = APIRouter()

@router.get("/weather/zones")
def list_zones():
    """Lista todas las zonas/islas disponibles con sus coordenadas."""
    return GALAPAGOS_ZONES

@router.get("/weather/zone/{zone_name}")
def weather_by_zone(
    zone_name: str,
    radius_km: float = Query(default=10.0, ge=1, le=200, description="Radio en kilómetros"),
    points: int = Query(default=4, ge=1, le=16, description="Número de puntos de muestreo")
):
    """
    Clima en una zona de Galápagos.
    - `zone_name`: nombre de la isla (ej: Santa Cruz, Isabela)
    - `radius_km`: radio de exploración en km (1-200)
    - `points`: cuántos puntos alrededor del centro (1-16)
    """
    # Permite nombres con guion bajo o espacio: santa_cruz -> Santa Cruz
    zone_name_clean = zone_name.replace("_", " ").title()
    result = get_weather_zone(zone_name_clean, radius_km=radius_km, num_points=points)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.get("/weather/grid")
def weather_grid(
    lat: float = Query(..., description="Latitud del punto central (ej: -0.74)"),
    lon: float = Query(..., description="Longitud del punto central (ej: -90.34)"),
    radius_km: float = Query(default=30.0, ge=1, le=300, description="Radio de búsqueda en km"),
    cell_size_km: float = Query(default=20.0, ge=5, le=100, description="Tamaño de cada celda en km")
):
    """
    Retorna el clima de todas las celdas de la grilla dentro del radio.
    - `lat` / `lon`: coordenadas del punto de interés
    - `radius_km`: radio de cobertura (1-300 km)
    - `cell_size_km`: resolución de la grilla (5-100 km)
    """
    result = get_grid_weather(lat=lat, lon=lon, radius_km=radius_km, cell_size_km=cell_size_km)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
