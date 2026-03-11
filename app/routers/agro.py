"""
/api/v1/agro — SCALESIA: ¿Qué siembro hoy?
Smart crop calendar with irrigation recommendations.
"""
import math
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException

from app.schemas.responses import AgroCalendarResponse, CropRecommendation
from app.services.ingestion import fetch_openmeteo_history
from app.ml.features import engineer_features, extract_window
from app.core.config import settings

router = APIRouter()

# Crop database for Galápagos highland agriculture
# Source: MAG Ecuador / INIAP adapted for San Cristóbal conditions
CROPS = [
    {
        "name": "Café",
        "temp_min": 15, "temp_max": 28,
        "soil_moisture_min": 0.25, "soil_moisture_opt": 0.40,
        "sow_months": [1, 2, 3, 10, 11, 12],    # hot/wet season
        "harvest_months": [5, 6, 7, 8, 9],
    },
    {
        "name": "Plátano",
        "temp_min": 18, "temp_max": 35,
        "soil_moisture_min": 0.30, "soil_moisture_opt": 0.45,
        "sow_months": [1, 2, 3, 4, 11, 12],
        "harvest_months": list(range(1, 13)),     # year-round
    },
    {
        "name": "Naranja",
        "temp_min": 15, "temp_max": 38,
        "soil_moisture_min": 0.20, "soil_moisture_opt": 0.35,
        "sow_months": [1, 2, 3],
        "harvest_months": [6, 7, 8, 9, 10],
    },
    {
        "name": "Piña",
        "temp_min": 20, "temp_max": 35,
        "soil_moisture_min": 0.20, "soil_moisture_opt": 0.30,
        "sow_months": [1, 2, 3, 4],
        "harvest_months": [7, 8, 9, 10, 11, 12],
    },
    {
        "name": "Hortalizas",
        "temp_min": 12, "temp_max": 30,
        "soil_moisture_min": 0.30, "soil_moisture_opt": 0.50,
        "sow_months": list(range(1, 13)),
        "harvest_months": list(range(1, 13)),
    },
]


def compute_et0(temp_c: float, rh_pct: float, wind_ms: float,
                solar_kw: float) -> float:
    """Simplified Penman-Monteith ET₀ (mm/day).

    Uses the FAO-56 simplified formula for reference evapotranspiration.
    All inputs come directly from the notebook's station variables.
    """
    # Net radiation approximation (MJ/m²/day from kW/m²)
    Rn = solar_kw * 3.6 * 24 * 0.77  # convert kW → MJ, assume 77% net

    # Psychrometric constant (kPa/°C) at sea level
    gamma = 0.066

    # Saturation vapor pressure slope (kPa/°C)
    delta = 4098 * (0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))) / (
        (temp_c + 237.3) ** 2
    )

    # Vapor pressures
    es = 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = es * rh_pct / 100

    # FAO-56 simplified ET₀
    numerator = 0.408 * delta * Rn + gamma * (900 / (temp_c + 273)) * wind_ms * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * wind_ms)

    et0 = max(0.0, numerator / denominator)
    return round(et0, 2)


@router.get("/calendar", response_model=AgroCalendarResponse)
async def get_agro_calendar(request: Request):
    """Get crop recommendations based on current and predicted conditions."""
    registry = request.app.state.models

    raw_df = await fetch_openmeteo_history(hours_back=48)
    df = engineer_features(raw_df)

    # Get current conditions from latest data
    stn = settings.TARGET_STATION
    latest = raw_df.iloc[-1]
    temp = latest.get(f"{stn}_temp_c", 22)
    rh = latest.get(f"{stn}_rh_avg", 75)
    wind = latest.get(f"{stn}_wind_speed_ms", 2)
    solar = latest.get(f"{stn}_solar_kw", 0.3)
    soil_m = latest.get(f"{stn}_soil_moisture_1", 0.30)

    # Compute ET₀
    et0 = compute_et0(temp, rh, wind, solar)

    # Get rain prediction
    rain_prob = 0.0
    if registry.models:
        first_model = next(iter(registry.models.values()))
        window = extract_window(df, first_model.feature_cols, first_model.lookback)
        if window is not None:
            predictions = registry.predict_all(window)
            rain_prob = predictions.get("heavy_rain", {}).get("probability", 0) or 0
            drought_prob = predictions.get("low_soil_moisture", {}).get("probability", 0) or 0
        else:
            drought_prob = 0.0
    else:
        drought_prob = 0.0

    # Irrigation need: ET₀ minus expected rain contribution
    expected_rain_mm = rain_prob * 5.0  # rough estimate: if 100% rain prob → ~5mm
    irrigation_need = max(0, et0 - expected_rain_mm - (soil_m * 10))

    # Build crop recommendations
    now = datetime.utcnow()
    month = now.month
    crops = []

    for crop_info in CROPS:
        # Determine action
        if month in crop_info["sow_months"] and soil_m >= crop_info["soil_moisture_min"]:
            action = "SEMBRAR"
        elif month in crop_info["harvest_months"]:
            action = "COSECHAR"
        elif soil_m < crop_info["soil_moisture_min"]:
            action = "REGAR"
        else:
            action = "ESPERAR"

        # Alert
        alert = None
        if soil_m < crop_info["soil_moisture_min"]:
            alert = f"Suelo seco ({soil_m:.0%}) — regar urgente"
        elif temp < crop_info["temp_min"]:
            alert = f"Temperatura baja ({temp:.1f}°C) — proteger cultivo"
        elif temp > crop_info["temp_max"]:
            alert = f"Temperatura alta ({temp:.1f}°C) — riesgo de estrés térmico"

        crops.append(CropRecommendation(
            crop=crop_info["name"],
            action=action,
            alert=alert,
            soil_moisture_pct=round(soil_m * 100, 1),
            rain_next_6h_prob=round(rain_prob, 3),
        ))

    return AgroCalendarResponse(
        timestamp=str(df.index[-1]),
        et0_mm_day=et0,
        irrigation_need_mm=round(irrigation_need, 1),
        drought_alert=drought_prob > 0.5,
        crops=crops,
    )
