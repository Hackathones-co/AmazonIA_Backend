"""
/api/v1/pesca — MANGLE: ¿Salgo a pescar?
Fishing safety score combining wind, rain, fog, and wave models.
"""
from fastapi import APIRouter, Request, HTTPException

from app.schemas.responses import PescaScoreResponse
from app.services.ingestion import fetch_openmeteo_history, fetch_openmeteo_marine
from app.ml.features import engineer_features, extract_window

router = APIRouter()

# Weights for fishing safety score (sum = 1.0)
WEIGHTS = {
    "high_wind": 0.40,      # wind is the #1 danger for artisanal fishers
    "heavy_rain": 0.15,     # heavy rain reduces visibility + rough seas
    "fog_event": 0.20,      # fog = zero visibility on water
    "wave": 0.25,           # wave height from marine API
}

WAVE_THRESHOLDS = {
    "safe": 1.0,       # < 1m = calm
    "caution": 1.5,    # 1-1.5m = moderate
    "danger": 2.5,     # > 2.5m = dangerous
}


@router.get("/score", response_model=PescaScoreResponse)
async def get_fishing_score(request: Request):
    """Calculate fishing safety score (0-100, higher = safer)."""
    registry = request.app.state.models
    if not registry.models:
        raise HTTPException(503, "No models loaded")

    # Fetch weather + marine data in parallel
    raw_df = await fetch_openmeteo_history(hours_back=48)
    marine = await fetch_openmeteo_marine()
    df = engineer_features(raw_df)

    first_model = next(iter(registry.models.values()))
    window = extract_window(df, first_model.feature_cols, first_model.lookback)
    if window is None:
        raise HTTPException(422, "Not enough data")

    predictions = registry.predict_all(window)

    # Extract probabilities (default to 0 if model not loaded)
    wind_prob = predictions.get("high_wind", {}).get("probability", 0) or 0
    rain_prob = predictions.get("heavy_rain", {}).get("probability", 0) or 0
    fog_prob = predictions.get("fog_event", {}).get("probability", 0) or 0

    # Wave risk from marine data
    wave_h = marine.get("wave_height_m")
    if wave_h is not None:
        if wave_h >= WAVE_THRESHOLDS["danger"]:
            wave_risk = 1.0
        elif wave_h >= WAVE_THRESHOLDS["caution"]:
            wave_risk = (wave_h - WAVE_THRESHOLDS["safe"]) / (
                WAVE_THRESHOLDS["danger"] - WAVE_THRESHOLDS["safe"]
            )
        else:
            wave_risk = 0.0
    else:
        wave_risk = 0.3  # unknown = moderate caution

    # Composite score: 100 = perfectly safe, 0 = extremely dangerous
    weighted_risk = (
        WEIGHTS["high_wind"] * wind_prob
        + WEIGHTS["heavy_rain"] * rain_prob
        + WEIGHTS["fog_event"] * fog_prob
        + WEIGHTS["wave"] * wave_risk
    )
    score = max(0, min(100, int(round((1 - weighted_risk) * 100))))

    # Recommendation
    if score >= 70:
        recommendation = "SEGURO"
        detail = "Condiciones favorables para salir a pescar. Mar en calma, buen tiempo."
    elif score >= 40:
        recommendation = "PRECAUCIÓN"
        detail = "Condiciones aceptables pero con riesgo moderado. Mantener atención."
    else:
        recommendation = "NO SALIR"
        detail = "Condiciones peligrosas. Viento fuerte, oleaje alto o visibilidad reducida."

    return PescaScoreResponse(
        score=score,
        recommendation=recommendation,
        detail=detail,
        wind_risk=round(wind_prob, 3),
        rain_risk=round(rain_prob, 3),
        fog_risk=round(fog_prob, 3),
        wave_height_m=wave_h,
        wave_risk=round(wave_risk, 3),
    )
