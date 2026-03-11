"""
/api/v1/visit — ENCANTADA: ¿Qué hago hoy?
Smart tourism activity recommendations based on weather predictions.
"""
from fastapi import APIRouter, Request, HTTPException

from app.schemas.responses import VisitRecommendResponse, ActivityRecommendation
from app.services.ingestion import fetch_openmeteo_history
from app.ml.features import engineer_features, extract_window
from app.core.config import settings

router = APIRouter()

# Activities with weather requirements
ACTIVITIES = [
    {
        "activity": "Snorkel en La Lobería",
        "location": "La Lobería",
        "type": "water",
        "wind_max": 0.4,      # max wind_prob before score drops
        "rain_max": 0.3,
        "fog_max": 0.3,
        "best_time": "8:00 - 11:00 AM",
        "base_score": 95,      # perfect-conditions score
    },
    {
        "activity": "Buceo en León Dormido",
        "location": "León Dormido (Kicker Rock)",
        "type": "water_deep",
        "wind_max": 0.3,
        "rain_max": 0.4,
        "fog_max": 0.4,
        "best_time": "7:00 - 10:00 AM",
        "base_score": 90,
    },
    {
        "activity": "Senderismo Cerro Tijeretas",
        "location": "Cerro Tijeretas",
        "type": "land",
        "wind_max": 0.7,      # hiking tolerates more wind
        "rain_max": 0.4,
        "fog_max": 0.6,
        "best_time": "6:00 - 9:00 AM",
        "base_score": 85,
    },
    {
        "activity": "Visitar Galapaguera (tortugas)",
        "location": "Galapaguera de Cerro Colorado",
        "type": "land",
        "wind_max": 0.8,
        "rain_max": 0.5,
        "fog_max": 0.8,
        "best_time": "9:00 AM - 12:00 PM",
        "base_score": 85,
    },
    {
        "activity": "Kayak en Bahía Naufragio",
        "location": "Puerto Baquerizo Moreno",
        "type": "water",
        "wind_max": 0.4,
        "rain_max": 0.3,
        "fog_max": 0.5,
        "best_time": "7:00 - 10:00 AM",
        "base_score": 80,
    },
    {
        "activity": "Surf en Punta Carola",
        "location": "Punta Carola",
        "type": "water",
        "wind_max": 0.5,      # some wind is actually good for surf
        "rain_max": 0.6,      # surfers don't mind rain
        "fog_max": 0.5,
        "best_time": "6:00 - 9:00 AM",
        "base_score": 75,
    },
    {
        "activity": "Laguna El Junco (zona alta)",
        "location": "El Junco",
        "type": "land",
        "wind_max": 0.6,
        "rain_max": 0.3,      # trail gets slippery
        "fog_max": 0.4,       # garúa blocks the view
        "best_time": "8:00 - 11:00 AM",
        "base_score": 80,
    },
    {
        "activity": "Centro de Interpretación",
        "location": "Puerto Baquerizo Moreno",
        "type": "indoor",
        "wind_max": 1.0,      # indoor = weather irrelevant
        "rain_max": 1.0,
        "fog_max": 1.0,
        "best_time": "Todo el día",
        "base_score": 65,      # lower base because it's a rainy-day fallback
    },
    {
        "activity": "Observación de lobos marinos",
        "location": "Malecón, Puerto Baquerizo Moreno",
        "type": "land",
        "wind_max": 0.7,
        "rain_max": 0.5,
        "fog_max": 0.6,
        "best_time": "6:00 - 8:00 AM o 4:00 - 6:00 PM",
        "base_score": 80,
    },
]


@router.get("/recommend", response_model=VisitRecommendResponse)
async def get_recommendations(request: Request, top_n: int = 5):
    """Get top N activity recommendations for today."""
    registry = request.app.state.models

    raw_df = await fetch_openmeteo_history(hours_back=48)
    df = engineer_features(raw_df)

    # Get predictions
    wind_prob, rain_prob, fog_prob = 0.0, 0.0, 0.0
    if registry.models:
        first_model = next(iter(registry.models.values()))
        window = extract_window(df, first_model.feature_cols, first_model.lookback)
        if window is not None:
            predictions = registry.predict_all(window)
            wind_prob = predictions.get("high_wind", {}).get("probability", 0) or 0
            rain_prob = predictions.get("heavy_rain", {}).get("probability", 0) or 0
            fog_prob = predictions.get("fog_event", {}).get("probability", 0) or 0

    scored = []
    for act in ACTIVITIES:
        # Score reduction per factor
        wind_penalty = max(0, (wind_prob - act["wind_max"]) / (1 - act["wind_max"] + 0.01)) * 40
        rain_penalty = max(0, (rain_prob - act["rain_max"]) / (1 - act["rain_max"] + 0.01)) * 35
        fog_penalty = max(0, (fog_prob - act["fog_max"]) / (1 - act["fog_max"] + 0.01)) * 25

        score = max(0, min(100, int(act["base_score"] - wind_penalty - rain_penalty - fog_penalty)))

        # Generate reason
        if score >= 80:
            reason = "Condiciones excelentes"
        elif score >= 60:
            issues = []
            if wind_prob > act["wind_max"]:
                issues.append("viento moderado")
            if rain_prob > act["rain_max"]:
                issues.append("posible lluvia")
            if fog_prob > act["fog_max"]:
                issues.append("garúa esperada")
            reason = f"Buenas condiciones, pero atención: {', '.join(issues)}" if issues else "Buenas condiciones"
        elif score >= 40:
            reason = "Condiciones aceptables con riesgo moderado"
        else:
            reason = "No recomendado hoy por condiciones adversas"

        scored.append(ActivityRecommendation(
            activity=act["activity"],
            location=act["location"],
            score=score,
            reason=reason,
            best_time=act["best_time"],
        ))

    # Sort by score descending, take top_n
    scored.sort(key=lambda a: -a.score)

    return VisitRecommendResponse(
        timestamp=str(df.index[-1]),
        activities=scored[:top_n],
    )
