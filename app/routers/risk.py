"""
/api/v1/risk — GARÚA: ¿Voy a la playa?
Risk zone analysis and disaster management.
"""
from fastapi import APIRouter, Request, HTTPException

from app.schemas.responses import RiskZonesResponse, RiskZone
from app.services.ingestion import fetch_openmeteo_history
from app.ml.features import engineer_features, extract_window
from app.core.config import settings

router = APIRouter()

# Risk zones for San Cristóbal
# In production, these would come from GeoJSON files with real geometry.
# For MVP, we define representative zones with vulnerability factors.
ZONES = [
    {
        "zone_id": "pb_moreno",
        "name": "Puerto Baquerizo Moreno (costa)",
        "slope_factor": 0.3,     # flat coastal = flood risk
        "coastal": True,
        "weights": {"rain": 0.4, "wind": 0.3, "soil": 0.1, "fog": 0.2},
    },
    {
        "zone_id": "loberia",
        "name": "La Lobería (playa)",
        "slope_factor": 0.2,
        "coastal": True,
        "weights": {"rain": 0.3, "wind": 0.4, "soil": 0.0, "fog": 0.3},
    },
    {
        "zone_id": "tijeretas",
        "name": "Cerro Tijeretas (sendero)",
        "slope_factor": 0.6,     # steep trail
        "coastal": False,
        "weights": {"rain": 0.5, "wind": 0.2, "soil": 0.2, "fog": 0.1},
    },
    {
        "zone_id": "junco",
        "name": "El Junco (zona alta agrícola)",
        "slope_factor": 0.5,
        "coastal": False,
        "weights": {"rain": 0.3, "wind": 0.1, "soil": 0.5, "fog": 0.1},
    },
    {
        "zone_id": "progreso",
        "name": "El Progreso (pueblo interior)",
        "slope_factor": 0.4,
        "coastal": False,
        "weights": {"rain": 0.4, "wind": 0.2, "soil": 0.3, "fog": 0.1},
    },
    {
        "zone_id": "leon_dormido",
        "name": "León Dormido (roca mar abierto)",
        "slope_factor": 0.1,
        "coastal": True,
        "weights": {"rain": 0.1, "wind": 0.5, "soil": 0.0, "fog": 0.4},
    },
    {
        "zone_id": "galapaguera",
        "name": "Galapaguera de Cerro Colorado",
        "slope_factor": 0.5,
        "coastal": False,
        "weights": {"rain": 0.3, "wind": 0.1, "soil": 0.4, "fog": 0.2},
    },
]


@router.get("/zones", response_model=RiskZonesResponse)
async def get_risk_zones(request: Request):
    """Get risk level for each zone based on current predictions."""
    registry = request.app.state.models
    if not registry.models:
        raise HTTPException(503, "No models loaded")

    raw_df = await fetch_openmeteo_history(hours_back=48)
    df = engineer_features(raw_df)
    first_model = next(iter(registry.models.values()))
    window = extract_window(df, first_model.feature_cols, first_model.lookback)

    if window is None:
        raise HTTPException(422, "Not enough data")

    predictions = registry.predict_all(window)

    # Event probabilities
    rain_p = predictions.get("heavy_rain", {}).get("probability", 0) or 0
    wind_p = predictions.get("high_wind", {}).get("probability", 0) or 0
    soil_p = predictions.get("low_soil_moisture", {}).get("probability", 0) or 0
    fog_p = predictions.get("fog_event", {}).get("probability", 0) or 0

    probs = {"rain": rain_p, "wind": wind_p, "soil": soil_p, "fog": fog_p}

    zones = []
    max_risk_score = 0

    for z in ZONES:
        # Weighted risk score
        score = sum(z["weights"][k] * probs[k] for k in probs)

        # Amplify by slope factor (steeper = more landslide risk from rain)
        if rain_p > 0.3:
            score *= 1 + z["slope_factor"]

        score = min(1.0, score)
        max_risk_score = max(max_risk_score, score)

        # Classify
        if score > 0.7:
            risk_level = "critical"
        elif score > 0.4:
            risk_level = "high"
        elif score > 0.2:
            risk_level = "medium"
        else:
            risk_level = "low"

        zones.append(RiskZone(
            zone_id=z["zone_id"],
            name=z["name"],
            risk_level=risk_level,
            risk_score=round(score, 3),
            factors={k: round(v, 3) for k, v in probs.items()},
        ))

    # Overall
    if max_risk_score > 0.6:
        overall = "red"
    elif max_risk_score > 0.3:
        overall = "yellow"
    else:
        overall = "green"

    return RiskZonesResponse(
        timestamp=str(df.index[-1]),
        zones=sorted(zones, key=lambda z: -z.risk_score),
        overall_risk=overall,
    )
