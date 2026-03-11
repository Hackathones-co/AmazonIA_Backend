"""
/api/v1/alerts — Aggregated alert system across all events.
"""
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException

from app.schemas.responses import AlertsResponse, Alert
from app.services.ingestion import fetch_openmeteo_history
from app.ml.features import engineer_features, extract_window

router = APIRouter()

# Severity thresholds for each event type
SEVERITY_THRESHOLDS = {
    "heavy_rain": {"medium": 0.3, "high": 0.6, "critical": 0.85},
    "high_wind": {"medium": 0.3, "high": 0.6, "critical": 0.85},
    "low_soil_moisture": {"medium": 0.4, "high": 0.7, "critical": 0.9},
    "fog_event": {"medium": 0.4, "high": 0.7, "critical": 0.9},
}

EVENT_MODULES = {
    "heavy_rain": "GARÚA",
    "high_wind": "MANGLE",
    "low_soil_moisture": "SCALESIA",
    "fog_event": "MANGLE",
}

EVENT_MESSAGES = {
    "heavy_rain": {
        "medium": "Lluvia moderada esperada en las próximas 3 horas",
        "high": "Lluvia intensa probable — evitar zonas bajas y quebradas",
        "critical": "⚠️ Lluvia extrema inminente — alerta máxima para zonas de riesgo",
    },
    "high_wind": {
        "medium": "Viento moderado — precaución en actividades marítimas",
        "high": "Viento fuerte esperado — no recomendable salir a pescar",
        "critical": "⚠️ Viento extremo — peligro para navegación y estructuras",
    },
    "low_soil_moisture": {
        "medium": "Suelo seco — considerar riego suplementario",
        "high": "Sequía moderada — riego urgente para cultivos sensibles",
        "critical": "⚠️ Sequía severa — estrés hídrico crítico en zona agrícola",
    },
    "fog_event": {
        "medium": "Garúa leve esperada — visibilidad reducida en zona alta",
        "high": "Niebla densa probable — precaución en navegación costera",
        "critical": "⚠️ Visibilidad mínima — evitar navegación y actividades aéreas",
    },
}


def classify_severity(event: str, prob: float) -> str:
    thresholds = SEVERITY_THRESHOLDS.get(event, {})
    if prob >= thresholds.get("critical", 0.85):
        return "critical"
    elif prob >= thresholds.get("high", 0.6):
        return "high"
    elif prob >= thresholds.get("medium", 0.3):
        return "medium"
    return "low"


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(request: Request):
    """Get all active alerts across the platform."""
    registry = request.app.state.models
    if not registry.models:
        raise HTTPException(503, "No models loaded")

    # Fetch data and run predictions
    raw_df = await fetch_openmeteo_history(hours_back=48)
    df = engineer_features(raw_df)
    first_model = next(iter(registry.models.values()))
    window = extract_window(df, first_model.feature_cols, first_model.lookback)

    if window is None:
        raise HTTPException(422, "Not enough data")

    predictions = registry.predict_all(window)

    # Build alerts
    active_alerts = []
    max_severity = "low"
    severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    for event, pred in predictions.items():
        prob = pred.get("probability")
        if prob is None:
            continue

        severity = classify_severity(event, prob)
        if severity != "low":
            messages = EVENT_MESSAGES.get(event, {})
            active_alerts.append(Alert(
                type=event,
                severity=severity,
                module=EVENT_MODULES.get(event, "CORE"),
                message=messages.get(severity, f"{event}: probability {prob:.0%}"),
                probability=prob,
            ))

        if severity_order.get(severity, 0) > severity_order.get(max_severity, 0):
            max_severity = severity

    # Overall risk color
    risk_map = {"low": "green", "medium": "yellow", "high": "red", "critical": "red"}

    return AlertsResponse(
        timestamp=str(df.index[-1]),
        active_alerts=sorted(active_alerts, key=lambda a: -a.probability),
        overall_risk=risk_map.get(max_severity, "green"),
    )
