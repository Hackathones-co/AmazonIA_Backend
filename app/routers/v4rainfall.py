"""
GET /api/v1/v4/rainfall/{station}?horizon=3

LightGBM v4 rainfall prediction endpoint.
Returns a 3-class classification (no rain / light rain / heavy rain)
for a given station and forecast horizon.
"""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Request

from app.ml.pkl_features import build_pkl_features, extract_latest_row
from app.schemas.responses import V4RainfallResponse
from app.services.ingestion import fetch_openmeteo_history

router = APIRouter()

STATIONS = {
    "cer":  "Cerro Brujo",
    "jun":  "El Junco",
    "merc": "Mercado Central",
    "mira": "El Mirador",
}


@router.get("/rainfall/{station}", response_model=V4RainfallResponse)
async def get_v4_rainfall(
    station: str,
    request: Request,
    horizon: int = Query(default=3, description="Forecast horizon in hours: 1, 3, or 6"),
):
    """LightGBM v4 rainfall classification for a station + horizon.

    Pipeline:
    1. Fetch last 48h from Open-Meteo (15-min upsampled)
    2. Build 237-feature vector via pkl_features pipeline
    3. Run pkl LightGBM model for (station, horizon)
    4. Return 3-class probabilities + current conditions
    """
    if station not in STATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Station '{station}' not found. Valid stations: {list(STATIONS.keys())}",
        )
    if horizon not in (1, 3, 6):
        raise HTTPException(status_code=400, detail="Horizon must be 1, 3, or 6 hours")

    pkl_models = getattr(request.app.state, "pkl_models", None)
    if pkl_models is None or not pkl_models.models:
        raise HTTPException(status_code=503, detail="PKL models not loaded. Check /health endpoint.")

    # 1. Fetch weather data
    try:
        raw_df = await fetch_openmeteo_history(hours_back=48)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch weather data: {exc}")

    if raw_df.empty:
        raise HTTPException(status_code=422, detail="Weather API returned empty dataset")

    # 2. Feature engineering
    try:
        feat_df = build_pkl_features(raw_df)
        feature_row = extract_latest_row(feat_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {exc}")

    # 3. Predict
    try:
        result = pkl_models.predict(station, horizon, feature_row)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # 4. Current conditions from the latest raw row
    latest = raw_df.iloc[-1]
    conditions = {
        "temp_c":    round(float(latest.get(f"{station}_temp_c",    0.0) or 0.0), 1),
        "rh_pct":    round(float(latest.get(f"{station}_rh_avg",    0.0) or 0.0), 1),
        "wind_ms":   round(float(latest.get(f"{station}_wind_speed_ms", 0.0) or 0.0), 2),
        "precip_mm": round(float(latest.get(f"{station}_rain_mm",   0.0) or 0.0), 2),
    }

    return V4RainfallResponse(
        station=station,
        station_name=STATIONS[station],
        horizon_h=horizon,
        timestamp=datetime.utcnow().replace(microsecond=0).isoformat(),
        pred_class=result["pred_class"],
        pred_prob_no_rain=result["pred_prob_no_rain"],
        pred_prob_light=result["pred_prob_light"],
        pred_prob_heavy=result["pred_prob_heavy"],
        class_label=result["class_label"],
        tl=result["tl"],
        th=result["th"],
        conditions=conditions,
        data_source="Open-Meteo + LightGBM v4",
    )
