"""
/api/v1/nowcast/{station} — Core prediction endpoint.
Runs all 4 event models and returns probabilities.
"""
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException

from app.schemas.responses import NowcastResponse, EventPrediction, RainPredictionResponse
from app.services.ingestion import fetch_openmeteo_history
from app.ml.features import engineer_features, extract_window
from app.core.config import settings

router = APIRouter()

# ── Station metadata ──────────────────────────────────────────────
STATIONS = {
    "cer": {"name": "Cerro Brujo"},
    "jun": {"name": "Junquillo"},
    "merc": {"name": "Mercado"},
    "mira": {"name": "Mirador"},
}

# ── Rain classification thresholds ────────────────────────────────
# Maps probability ranges to rain classes and their confidence
THRESHOLDS = {
    1: {"no_rain_max": 0.33, "light_max": 0.66, "heavy_min": 0.66},
    3: {"no_rain_max": 0.33, "light_max": 0.66, "heavy_min": 0.66},
    6: {"no_rain_max": 0.33, "light_max": 0.66, "heavy_min": 0.66},
}


@router.get("/nowcast/{station}", response_model=NowcastResponse)
async def get_nowcast(station: str, request: Request):
    """Get nowcast predictions for all 4 events at a given station.

    The pipeline:
    1. Fetch last 48h of data from Open-Meteo
    2. Apply feature engineering (cell 28 of notebooks)
    3. Extract last 24h window (96 timesteps × 15min)
    4. Run all 4 models (heavy_rain, high_wind, low_soil_moisture, fog_event)
    5. Return calibrated probabilities + alert flags
    """
    if station not in settings.STATIONS:
        raise HTTPException(404, f"Station '{station}' not found. Available: {settings.STATIONS}")

    registry = request.app.state.models
    if not registry.models:
        raise HTTPException(503, "No models loaded. Check /health endpoint.")

    # 1. Fetch data
    try:
        raw_df = await fetch_openmeteo_history(hours_back=48)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch weather data: {e}")

    # 2. Feature engineering
    df = engineer_features(raw_df)

    # 3. Extract window — use feature_cols from first model (all models share the same)
    first_model = next(iter(registry.models.values()))
    window = extract_window(df, first_model.feature_cols, first_model.lookback)

    if window is None:
        raise HTTPException(422, "Not enough data to form a prediction window")

    # 4. Run all models
    predictions_raw = registry.predict_all(window)

    # 5. Format response
    predictions = {
        event: EventPrediction(**pred)
        for event, pred in predictions_raw.items()
    }

    return NowcastResponse(
        station=station,
        timestamp=str(df.index[-1]),
        predictions=predictions,
    )


# ── Helper: Get rain prediction ───────────────────────────────────
def get_prediction(station: str, horizon: int, registry, raw_df, df):
    """
    Get rain classification (0=No Rain, 1=Light, 2=Heavy) with confidence
    from the heavy_rain model probability.
    
    Returns a dict with pred_class, pred_prob, sensor readings, etc.
    """
    if not registry.models:
        raise HTTPException(503, "No models loaded")

    # Extract prediction window
    first_model = next(iter(registry.models.values()))
    window = extract_window(df, first_model.feature_cols, first_model.lookback)
    if window is None:
        raise HTTPException(422, "Not enough data to form a prediction window")

    # Get heavy_rain prediction
    heavy_rain_model = registry.get("heavy_rain")
    if not heavy_rain_model:
        raise HTTPException(503, "Heavy rain model not loaded")

    pred_output = heavy_rain_model.predict(window)
    rain_prob = pred_output.get("probability", 0.0) or 0.0

    # Classify: 0=No Rain, 1=Light, 2=Heavy
    thresholds = THRESHOLDS[horizon]
    if rain_prob < thresholds["no_rain_max"]:
        pred_class = 0
    elif rain_prob < thresholds["light_max"]:
        pred_class = 1
    else:
        pred_class = 2

    # Get current sensor readings (latest from raw_df)
    latest = raw_df.iloc[-1]
    rh_avg = latest.get(f"{station}_rh_avg", 0.0)
    temp_c = latest.get(f"{station}_temp_c", 0.0)
    wind_ms = latest.get(f"{station}_wind_speed_ms", 0.0)
    precip_mm = latest.get(f"{station}_precip_mm", 0.0)

    return {
        "pred_class": pred_class,
        "pred_prob": round(rain_prob, 4),
        "obs_precip_mm": round(float(precip_mm) if precip_mm else 0.0, 2),
        "rh_avg": round(float(rh_avg) if rh_avg else 0.0, 1),
        "temp_c": round(float(temp_c) if temp_c else 0.0, 1),
        "wind_ms": round(float(wind_ms) if wind_ms else 0.0, 2),
        "data_source": "Open-Meteo",
    }


@router.get("/rainfall/{station}", response_model=RainPredictionResponse)
async def get_rainfall_prediction(station: str, request: Request, horizon: int = 1):
    """Get rain classification (0=No Rain, 1=Light, 2=Heavy) with sensor readings.

    Returns pred_class (0=No Rain, 1=Light, 2=Heavy),
    pred_prob (model confidence), and current sensor readings.
    """
    if station not in STATIONS:
        raise HTTPException(status_code=404, detail=f"Station '{station}' not found. Valid: {list(STATIONS.keys())}")
    if horizon not in (1, 3, 6):
        raise HTTPException(status_code=400, detail="Horizon must be 1, 3, or 6")

    registry = request.app.state.models

    # Fetch and engineer data
    try:
        raw_df = await fetch_openmeteo_history(hours_back=48)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch weather data: {e}")

    df = engineer_features(raw_df)

    # Get prediction
    pred = get_prediction(station, horizon, registry, raw_df, df)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "station_id": station,
        "station_name": STATIONS[station]["name"],
        "horizon_h": horizon,
        "pred_class": pred["pred_class"],
        "pred_prob": pred["pred_prob"],
        "class_label": ["No Rain", "Light Rain", "Heavy Rain"][pred["pred_class"]],
        "obs_precip_mm": pred["obs_precip_mm"],
        "conditions": {
            "rh_avg": pred["rh_avg"],
            "temp_c": pred["temp_c"],
            "wind_ms": pred["wind_ms"],
        },
        "thresholds": THRESHOLDS[horizon],
        "data_source": pred["data_source"],
    }

