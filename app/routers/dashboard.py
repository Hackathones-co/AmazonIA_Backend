"""
Dashboard endpoints — advisories, predictions, and station data.

Uses only the LightGBM pkl models (no .pt models required).
Existing endpoints are NOT affected.

Endpoints:
  GET /api/v1/advisories              → 5 advisories + all predictions + stations
  GET /api/v1/advisories/{module}     → single advisory module
  GET /api/v1/predictions             → raw 12 predictions (4 stations × 3 horizons)
  GET /api/v1/stations                → current Google Weather data for 4 stations
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Request

from app.ml.pkl_features import build_pkl_features, extract_latest_row
from app.services.google_weather import fetch_all_stations
from app.services.ingestion import fetch_openmeteo_history

logger = logging.getLogger(__name__)
router = APIRouter()

MODULES = ("pesca", "agro", "biodiversidad", "riesgo", "turismo")
STATIONS_ALL = ("cer", "jun", "merc", "mira")
HORIZONS = (1, 3, 6)

# ── helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


async def _get_predictions(pkl_models) -> dict:
    """Run all 12 LightGBM predictions. Returns nested dict[station][horizon]."""
    raw_df = await fetch_openmeteo_history(hours_back=48)
    if raw_df.empty:
        raise HTTPException(status_code=422, detail="Open-Meteo returned empty dataset")

    feat_df = build_pkl_features(raw_df)
    feature_row = extract_latest_row(feat_df)

    preds = {}
    for stn in STATIONS_ALL:
        preds[stn] = {}
        for h in HORIZONS:
            try:
                r = pkl_models.predict(stn, h, feature_row)
                preds[stn][f"{h}h"] = {
                    "pred_class": r["pred_class"],
                    "class_label": r["class_label"],
                    "probs": {
                        "no_rain": r["pred_prob_no_rain"],
                        "light":   r["pred_prob_light"],
                        "heavy":   r["pred_prob_heavy"],
                    },
                }
            except Exception as e:
                logger.warning(f"Prediction failed {stn}/{h}h: {e}")
                preds[stn][f"{h}h"] = {"pred_class": 0, "class_label": "Sin lluvia",
                                       "probs": {"no_rain": 1.0, "light": 0.0, "heavy": 0.0}}
    return preds


def _is_heavy(preds: dict, station: str, horizon: str) -> bool:
    return preds.get(station, {}).get(horizon, {}).get("pred_class", 0) == 2


def _is_rain(preds: dict, station: str, horizon: str) -> bool:
    return preds.get(station, {}).get(horizon, {}).get("pred_class", 0) >= 1


# ── Advisory builders ──────────────────────────────────────────────────────────

def _advisory_pesca(preds: dict, stations: dict) -> dict:
    """Fishing safety — uses coastal stations (merc, mira) + wind speed."""
    wind_ms = max(
        stations.get("merc", {}).get("wind_ms", 0.0),
        stations.get("mira", {}).get("wind_ms", 0.0),
    )
    heavy_coastal_1h = _is_heavy(preds, "merc", "1h") or _is_heavy(preds, "mira", "1h")
    heavy_coastal_3h = _is_heavy(preds, "merc", "3h") or _is_heavy(preds, "mira", "3h")
    heavy_coastal = heavy_coastal_1h or heavy_coastal_3h

    if heavy_coastal and wind_ms >= 10:
        level, title, action = "danger", "PELIGRO — No salir", "Permanecer en puerto"
        detail = f"Lluvia intensa prevista + viento {wind_ms:.1f} m/s"
    elif heavy_coastal and wind_ms >= 6:
        level, title, action = "caution", "PRECAUCIÓN — Mar agitado", "Navegar cerca de costa"
        detail = f"Lluvia prevista + viento {wind_ms:.1f} m/s"
    elif wind_ms >= 10:
        level, title, action = "caution", "VIENTO FUERTE — Navegar con cuidado", "Navegar cerca de costa"
        detail = f"Viento {wind_ms:.1f} m/s sin lluvia significativa"
    else:
        level, title, action = "safe", "SEGURO — Buen tiempo para pescar", "Condiciones óptimas"
        detail = f"Viento {wind_ms:.1f} m/s, sin lluvia prevista"

    return {
        "module": "pesca",
        "emoji": "🌊",
        "level": level,
        "title": title,
        "detail": detail,
        "action": action,
        "wind_ms": wind_ms,
        "wind_cardinal": stations.get("merc", {}).get("wind_cardinal", ""),
        "heavy_rain_1h": heavy_coastal_1h,
        "heavy_rain_3h": heavy_coastal_3h,
    }


def _advisory_agro(preds: dict, stations: dict) -> dict:
    """Agro — uses highland stations (cer, jun) + 24h rain as soil proxy."""
    rain_24h = max(
        stations.get("cer", {}).get("rain_24h_mm", 0.0),
        stations.get("jun", {}).get("rain_24h_mm", 0.0),
    )
    heavy_highland_6h = _is_heavy(preds, "cer", "6h") or _is_heavy(preds, "jun", "6h")
    rain_highland_6h = _is_rain(preds, "cer", "6h") or _is_rain(preds, "jun", "6h")

    if rain_24h > 15 and heavy_highland_6h:
        level, title, action = "danger", "NO FERTILIZAR — Suelo saturado", "Esperar 24h antes de fertilizar"
        detail = f"Lluvia acumulada {rain_24h:.1f} mm + lluvia intensa prevista a +6h"
    elif heavy_highland_6h:
        level, title, action = "caution", "ESPERAR — Lluvia intensa en 6h", "Posponer actividades de campo"
        detail = f"Lluvia intensa prevista a +6h en tierras altas"
    elif rain_24h > 5 or rain_highland_6h:
        level, title, action = "info", "SUELO HÚMEDO — Actividades ligeras OK", "Evitar labrar el suelo"
        detail = f"Lluvia acumulada {rain_24h:.1f} mm últimas 24h"
    else:
        level, title, action = "safe", "APTO — Condiciones óptimas", "Buen momento para sembrar o fertilizar"
        detail = f"Suelo sin exceso de humedad, {rain_24h:.1f} mm en 24h"

    return {
        "module": "agro",
        "emoji": "🌿",
        "level": level,
        "title": title,
        "detail": detail,
        "action": action,
        "rain_24h_mm": rain_24h,
        "heavy_6h": heavy_highland_6h,
    }


def _advisory_biodiversidad(preds: dict, stations: dict) -> dict:
    """Biodiversidad — highland temp/humidity + rain predictions."""
    temp = max(
        stations.get("jun", {}).get("temp_c", 0.0),
        stations.get("cer", {}).get("temp_c", 0.0),
    )
    rh = (
        stations.get("jun", {}).get("rh_pct", 70)
        + stations.get("cer", {}).get("rh_pct", 70)
    ) / 2

    heavy_highland = _is_heavy(preds, "jun", "6h") or _is_heavy(preds, "cer", "6h")
    no_rain_highland = (
        preds.get("jun", {}).get("6h", {}).get("pred_class", 0) == 0
        and preds.get("cer", {}).get("6h", {}).get("pred_class", 0) == 0
    )

    alerts = []
    if temp > 30:
        alerts.append("estrés_térmico")
    if temp < 15:
        alerts.append("frío_extremo")
    if rh < 50 and no_rain_highland:
        alerts.append("pozas_secándose")
    if heavy_highland:
        alerts.append("recarga_pozas")

    if "estrés_térmico" in alerts:
        level, title = "danger", "ESTRÉS TÉRMICO — Tortugas en riesgo"
        detail = f"Temperatura {temp:.1f}°C supera los 30°C — riesgo de deshidratación"
        action = "Activar monitoreo en Laguna El Junco"
    elif "frío_extremo" in alerts:
        level, title = "caution", "FRÍO EXTREMO — Monitorear tortugas"
        detail = f"Temperatura {temp:.1f}°C por debajo de 15°C"
        action = "Verificar refugios térmicos"
    elif "pozas_secándose" in alerts:
        level, title = "caution", "HUMEDAD BAJA — Pozas en riesgo"
        detail = f"Humedad {rh:.0f}% y sin lluvia prevista — pozas pueden secarse"
        action = "Monitoreo de fuentes de agua"
    elif "recarga_pozas" in alerts:
        level, title = "info", "LLUVIA PREVISTA — Recarga de hábitat"
        detail = f"Lluvia intensa en highlands recargará pozas y vegetación"
        action = "Condiciones favorables para fauna"
    else:
        level, title = "safe", "Hábitat estable"
        detail = f"Temperatura {temp:.1f}°C, humedad {rh:.0f}% — condiciones normales"
        action = "No se requiere intervención"

    return {
        "module": "biodiversidad",
        "emoji": "🐢",
        "level": level,
        "title": title,
        "detail": detail,
        "action": action,
        "temp_c": temp,
        "rh_pct": rh,
        "alerts": alerts,
    }


def _advisory_riesgo(preds: dict, stations: dict) -> dict:
    """Riesgo — flash flood detection: heavy rain highlands + clear coast."""
    heavy_highland_1h = _is_heavy(preds, "jun", "1h") or _is_heavy(preds, "cer", "1h")
    heavy_highland_3h = _is_heavy(preds, "jun", "3h") or _is_heavy(preds, "cer", "3h")
    clear_coastal = (
        preds.get("merc", {}).get("1h", {}).get("pred_class", 0) == 0
        and preds.get("mira", {}).get("1h", {}).get("pred_class", 0) == 0
    )
    rain_highland_any = (
        _is_rain(preds, "jun", "1h") or _is_rain(preds, "cer", "1h")
        or _is_rain(preds, "jun", "3h") or _is_rain(preds, "cer", "3h")
    )

    flash_flood_risk = (heavy_highland_1h or heavy_highland_3h) and clear_coastal

    if flash_flood_risk:
        level, title = "danger", "ALERTA — Riesgo de crecida repentina"
        detail = "Lluvia intensa en tierras altas con costa despejada — agua baja por quebradas"
        action = "Evitar quebradas y zonas bajas. Mantenerse alejado de arroyos."
    elif heavy_highland_1h or heavy_highland_3h:
        level, title = "caution", "PRECAUCIÓN — Lluvia intensa en zonas altas"
        detail = "Lluvia intensa prevista en El Junco / Cerro Alto"
        action = "Evitar senderos de altura. Usar rutas costeras."
    elif rain_highland_any:
        level, title = "info", "Lluvia moderada en tierras altas"
        detail = "Lluvia leve a moderada en highlands — caminos pueden estar resbaladizos"
        action = "Precaución en senderos de montaña"
    else:
        level, title = "safe", "SEGURO — Todos los senderos abiertos"
        detail = "Sin precipitaciones significativas en ninguna zona"
        action = "Condiciones favorables para actividades al aire libre"

    return {
        "module": "riesgo",
        "emoji": "⚠️",
        "level": level,
        "title": title,
        "detail": detail,
        "action": action,
        "flash_flood_risk": flash_flood_risk,
        "heavy_highland_1h": heavy_highland_1h,
        "heavy_highland_3h": heavy_highland_3h,
        "clear_coastal": clear_coastal,
    }


def _advisory_turismo(preds: dict, stations: dict) -> dict:
    """Turismo — activity recommendations based on overall weather."""
    temp = (
        stations.get("merc", {}).get("temp_c", 22.0)
        + stations.get("mira", {}).get("temp_c", 22.0)
    ) / 2
    rh = (
        stations.get("merc", {}).get("rh_pct", 65)
        + stations.get("mira", {}).get("rh_pct", 65)
    ) / 2
    condition = stations.get("merc", {}).get("condition_desc", "")

    heavy_everywhere = all(
        _is_heavy(preds, stn, "3h") for stn in STATIONS_ALL
    )
    heavy_highlands_only = (
        (_is_heavy(preds, "jun", "3h") or _is_heavy(preds, "cer", "3h"))
        and not (_is_heavy(preds, "merc", "3h") or _is_heavy(preds, "mira", "3h"))
    )
    heavy_coastal_only = (
        (_is_heavy(preds, "merc", "3h") or _is_heavy(preds, "mira", "3h"))
        and not (_is_heavy(preds, "jun", "3h") or _is_heavy(preds, "cer", "3h"))
    )
    any_rain = any(_is_rain(preds, stn, "3h") for stn in STATIONS_ALL)

    # Comfort index: 0-100
    # Optimal: 20-28°C, 50-70% RH
    temp_score = max(0, 100 - abs(temp - 24) * 10)
    rh_score = max(0, 100 - abs(rh - 60) * 2)
    comfort = int((temp_score + rh_score) / 2)

    if heavy_everywhere:
        level, title = "caution", "LLUVIA PREVISTA — Bajo techo"
        detail = f"Lluvia intensa en toda la isla. Temperatura {temp:.1f}°C"
        activities = ["🏛️ Centro Darwin", "🍽️ Gastronomía", "🎨 Artesanías", "📚 Museo"]
    elif heavy_highlands_only:
        level, title = "info", "Costa despejada — Lluvia en tierras altas"
        detail = f"Playa y costa despejadas. Lluvia en senderos de altura. {temp:.1f}°C"
        activities = ["🏖️ Playa", "🥤 Puerto", "🚣 Kayak", "🐠 Snorkel"]
    elif heavy_coastal_only:
        level, title = "info", "Costa con lluvia — Senderismo highland OK"
        detail = f"Lluvia en costa. Tierras altas despejadas. {temp:.1f}°C"
        activities = ["🥾 Senderismo", "📸 Fauna", "🌿 El Junco", "🐢 Tortugas"]
    elif any_rain:
        level, title = "info", "Lluvia moderada — Actividades mixtas"
        detail = f"Lluvia leve en algunas zonas. Temperatura {temp:.1f}°C"
        activities = ["🏖️ Playa", "🎨 Artesanías", "🍽️ Gastronomía", "🐢 Centro de Rescate"]
    else:
        level, title = "safe", "DÍA PERFECTO — Todo al aire libre"
        detail = f"Cielo despejado, {temp:.1f}°C, confort {comfort}/100. {condition}"
        activities = ["🏖️ Playa", "🥾 Senderismo", "📸 Fauna", "🚣 Kayak", "🐠 Snorkel", "🐢 Tortugas"]

    return {
        "module": "turismo",
        "emoji": "📸",
        "level": level,
        "title": title,
        "detail": detail,
        "action": activities[0] if activities else "",
        "temp_c": round(temp, 1),
        "rh_pct": round(rh),
        "comfort_index": comfort,
        "condition_desc": condition,
        "activities": activities,
    }


def _build_all_advisories(preds: dict, stations: dict) -> dict:
    return {
        "pesca":        _advisory_pesca(preds, stations),
        "agro":         _advisory_agro(preds, stations),
        "biodiversidad": _advisory_biodiversidad(preds, stations),
        "riesgo":       _advisory_riesgo(preds, stations),
        "turismo":      _advisory_turismo(preds, stations),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/advisories")
async def get_advisories(request: Request):
    """Main dashboard endpoint — 5 advisories + 12 predictions + 4 station readings."""
    pkl_models = getattr(request.app.state, "pkl_models", None)
    if pkl_models is None or not pkl_models.models:
        raise HTTPException(status_code=503, detail="PKL models not loaded")

    try:
        preds, stations = await asyncio.gather(
            _get_predictions(pkl_models),
            fetch_all_stations(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    advisories = _build_all_advisories(preds, stations)

    return {
        "advisories": advisories,
        "predictions": preds,
        "stations": stations,
        "timestamp": _ts(),
    }


@router.get("/advisories/{module}")
async def get_advisory_module(
    module: Literal["pesca", "agro", "biodiversidad", "riesgo", "turismo"],
    request: Request,
):
    """Single advisory module."""
    pkl_models = getattr(request.app.state, "pkl_models", None)
    if pkl_models is None or not pkl_models.models:
        raise HTTPException(status_code=503, detail="PKL models not loaded")

    try:
        preds, stations = await asyncio.gather(
            _get_predictions(pkl_models),
            fetch_all_stations(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    builders = {
        "pesca":        _advisory_pesca,
        "agro":         _advisory_agro,
        "biodiversidad": _advisory_biodiversidad,
        "riesgo":       _advisory_riesgo,
        "turismo":      _advisory_turismo,
    }
    return {**builders[module](preds, stations), "timestamp": _ts()}


@router.get("/predictions")
async def get_predictions(request: Request):
    """Raw 12 LightGBM predictions (4 stations × 3 horizons)."""
    pkl_models = getattr(request.app.state, "pkl_models", None)
    if pkl_models is None or not pkl_models.models:
        raise HTTPException(status_code=503, detail="PKL models not loaded")

    try:
        preds = await _get_predictions(pkl_models)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"predictions": preds, "timestamp": _ts()}


@router.get("/stations")
async def get_stations():
    """Current conditions for all 4 stations from Google Weather API."""
    stations = await fetch_all_stations()
    return {"stations": stations, "timestamp": _ts()}
