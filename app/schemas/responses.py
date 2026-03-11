"""Pydantic schemas for all API responses."""
from pydantic import BaseModel, Field


# ── Core Nowcast ──────────────────────────────────────────────────
class EventPrediction(BaseModel):
    event: str = Field(description="Event type: heavy_rain, high_wind, low_soil_moisture, fog_event")
    probability: float | None = Field(description="Calibrated probability [0, 1]")
    alert: bool | None = Field(description="True if probability >= threshold")
    threshold: float = Field(description="Optimal threshold from validation set")


class NowcastResponse(BaseModel):
    station: str
    timestamp: str
    predictions: dict[str, EventPrediction]


# ── Rain Prediction (Rainfall) ────────────────────────────────────
class RainPredictionResponse(BaseModel):
    """Returns pred_class (0=No Rain, 1=Light, 2=Heavy),
    pred_prob (model confidence), and current sensor readings."""
    timestamp: str = Field(description="UTC timestamp in ISO format")
    station_id: str = Field(description="Station identifier (cer, jun, merc, mira)")
    station_name: str = Field(description="Full station name")
    horizon_h: int = Field(description="Forecast horizon in hours (1, 3, or 6)")
    pred_class: int = Field(ge=0, le=2, description="Classification: 0=No Rain, 1=Light, 2=Heavy")
    pred_prob: float = Field(ge=0.0, le=1.0, description="Model confidence [0, 1]")
    class_label: str = Field(description="Human-readable label (No Rain, Light Rain, Heavy Rain)")
    obs_precip_mm: float = Field(description="Current observed precipitation (mm)")
    conditions: dict = Field(description="Current sensor readings")
    thresholds: dict = Field(description="Classification thresholds for this horizon")
    data_source: str = Field(description="Source of the data (Open-Meteo, etc)")


# ── Alerts ────────────────────────────────────────────────────────
class Alert(BaseModel):
    type: str           # e.g. "heavy_rain", "high_wind"
    severity: str       # "low", "medium", "high", "critical"
    module: str         # which module cares most
    message: str
    probability: float


class AlertsResponse(BaseModel):
    timestamp: str
    active_alerts: list[Alert]
    overall_risk: str   # "green", "yellow", "red"


# ── MANGLE (Pesca) ───────────────────────────────────────────────
class PescaScoreResponse(BaseModel):
    score: int = Field(ge=0, le=100, description="Fishing safety score 0-100")
    recommendation: str      # "SEGURO", "PRECAUCIÓN", "NO SALIR"
    detail: str              # Human-readable explanation
    wind_risk: float
    rain_risk: float
    fog_risk: float
    wave_height_m: float | None
    wave_risk: float


# ── SCALESIA (Agro) ──────────────────────────────────────────────
class CropRecommendation(BaseModel):
    crop: str
    action: str              # "SEMBRAR", "REGAR", "COSECHAR", "ESPERAR"
    alert: str | None
    soil_moisture_pct: float | None
    rain_next_6h_prob: float


class AgroCalendarResponse(BaseModel):
    timestamp: str
    et0_mm_day: float | None  # evapotranspiration
    irrigation_need_mm: float | None
    drought_alert: bool
    crops: list[CropRecommendation]


# ── GALÁPAGO (Bio) ───────────────────────────────────────────────
class SpeciesStatus(BaseModel):
    species: str
    risk_level: str          # "normal", "watch", "alert", "critical"
    stress_factor: str | None
    reproductive_phase: str | None
    detail: str


class BioStatusResponse(BaseModel):
    timestamp: str
    species: list[SpeciesStatus]
    nesting_active: bool
    thermal_stress_index: float


# ── GARÚA (Risk) ─────────────────────────────────────────────────
class RiskZone(BaseModel):
    zone_id: str
    name: str
    risk_level: str          # "low", "medium", "high", "critical"
    risk_score: float
    factors: dict[str, float]  # e.g. {"rain": 0.8, "wind": 0.3, "soil": 0.6}


class RiskZonesResponse(BaseModel):
    timestamp: str
    zones: list[RiskZone]
    overall_risk: str


# ── ENCANTADA (Tourism) ──────────────────────────────────────────
class ActivityRecommendation(BaseModel):
    activity: str
    location: str
    score: int = Field(ge=0, le=100)
    reason: str
    best_time: str | None


class VisitRecommendResponse(BaseModel):
    timestamp: str
    activities: list[ActivityRecommendation]


# ── Chat ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    module: str | None = None   # optional context: "pesca", "agro", etc.
    history: list[dict] = []    # previous messages


class ChatResponse(BaseModel):
    response: str
    module_data: dict | None = None
    sources: list[str] = []


# ── Voice ─────────────────────────────────────────────────────────
class VoiceRequest(BaseModel):
    audio_b64: str              # base64-encoded audio
    module: str | None = None


class VoiceResponse(BaseModel):
    transcript: str
    response_text: str
    audio_b64: str              # base64-encoded response audio


class SignedUrlResponse(BaseModel):
    url: str                    # wss://... signed URL for ElevenLabs Conversational AI
    agent_id: str               # the agent this URL is scoped to
    dynamic_variables: dict     # live sensor + prediction snapshot to send on WS init
