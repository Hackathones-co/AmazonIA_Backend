"""
/api/v1/voice — Real-time voice calls powered by ElevenLabs Conversational AI.

Architecture (direct / signed-URL approach):
  1. Client calls GET /api/v1/voice/signed-url
     → Backend fetches LIVE sensor data + ML predictions from Open-Meteo + our models
     → Packages as dynamic_variables (temp, rain, soil, event probabilities)
     → Calls ElevenLabs API server-side to get a short-lived signed WSS URL
  2. Client opens the WSS URL directly to ElevenLabs
     → Agent receives all real data inline in its first turn context
     → ElevenLabs handles STT → LLM → TTS natively, sub-200 ms latency
  3. (Optional) WebSocket /api/v1/voice/ws
     → Same flow proxied through our server (logging, restricted networks)

The ElevenLabs agent system prompt uses {{...}} placeholders for the injected data.
"""
import asyncio
import json
import logging
from datetime import datetime

from typing import Union
import httpx
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from app.schemas.responses import SignedUrlResponse
from app.core.config import settings
from app.services.ingestion import fetch_openmeteo_history, fetch_openmeteo_marine
from app.ml.features import engineer_features, extract_window

router = APIRouter()
logger = logging.getLogger(__name__)

_EL_BASE = "https://api.elevenlabs.io/v1"
_EL_WSS  = "wss://api.elevenlabs.io/v1/convai/conversation"


# ── Live data snapshot ────────────────────────────────────────────────────────

async def _build_live_snapshot(request: Union[Request, WebSocket]) -> dict:
    """
    Fetch current sensor readings + ML predictions and return as a flat dict
    of string-serialisable values suitable for ElevenLabs dynamic_variables.

    All values fall back gracefully so a missing model / API failure never
    blocks the call from starting.
    """
    stn = settings.TARGET_STATION
    snapshot: dict = {
        # Metadata
        "timestamp":           datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "station":             stn.upper(),
        # Sensor defaults
        "temp_c":              "N/D",
        "rh_pct":              "N/D",
        "wind_ms":             "N/D",
        "rain_mm_1h":          "N/D",
        "soil_moisture_pct":   "N/D",
        # ML prediction defaults
        "prob_heavy_rain":     "N/D",
        "prob_high_wind":      "N/D",
        "prob_drought":        "N/D",
        "prob_fog":            "N/D",
        "overall_alert":       "sin alertas activas",
        # Marine defaults
        "wave_height_m":       "N/D",
    }

    # ── 1. Sensor data (Open-Meteo) ──────────────────────────────────────────
    try:
        raw_df = await fetch_openmeteo_history(hours_back=48)
        latest = raw_df.iloc[-1]

        temp  = latest.get(f"{stn}_temp_c",        None)
        rh    = latest.get(f"{stn}_rh_avg",        None)
        wind  = latest.get(f"{stn}_wind_speed_ms", None)
        rain  = latest.get(f"{stn}_rain_mm",       None)
        soil  = latest.get(f"{stn}_soil_moisture_1", None)

        if temp  is not None: snapshot["temp_c"]            = f"{temp:.1f} °C"
        if rh    is not None: snapshot["rh_pct"]            = f"{rh:.0f}%"
        if wind  is not None: snapshot["wind_ms"]           = f"{wind:.1f} m/s"
        if rain  is not None: snapshot["rain_mm_1h"]        = f"{rain:.1f} mm"
        if soil  is not None: snapshot["soil_moisture_pct"] = f"{soil*100:.0f}%"

        # ── 2. ML predictions ─────────────────────────────────────────────────
        registry = getattr(request.app.state, "models", None)
        if registry and registry.models:
            df = engineer_features(raw_df)
            first_model = next(iter(registry.models.values()))
            window = extract_window(df, first_model.feature_cols, first_model.lookback)
            if window is not None:
                preds = registry.predict_all(window)

                def _pct(event: str) -> str:
                    p = preds.get(event, {}).get("probability")
                    return f"{p*100:.0f}%" if p is not None else "N/D"

                snapshot["prob_heavy_rain"] = _pct("heavy_rain")
                snapshot["prob_high_wind"]  = _pct("high_wind")
                snapshot["prob_drought"]    = _pct("low_soil_moisture")
                snapshot["prob_fog"]        = _pct("fog_event")

                # Build human-readable alert summary
                alerts = []
                for event, label in [
                    ("heavy_rain",       "lluvia intensa"),
                    ("high_wind",        "viento fuerte"),
                    ("low_soil_moisture","sequía"),
                    ("fog_event",        "niebla densa"),
                ]:
                    p = preds.get(event, {}).get("probability") or 0
                    if p >= 0.6:
                        alerts.append(f"{label} ({p*100:.0f}%)")

                if alerts:
                    snapshot["overall_alert"] = "ALERTA: " + ", ".join(alerts)

    except Exception as e:
        logger.warning(f"[voice] Sensor/ML snapshot failed: {e}")

    # ── 3. Marine data ────────────────────────────────────────────────────────
    try:
        marine = await fetch_openmeteo_marine()
        wh = marine.get("wave_height_m")
        if wh is not None:
            snapshot["wave_height_m"] = f"{wh:.1f} m"
    except Exception as e:
        logger.warning(f"[voice] Marine snapshot failed: {e}")

    return snapshot


# ── ElevenLabs signed-URL fetch ───────────────────────────────────────────────

async def _fetch_signed_url() -> str:
    """
    GET /convai/conversation/get-signed-url — confirmed working method.
    Returns a short-lived WSS URL. API key stays server-side.

    Dynamic variables (live sensor data) cannot be injected here — they are
    sent as a `conversation_initiation_client_data` WebSocket message right
    after the client connects, either by the frontend (direct flow) or by our
    proxy (WS proxy flow).
    """
    if not settings.ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY not configured")
    if not settings.ELEVENLABS_AGENT_ID:
        raise HTTPException(503, "ELEVENLABS_AGENT_ID not configured")

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{_EL_BASE}/convai/conversation/get-signed-url",
            params={"agent_id": settings.ELEVENLABS_AGENT_ID},
            headers={"xi-api-key": settings.ELEVENLABS_API_KEY.strip()},
        )

    if resp.status_code != 200:
        logger.error(f"ElevenLabs signed-URL error {resp.status_code}: {resp.text}")
        raise HTTPException(502, f"ElevenLabs error: {resp.text}")

    data = resp.json()
    url = data.get("signed_url") or data.get("url")
    if not url:
        raise HTTPException(502, f"Unexpected ElevenLabs response: {data}")
    return url


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/voice/signed-url", response_model=SignedUrlResponse)
async def get_signed_url(request: Request):
    """
    Returns a short-lived signed WSS URL for ElevenLabs Conversational AI,
    plus a `dynamic_variables` snapshot of current real conditions.

    Flow:
      1. Frontend calls this endpoint → gets {url, agent_id, dynamic_variables}
      2. Frontend opens WebSocket to `url`
      3. Frontend immediately sends this JSON as the first WS message:
         {
           "type": "conversation_initiation_client_data",
           "dynamic_variables": <dynamic_variables from step 1>
         }
      4. ElevenLabs agent receives live sensor data and can reference them in conversation.
    """
    snapshot = await _build_live_snapshot(request)
    logger.info(f"[voice] Live snapshot: {snapshot}")
    url = await _fetch_signed_url()
    return SignedUrlResponse(url=url, agent_id=settings.ELEVENLABS_AGENT_ID, dynamic_variables=snapshot)


@router.websocket("/voice/ws")
async def voice_ws_proxy(ws: WebSocket):
    """
    WebSocket proxy — relays audio between browser and ElevenLabs WSS,
    and auto-injects live sensor data as conversation_initiation_client_data.
    Useful for restricted networks or when you need server-side call logging.
    """
    await ws.accept()

    if not settings.ELEVENLABS_API_KEY or not settings.ELEVENLABS_AGENT_ID:
        await ws.send_json({"type": "error", "message": "ElevenLabs not configured"})
        await ws.close(code=1011)
        return

    try:
        snapshot = await _build_live_snapshot(ws)
        signed_url = await _fetch_signed_url()
    except HTTPException as e:
        await ws.send_json({"type": "error", "message": str(e.detail)})
        await ws.close(code=1011)
        return

    import websockets  # deferred — only needed for proxy path

    try:
        async with websockets.connect(signed_url) as el_ws:
            logger.info("[voice/ws] Connected to ElevenLabs")

            # Inject live data immediately after connecting
            await el_ws.send(json.dumps({
                "type": "conversation_initiation_client_data",
                "dynamic_variables": snapshot,
            }))

            async def client_to_el():
                try:
                    while True:
                        data = await ws.receive()
                        if "bytes" in data:
                            await el_ws.send(data["bytes"])
                        elif "text" in data:
                            await el_ws.send(data["text"])
                except WebSocketDisconnect:
                    pass

            async def el_to_client():
                try:
                    async for message in el_ws:
                        if isinstance(message, bytes):
                            await ws.send_bytes(message)
                        else:
                            await ws.send_text(message)
                except Exception:
                    pass

            await asyncio.gather(client_to_el(), el_to_client())

    except Exception as e:
        logger.error(f"[voice/ws] Proxy error: {e}")
        try:
            await ws.send_json({"type": "error", "message": "Connection to ElevenLabs failed"})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("[voice/ws] Connection closed")


