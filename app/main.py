"""
SALA Galápagos — Backend API
Plataforma Inteligente de Clima y Ecosistemas para San Cristóbal, Galápagos

Modules:
  - GARÚA    (¿Voy a la playa?)     → Risk zones & disaster management
  - MANGLE   (¿Salgo a pescar?)     → Fishing safety assistant  
  - SCALESIA (¿Qué siembro hoy?)    → Smart crop calendar
  - GALÁPAGO (¿Cómo están las tortugas?) → Endemic wildlife protection
  - ENCANTADA (¿Qué hago hoy?)      → Smart tourism planner
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.ml.model_registry import ModelRegistry
from app.ml.pkl_registry import PklRegistry
from app.routers import nowcast, alerts, pesca, agro, bio, risk, chat, voice, weather
from app.routers import v4rainfall

logger = logging.getLogger(__name__)

# ── Lifespan: load models on startup ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models into memory on startup, release on shutdown."""
    logger.info("Loading ML models...")
    registry = ModelRegistry(settings.MODELS_DIR)
    registry.load_all()
    app.state.models = registry
    logger.info(f"Loaded {len(registry.models)} models: {list(registry.models.keys())}")

    pkl_registry = PklRegistry(settings.MODELS_DIR)
    pkl_registry.load_all()
    app.state.pkl_models = pkl_registry
    logger.info(f"Loaded {len(pkl_registry.models)} pkl models")
    yield
    logger.info("Shutting down, releasing models...")
    del app.state.models

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="SALA Galápagos API",
    description=__doc__,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────
app.include_router(nowcast.router,  prefix="/api/v1", tags=["Nowcast Core"])
app.include_router(alerts.router,   prefix="/api/v1", tags=["Alerts"])
app.include_router(pesca.router,    prefix="/api/v1/pesca",  tags=["MANGLE — Pesca"])
app.include_router(agro.router,     prefix="/api/v1/agro",   tags=["SCALESIA — Agro"])
app.include_router(bio.router,      prefix="/api/v1/bio",    tags=["GALÁPAGO — Bio"])
app.include_router(risk.router,     prefix="/api/v1/risk",   tags=["GARÚA — Risk"])
app.include_router(chat.router,     prefix="/api/v1",        tags=["Chat & Voice"])
app.include_router(voice.router,    prefix="/api/v1",        tags=["Chat & Voice"])
app.include_router(weather.router,  prefix="/api/v1",        tags=["Weather & Grid"])
app.include_router(v4rainfall.router, prefix="/api/v1/v4",   tags=["V4 Rainfall — LightGBM"])

@app.get("/health")
async def health():
    models_loaded = hasattr(app.state, "models") and len(app.state.models.models) > 0
    return {"status": "ok", "models_loaded": models_loaded}
