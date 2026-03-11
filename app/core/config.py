"""Central configuration — reads from env vars with sane defaults."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Paths ─────────────────────────────────────────────────────
    MODELS_DIR: str = "models"             # directory with .pt checkpoints
    DATA_DIR: str = "app/data"             # static JSONs (species, crops, POIs)

    # ── API keys (optional — only needed for enhanced features) ───
    OPENMETEO_BASE: str = "https://api.open-meteo.com/v1"
    OPENAI_API_KEY: str = ""               # for chatbot (OpenAI)

    # ── ElevenLabs (real-time voice calls) ───────────────────────
    ELEVENLABS_API_KEY: str | None = None  # Voice (ElevenLabs)
    ELEVENLABS_AGENT_ID: str | None = None # Conversational AI agent ID

    # Weather (Google)
    GOOGLE_WEATHER_API_KEY: str | None = None

    # ── Redis cache (optional) ────────────────────────────────────
    REDIS_URL: str = ""                    # e.g. redis://10.0.0.2:6379
    CACHE_TTL_NOWCAST: int = 900           # 15 min
    CACHE_TTL_SCORES: int = 3600           # 1 hour
    CACHE_TTL_STATIC: int = 21600          # 6 hours

    # ── Server ────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["*"]
    LOG_LEVEL: str = "INFO"

    # ── Stations ──────────────────────────────────────────────────
    TARGET_STATION: str = "jun"
    STATIONS: list[str] = ["cer", "jun", "merc", "mira"]
    LOOKBACK: int = 96                     # 24h at 15-min resolution

    # ── San Cristóbal coordinates ─────────────────────────────────
    ISLAND_LAT: float = -0.9
    ISLAND_LON: float = -89.6

    model_config = {"env_prefix": "SALA_", "env_file": ".env"}


settings = Settings()
