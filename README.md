# SALA Galápagos — Backend API

Plataforma Inteligente de Clima y Ecosistemas para San Cristóbal, Galápagos.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (React)                       │
│   Dashboard │ MANGLE │ SCALESIA │ GALÁPAGO │ GARÚA │ ENCANTADA │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/JSON
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI Application                       │
│                                                              │
│  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ /nowcast │  │ /alerts │  │ /pesca   │  │ /agro       │ │
│  │ (core)   │  │         │  │ (MANGLE) │  │ (SCALESIA)  │ │
│  └────┬─────┘  └────┬────┘  └────┬─────┘  └──────┬──────┘ │
│       │              │            │                │         │
│  ┌────▼──────────────▼────────────▼────────────────▼──────┐ │
│  │              Model Registry (4 models)                 │ │
│  │  heavy_rain │ high_wind │ low_soil_moisture │ fog_event│ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐ │
│  │            Feature Engineering Pipeline                 │ │
│  │   (replicates notebook cell 28 exactly)                │ │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                  │
│  ┌────────────────────────▼───────────────────────────────┐ │
│  │            Data Ingestion (Open-Meteo API)             │ │
│  │   weather: 15-min resolution │ marine: wave/current    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐ │
│  │ /bio     │  │ /risk    │  │ /visit │  │ /chat /voice │ │
│  │(GALÁPAGO)│  │ (GARÚA)  │  │(ENCANT)│  │ (LLM + STT) │ │
│  └──────────┘  └──────────┘  └────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Setup rápido

```bash
# 1. Clonar e instalar
cd sala-backend
pip install -r requirements.txt

# 2. Poner los modelos exportados (Section 12 de cada notebook)
#    Los 4 archivos .pt van en models/
ls models/
#   best_heavy_rain_3h.pt
#   best_high_wind_3h.pt
#   best_low_soil_moisture_3h.pt
#   best_fog_event_3h.pt

# 3. Configurar (opcional — funciona sin API keys con datos de Open-Meteo)
export SALA_ANTHROPIC_API_KEY="sk-ant-..."   # para chatbot
export SALA_OPENAI_API_KEY="sk-..."          # para voz (Whisper STT)

# 4. Correr
uvicorn app.main:app --reload --port 8000

# 5. Ver docs interactivos
open http://localhost:8000/docs
```

## Endpoints

| Método | Ruta | Módulo | Descripción |
|--------|------|--------|-------------|
| GET | `/api/v1/nowcast/{station}` | CORE | Probabilidades de los 4 eventos |
| GET | `/api/v1/alerts` | CORE | Alertas activas agregadas |
| GET | `/api/v1/pesca/score` | MANGLE | Score de pesca 0-100 |
| GET | `/api/v1/agro/calendar` | SCALESIA | Calendario de cultivos + riego |
| GET | `/api/v1/bio/status` | GALÁPAGO | Estado de fauna endémica |
| GET | `/api/v1/risk/zones` | GARÚA | Mapa de riesgo por zona |
| GET | `/api/v1/visit/recommend` | ENCANTADA | Top actividades turísticas |
| POST | `/api/v1/chat` | CHAT | Chatbot conversacional |
| POST | `/api/v1/voice` | VOZ | Audio in → respuesta → audio out |
| GET | `/health` | — | Health check |

## Pipeline de datos (por request)

```
1. Open-Meteo API → 48h de datos horarios
2. Upsample a 15-min (interpolación temporal)
3. Replicar datos en formato multi-estación (cer_, jun_, merc_, mira_)
4. Feature engineering (cell 28 del notebook):
   - Cyclical time features (hour_sin/cos, doy_sin/cos)
   - Wind vector decomposition (wind_x, wind_y)
   - Dewpoint + dewpoint depression (Magnus formula)
   - Soil moisture tendency (diff 3h)
   - Rolling stats: rain_sum, temp_mean/std, wind_mean, rh_mean (1h/3h/6h)
5. Extraer ventana de 96 timesteps (últimas 24h)
6. Normalizar con train_mean/train_std del checkpoint
7. Forward pass por modelo → logit → sigmoid → probabilidad calibrada
8. Comparar con threshold → alerta sí/no
```

## Modelos

Cada modelo es un `RecurrentClassifier` (RNN/LSTM/GRU) con:
- **Encoder:** 2 capas, hidden_dim=128, dropout=0.3
- **Head:** Linear(128, 1) → logit → sigmoid
- **Input:** ventana de (96, N_features) = 24h × 15min
- **Output:** probabilidad de evento en las próximas 3h
- **Calibración:** Platt scaling + optimal threshold (F1-maximizing)

Los checkpoints incluyen `feature_cols`, `train_mean`, `train_std`, y `threshold`,
así que la API no necesita recalcular nada.

## Módulos: de dónde salen los scores

### MANGLE (score pesca)
```
score = 100 × (1 - Σ(weight_i × prob_i))
  wind:  40% × P(high_wind)
  rain:  15% × P(heavy_rain)
  fog:   20% × P(fog_event)
  wave:  25% × wave_risk(Open-Meteo Marine)
```

### GARÚA (riesgo por zona)
```
risk_score = Σ(zone_weight_i × prob_i) × (1 + slope_factor × rain_flag)
```

### SCALESIA (riego)
```
irrigation_need = ET₀(Penman-Monteith) - expected_rain - soil_moisture
```

### GALÁPAGO (estrés fauna)
```
species_risk = Σ(relevant_event_probs) × nesting_amplifier × temp_stress
```

### ENCANTADA (score actividad)
```
activity_score = base_score - Σ(penalty_i × max(0, prob_i - tolerance_i))
```

## Deploy a Cloud Run

```bash
# Build
docker build -t sala-backend .

# Tag
docker tag sala-backend gcr.io/PROJECT_ID/sala-backend

# Push
docker push gcr.io/PROJECT_ID/sala-backend

# Deploy
gcloud run deploy sala-backend \
  --image gcr.io/PROJECT_ID/sala-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --set-env-vars "SALA_ANTHROPIC_API_KEY=sk-ant-..."
```

## Estructura de archivos

```
sala-backend/
├── app/
│   ├── main.py              # FastAPI app + lifespan (model loading)
│   ├── core/
│   │   └── config.py        # Settings (env vars, defaults)
│   ├── ml/
│   │   ├── classifier.py    # RecurrentClassifier + LoadedModel (inference)
│   │   ├── model_registry.py # Loads all .pt files on startup
│   │   └── features.py      # Feature engineering (mirrors notebook cell 28)
│   ├── routers/
│   │   ├── nowcast.py       # GET /nowcast/{station} — core predictions
│   │   ├── alerts.py        # GET /alerts — aggregated alerts
│   │   ├── pesca.py         # GET /pesca/score — MANGLE fishing score
│   │   ├── agro.py          # GET /agro/calendar — SCALESIA crops
│   │   ├── bio.py           # GET /bio/status — GALÁPAGO wildlife
│   │   ├── risk.py          # GET /risk/zones — GARÚA risk map
│   │   ├── visit.py         # GET /visit/recommend — ENCANTADA tourism
│   │   ├── chat.py          # POST /chat — Claude chatbot
│   │   └── voice.py         # POST /voice — Whisper STT + Edge TTS
│   ├── schemas/
│   │   └── responses.py     # Pydantic models for all responses
│   └── services/
│       └── ingestion.py     # Open-Meteo data fetching
├── models/                  # .pt checkpoints (from notebook Section 12)
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```
