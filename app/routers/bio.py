"""
/api/v1/bio — GALÁPAGO: ¿Cómo están las tortugas?
Endemic wildlife protection monitoring.
"""
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException

from app.schemas.responses import BioStatusResponse, SpeciesStatus
from app.services.ingestion import fetch_openmeteo_history
from app.ml.features import engineer_features, extract_window
from app.core.config import settings

router = APIRouter()

# Species database with environmental tolerance ranges
SPECIES = [
    {
        "name": "Tortuga Gigante de San Cristóbal",
        "scientific": "Chelonoidis chathamensis",
        "temp_stress_high": 33,     # °C — above this, egg sex ratio skews
        "temp_stress_low": 15,
        "nesting_months": [1, 2, 3, 4, 5, 6],
        "rain_sensitivity": "low",
        "module_factors": ["temp_extreme", "heavy_rain", "low_soil_moisture"],
    },
    {
        "name": "Iguana Marina",
        "scientific": "Amblyrhynchus cristatus",
        "temp_stress_high": 35,     # body temp above this = severe stress
        "temp_stress_low": 20,      # below this, can't forage effectively
        "nesting_months": [1, 2, 3],
        "rain_sensitivity": "medium",
        "module_factors": ["temp_extreme", "high_wind"],
    },
    {
        "name": "Fragata Magnífica",
        "scientific": "Fregata magnificens",
        "temp_stress_high": 38,
        "temp_stress_low": 18,
        "nesting_months": [3, 4, 5, 6, 7],
        "rain_sensitivity": "high",  # chicks vulnerable to heavy rain
        "module_factors": ["heavy_rain", "high_wind", "fog_event"],
    },
    {
        "name": "Piquero Patas Azules",
        "scientific": "Sula nebouxii",
        "temp_stress_high": 36,
        "temp_stress_low": 16,
        "nesting_months": [5, 6, 7, 8, 9, 10, 11],
        "rain_sensitivity": "medium",
        "module_factors": ["high_wind", "fog_event"],
    },
    {
        "name": "Lobo Marino de Galápagos",
        "scientific": "Zalophus wollebaeki",
        "temp_stress_high": 30,     # sensitive to warming
        "temp_stress_low": 14,
        "nesting_months": [7, 8, 9, 10, 11],
        "rain_sensitivity": "low",
        "module_factors": ["temp_extreme", "high_wind"],
    },
    {
        "name": "Pinzón de Darwin",
        "scientific": "Geospiza spp.",
        "temp_stress_high": 35,
        "temp_stress_low": 14,
        "nesting_months": [1, 2, 3, 4],  # wet season
        "rain_sensitivity": "high",       # Philornis downsi fly larvae in nests
        "module_factors": ["heavy_rain", "fog_event"],
    },
]


@router.get("/status", response_model=BioStatusResponse)
async def get_bio_status(request: Request):
    """Get wildlife status based on current and predicted conditions."""
    registry = request.app.state.models

    raw_df = await fetch_openmeteo_history(hours_back=48)
    df = engineer_features(raw_df)

    stn = settings.TARGET_STATION
    latest = raw_df.iloc[-1]
    temp = latest.get(f"{stn}_temp_c", 24)
    solar = latest.get(f"{stn}_solar_kw", 0.3)

    # Get predictions
    predictions = {}
    if registry.models:
        first_model = next(iter(registry.models.values()))
        window = extract_window(df, first_model.feature_cols, first_model.lookback)
        if window is not None:
            predictions = registry.predict_all(window)

    now = datetime.utcnow()
    month = now.month

    # Thermal stress index (0-1, where 1 = extreme)
    if temp > 32:
        thermal_stress = min(1.0, (temp - 32) / 8)
    elif temp < 16:
        thermal_stress = min(1.0, (16 - temp) / 8)
    else:
        thermal_stress = 0.0

    species_list = []
    any_nesting = False

    for sp in SPECIES:
        # Nesting phase
        is_nesting = month in sp["nesting_months"]
        if is_nesting:
            any_nesting = True
            reproductive_phase = "Temporada de anidación activa"
        else:
            reproductive_phase = "Fuera de temporada reproductiva"

        # Compute risk from relevant model predictions
        risk_score = 0.0
        stress_factors = []

        for factor in sp["module_factors"]:
            prob = predictions.get(factor, {}).get("probability", 0) or 0
            risk_score += prob

            if prob > 0.5:
                factor_names = {
                    "heavy_rain": "lluvia intensa",
                    "high_wind": "viento fuerte",
                    "fog_event": "niebla/garúa",
                    "low_soil_moisture": "sequía",
                    "temp_extreme": "temperatura extrema",
                }
                stress_factors.append(factor_names.get(factor, factor))

        # Temperature-based stress
        if temp > sp["temp_stress_high"]:
            risk_score += 0.5
            stress_factors.append(f"calor ({temp:.1f}°C > {sp['temp_stress_high']}°C)")
        elif temp < sp["temp_stress_low"]:
            risk_score += 0.3
            stress_factors.append(f"frío ({temp:.1f}°C < {sp['temp_stress_low']}°C)")

        # Nesting amplifies risk
        if is_nesting:
            risk_score *= 1.3

        # Classify risk level
        risk_score = min(risk_score, 3.0) / 3.0  # normalize to 0-1
        if risk_score > 0.7:
            risk_level = "critical"
        elif risk_score > 0.4:
            risk_level = "alert"
        elif risk_score > 0.2:
            risk_level = "watch"
        else:
            risk_level = "normal"

        detail_parts = []
        if stress_factors:
            detail_parts.append(f"Estrés por: {', '.join(stress_factors)}")
        if is_nesting:
            detail_parts.append("Período reproductivo — mayor vulnerabilidad")
        if not detail_parts:
            detail_parts.append("Condiciones normales para la especie")

        species_list.append(SpeciesStatus(
            species=sp["name"],
            risk_level=risk_level,
            stress_factor=", ".join(stress_factors) if stress_factors else None,
            reproductive_phase=reproductive_phase,
            detail=". ".join(detail_parts),
        ))

    return BioStatusResponse(
        timestamp=str(df.index[-1]),
        species=species_list,
        nesting_active=any_nesting,
        thermal_stress_index=round(thermal_stress, 3),
    )
