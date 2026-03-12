"""
Historical climate events — ERA5 daily data for Galápagos.

Endpoints:
  GET /api/v1/historical                  → list of available events + metadata
  GET /api/v1/historical/{event}          → full daily series (optionally filtered)
  GET /api/v1/historical/{event}/summary  → aggregate stats (mean/max/min)

Events available:
  el_nino_1982_83   El Niño 1982–1983  (Sep 1982 – Jun 1983, 303 days)
  el_nino_1997_98   El Niño 1997–1998  (Sep 1997 – Jul 1998, 457 days)
  el_nino_2015_16   El Niño 2015–2016  (Sep 2015 – Jun 2016, 488 days)
  la_nina_2010_11   La Niña 2010–2011  (Jun 2010 – Apr 2011, 334 days)
  floods_2023_24    Inundaciones 2023–2024 (Oct 2023 – Feb 2024, 152 days)
"""
import csv
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Event metadata ─────────────────────────────────────────────────────────────

EVENTS: dict[str, dict] = {
    "el_nino_1982_83": {
        "id":          "el_nino_1982_83",
        "label":       "El Niño 1982–1983",
        "type":        "el_nino",
        "description": "Uno de los El Niño más intensos del siglo XX. Lluvias extremas en Galápagos.",
        "file":        "el_nino_1982_83.csv",
    },
    "el_nino_1997_98": {
        "id":          "el_nino_1997_98",
        "label":       "El Niño 1997–1998",
        "type":        "el_nino",
        "description": "El Niño más intenso registrado. Inundaciones masivas y daños severos.",
        "file":        "el_nino_1997_98.csv",
    },
    "el_nino_2015_16": {
        "id":          "el_nino_2015_16",
        "label":       "El Niño 2015–2016",
        "type":        "el_nino",
        "description": "Tercer El Niño más fuerte en el registro histórico.",
        "file":        "el_nino_2015_16.csv",
    },
    "la_nina_2010_11": {
        "id":          "la_nina_2010_11",
        "label":       "La Niña 2010–2011",
        "type":        "la_nina",
        "description": "Período de sequía y temperaturas frías en el Pacífico ecuatorial.",
        "file":        "la_nina_2010_11.csv",
    },
    "floods_2023_24": {
        "id":          "floods_2023_24",
        "label":       "Inundaciones 2023–2024",
        "type":        "floods",
        "description": "Evento de inundaciones recientes en San Cristóbal asociado a El Niño 2023.",
        "file":        "floods_2023_24.csv",
    },
}

# In-memory cache: event_id → list of row dicts
_cache: dict[str, list] = {}

# Columns to expose with clean names
_COL_MAP = {
    "valid_time":   "date",
    "t2m_c":        "temp_c",
    "d2m_c":        "dewpoint_c",
    "sst_c":        "sst_c",
    "wind_speed_ms":"wind_ms",
    "wind_dir_deg": "wind_dir_deg",
    "sp_hpa":       "pressure_hpa",
    "msl_hpa":      "msl_hpa",
    "tp":           "_tp_raw",   # m → mm conversion done below
    "ssrd":         "_ssrd_raw", # J/m² → W/m² conversion done below
}
_NUMERIC = {"temp_c", "dewpoint_c", "sst_c", "wind_ms", "wind_dir_deg",
            "pressure_hpa", "msl_hpa", "precip_mm", "solar_wm2"}


def _load_event(event_id: str) -> list[dict]:
    """Load CSV into memory, apply unit conversions, cache result."""
    if event_id in _cache:
        return _cache[event_id]

    meta = EVENTS[event_id]
    path = Path(settings.DATA_DIR) / meta["file"]
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row: dict = {}
            for src, dst in _COL_MAP.items():
                val = raw.get(src, "")
                row[dst] = val
            # Unit conversions
            try:
                row["precip_mm"] = round(float(row.pop("_tp_raw", 0) or 0) * 1000, 3)
            except (ValueError, TypeError):
                row["precip_mm"] = 0.0
            try:
                # ssrd is accumulated J/m² over 24h → mean W/m²
                row["solar_wm2"] = round(float(row.pop("_ssrd_raw", 0) or 0) / 86400, 2)
            except (ValueError, TypeError):
                row["solar_wm2"] = 0.0
            # Round numeric fields
            for field in _NUMERIC:
                if field in row and row[field] not in ("", None):
                    try:
                        row[field] = round(float(row[field]), 3)
                    except (ValueError, TypeError):
                        pass
            rows.append(row)

    _cache[event_id] = rows
    logger.info(f"[historical] loaded {event_id}: {len(rows)} rows")
    return rows


def _add_meta(rows: list[dict], event_id: str) -> dict:
    meta = EVENTS[event_id].copy()
    if rows:
        meta["date_start"] = rows[0]["date"]
        meta["date_end"]   = rows[-1]["date"]
        meta["total_days"] = len(rows)
    return meta


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/historical")
async def list_events():
    """List all available historical climate events with metadata."""
    result = []
    for event_id, meta in EVENTS.items():
        try:
            rows = _load_event(event_id)
            entry = meta.copy()
            entry.pop("file", None)
            entry["date_start"] = rows[0]["date"] if rows else None
            entry["date_end"]   = rows[-1]["date"] if rows else None
            entry["total_days"] = len(rows)
        except Exception as e:
            entry = meta.copy()
            entry.pop("file", None)
            entry["error"] = str(e)
        result.append(entry)
    return {"events": result, "count": len(result)}


@router.get("/historical/{event_id}/summary")
async def get_event_summary(event_id: str):
    """Aggregate statistics (mean/max/min) for a historical event."""
    if event_id not in EVENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Event '{event_id}' not found. Available: {list(EVENTS.keys())}",
        )
    try:
        rows = _load_event(event_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not rows:
        raise HTTPException(status_code=422, detail="Event dataset is empty")

    # Compute stats for numeric columns
    stats: dict[str, dict] = {}
    for field in _NUMERIC:
        values = []
        for r in rows:
            v = r.get(field)
            if v not in (None, ""):
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    pass
        if values:
            stats[field] = {
                "mean": round(sum(values) / len(values), 3),
                "max":  round(max(values), 3),
                "min":  round(min(values), 3),
            }

    meta = _add_meta(rows, event_id)
    meta.pop("file", None)
    return {"event": meta, "stats": stats}


@router.get("/historical/{event_id}")
async def get_event_data(
    event_id: str,
    from_date: Optional[str] = Query(default=None, alias="from", description="Start date YYYY-MM-DD"),
    to_date:   Optional[str] = Query(default=None, alias="to",   description="End date YYYY-MM-DD"),
    limit:     int           = Query(default=0,    description="Max rows to return (0 = all)"),
):
    """Daily ERA5 series for a historical event.

    Optionally filter with ?from=YYYY-MM-DD&to=YYYY-MM-DD.
    Use ?limit=N to cap the response size.
    """
    if event_id not in EVENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Event '{event_id}' not found. Available: {list(EVENTS.keys())}",
        )
    try:
        rows = _load_event(event_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Date filtering (string comparison works because format is ISO YYYY-MM-DD HH:MM:SS)
    if from_date:
        rows = [r for r in rows if r["date"] >= from_date]
    if to_date:
        rows = [r for r in rows if r["date"] <= to_date + " 23:59:59"]
    if limit and limit > 0:
        rows = rows[:limit]

    meta = _add_meta(rows, event_id)
    meta.pop("file", None)

    return {
        "event":  meta,
        "count":  len(rows),
        "data":   rows,
    }
