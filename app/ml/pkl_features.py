"""
Feature engineering pipeline for the LightGBM v4 rainfall models.

Takes the wide DataFrame produced by fetch_openmeteo_history() and builds
the exact 237-feature vector the pkl models expect.

Rolling window sizes (15-min data):
    30min = 2 rows    1h  = 4 rows    2h  = 8 rows    3h  = 12 rows
    6h  = 24 rows    12h = 48 rows   24h = 96 rows   48h = 192 rows
"""
import numpy as np
import pandas as pd

# ── Feature column list (must match training order exactly) ──────────────────

_RAW_CER = [
    "cer_rain_mm", "cer_temp_c", "cer_rh_avg", "cer_rh_max",
    "cer_solar_kw", "cer_wind_speed_ms", "cer_wind_speed_max", "cer_wind_dir",
    "cer_soil_moist_1", "cer_soil_moist_2", "cer_soil_moist_3",
    "cer_leaf_wetness_mv", "cer_leaf_wet_min", "cer_battery_v",
]
_RAW_JUN = [
    "jun_rain_mm", "jun_temp_c", "jun_rh_avg", "jun_rh_max",
    "jun_solar_kw", "jun_net_rad_wm2", "jun_wind_speed_ms", "jun_wind_speed_max", "jun_wind_dir",
    "jun_soil_moist_1", "jun_soil_moist_2", "jun_soil_moist_3",
    "jun_leaf_wetness_mv", "jun_leaf_wet_min", "jun_battery_v",
]
_RAW_MERC = [
    "merc_rain_mm", "merc_temp_c", "merc_rh_avg", "merc_rh_max",
    "merc_solar_kw", "merc_net_rad_wm2", "merc_wind_speed_ms", "merc_wind_speed_max", "merc_wind_dir",
    "merc_soil_moist_1", "merc_soil_moist_2", "merc_soil_moist_3",
    "merc_leaf_wetness_mv", "merc_leaf_wet_min", "merc_battery_v",
]
_RAW_MIRA = [
    "mira_rain_mm", "mira_temp_c", "mira_rh_avg", "mira_rh_max",
    "mira_solar_kw", "mira_net_rad_wm2", "mira_wind_speed_ms", "mira_wind_speed_max", "mira_wind_dir",
    "mira_soil_moist_1", "mira_soil_moist_2", "mira_soil_moist_3",
    "mira_leaf_wetness_mv", "mira_leaf_wet_min", "mira_battery_v",
]
_MISSING_INDICATORS = [
    "cer_rain_missing", "jun_rain_missing", "merc_rain_missing", "mira_rain_missing",
]
_TIME_CYCLIC = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
]

def _per_station_engineered_cols(stn: str) -> list[str]:
    return [
        f"{stn}_wind_x", f"{stn}_wind_y",
        f"{stn}_wind_std_1h", f"{stn}_wind_mean_3h",
        f"{stn}_dewpoint", f"{stn}_dew_depr",
        f"{stn}_temp_trend_1h",
        f"{stn}_rain_sum_30min", f"{stn}_rain_sum_1h", f"{stn}_rain_sum_2h",
        f"{stn}_rain_sum_3h", f"{stn}_rain_sum_6h", f"{stn}_rain_sum_12h",
        f"{stn}_rain_sum_24h", f"{stn}_rain_sum_48h",
        f"{stn}_rain_tr_1h_3h", f"{stn}_rain_tr_3h_12h", f"{stn}_rain_tr_6h_48h",
        f"{stn}_rain_now",
        f"{stn}_sm1_tend_1h", f"{stn}_sm1_tend_3h", f"{stn}_sm1_tend_6h", f"{stn}_sm1_roll_6h",
        f"{stn}_sm2_tend_1h", f"{stn}_sm2_tend_3h", f"{stn}_sm2_tend_6h", f"{stn}_sm2_roll_6h",
        f"{stn}_sm3_tend_1h", f"{stn}_sm3_tend_3h", f"{stn}_sm3_tend_6h", f"{stn}_sm3_roll_6h",
        f"{stn}_solar_mean_3h", f"{stn}_solar_drop_1h",
        f"{stn}_lw_mean_1h", f"{stn}_lw_mean_6h",
    ]

_GRAD_NAMES = [
    "grad_highland_lowland",   # jun - cer
    "grad_highland_coastal",   # jun - merc
    "grad_mid_coastal",        # merc - cer
    "grad_mid_inland",         # mira - jun
    "grad_summit_diff",        # max(jun,mira) - min(cer,merc)
]
_GRAD_HORIZONS = ["1h", "3h", "6h", "12h"]
_GRADIENT_COLS = [f"{g}_{w}" for g in _GRAD_NAMES for w in _GRAD_HORIZONS]

_AGGREGATE_COLS = ["avg_soil_highland", "avg_soil_lowland"]
_LAG_COLS = [
    "lag_merc_jun_15m", "lag_merc_jun_1h",
    "lag_mira_cer_15m", "lag_mira_cer_1h",
    "lag_mira_jun_1h", "lag_mira_jun_3h",
]

FEATURE_COLS: list[str] = (
    _RAW_CER + _RAW_JUN + _RAW_MERC + _RAW_MIRA
    + _MISSING_INDICATORS
    + _TIME_CYCLIC
    + _per_station_engineered_cols("cer")
    + _per_station_engineered_cols("jun")
    + _per_station_engineered_cols("merc")
    + _per_station_engineered_cols("mira")
    + _GRADIENT_COLS
    + _AGGREGATE_COLS
    + _LAG_COLS
)

assert len(FEATURE_COLS) == 237, f"Expected 237 features, got {len(FEATURE_COLS)}"
assert len(set(FEATURE_COLS)) == 237, "Duplicate feature names detected"

# ── Rolling window sizes at 15-min resolution ─────────────────────────────────
W = {
    "30min": 2,
    "1h": 4,
    "2h": 8,
    "3h": 12,
    "6h": 24,
    "12h": 48,
    "24h": 96,
    "48h": 192,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Return column from df or a constant Series if missing."""
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _roll_sum(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).sum()


def _roll_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def _roll_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).std().fillna(0.0)


def _tend(s: pd.Series, shift: int) -> pd.Series:
    """Difference vs. `shift` steps ago (trend feature)."""
    return (s - s.shift(shift)).fillna(0.0)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_pkl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all 237 features from the wide DataFrame returned by
    fetch_openmeteo_history().

    Args:
        df: wide DataFrame indexed by datetime with columns like
            {stn}_rain_mm, {stn}_temp_c, {stn}_rh_avg, etc.

    Returns:
        DataFrame with exactly FEATURE_COLS columns (237), same index as df.
    """
    out = pd.DataFrame(index=df.index)

    # ── 1. Raw per-station columns ────────────────────────────────────────────
    for stn in ["cer", "jun", "merc", "mira"]:
        out[f"{stn}_rain_mm"]      = _col(df, f"{stn}_rain_mm")
        out[f"{stn}_temp_c"]       = _col(df, f"{stn}_temp_c")
        out[f"{stn}_rh_avg"]       = _col(df, f"{stn}_rh_avg")
        out[f"{stn}_rh_max"]       = _col(df, f"{stn}_rh_max", default=_col(df, f"{stn}_rh_avg").iloc[-1])
        # rh_max: prefer dedicated column, fall back to rh_avg
        if f"{stn}_rh_max" in df.columns:
            out[f"{stn}_rh_max"] = df[f"{stn}_rh_max"].astype(float)
        else:
            out[f"{stn}_rh_max"] = _col(df, f"{stn}_rh_avg")

        out[f"{stn}_solar_kw"]     = _col(df, f"{stn}_solar_kw")
        out[f"{stn}_wind_speed_ms"]  = _col(df, f"{stn}_wind_speed_ms")
        out[f"{stn}_wind_speed_max"] = _col(df, f"{stn}_wind_speed_ms")  # approximation: no gust data
        out[f"{stn}_wind_dir"]     = _col(df, f"{stn}_wind_dir")
        # soil moisture rename: soil_moisture_N → soil_moist_N
        out[f"{stn}_soil_moist_1"] = _col(df, f"{stn}_soil_moisture_1")
        out[f"{stn}_soil_moist_2"] = _col(df, f"{stn}_soil_moisture_2")
        out[f"{stn}_soil_moist_3"] = _col(df, f"{stn}_soil_moisture_3")
        # sensors unavailable from Open-Meteo
        out[f"{stn}_leaf_wetness_mv"] = 0.0
        out[f"{stn}_leaf_wet_min"]    = 0.0
        out[f"{stn}_battery_v"]       = 0.0
        # net_rad only for jun / merc / mira
        if stn != "cer":
            out[f"{stn}_net_rad_wm2"] = out[f"{stn}_solar_kw"] * 1000.0 * 0.5

    # ── 2. Missing indicators ─────────────────────────────────────────────────
    for stn in ["cer", "jun", "merc", "mira"]:
        out[f"{stn}_rain_missing"] = _col(df, f"{stn}_rain_missing", default=0.0)

    # ── 3. Cyclical time features ─────────────────────────────────────────────
    idx = df.index
    hour  = idx.hour + idx.minute / 60.0
    doy   = idx.dayofyear.astype(float)
    month = idx.month.astype(float)

    out["hour_sin"]  = np.sin(2 * np.pi * hour  / 24.0)
    out["hour_cos"]  = np.cos(2 * np.pi * hour  / 24.0)
    out["doy_sin"]   = np.sin(2 * np.pi * doy   / 365.0)
    out["doy_cos"]   = np.cos(2 * np.pi * doy   / 365.0)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # ── 4. Per-station engineered features ───────────────────────────────────
    for stn in ["cer", "jun", "merc", "mira"]:
        rain   = out[f"{stn}_rain_mm"]
        temp   = out[f"{stn}_temp_c"]
        rh     = out[f"{stn}_rh_avg"]
        wind_s = out[f"{stn}_wind_speed_ms"]
        wind_d = out[f"{stn}_wind_dir"]
        solar  = out[f"{stn}_solar_kw"]
        sm1    = out[f"{stn}_soil_moist_1"]
        sm2    = out[f"{stn}_soil_moist_2"]
        sm3    = out[f"{stn}_soil_moist_3"]
        lw     = pd.Series(0.0, index=df.index)  # leaf wetness unavailable

        # Wind components
        wind_d_rad = np.deg2rad(wind_d)
        out[f"{stn}_wind_x"] = wind_s * np.cos(wind_d_rad)
        out[f"{stn}_wind_y"] = wind_s * np.sin(wind_d_rad)

        # Wind rolling stats
        out[f"{stn}_wind_std_1h"]   = _roll_std(wind_s, W["1h"])
        out[f"{stn}_wind_mean_3h"]  = _roll_mean(wind_s, W["3h"])

        # Dewpoint (simplified: Td ≈ T - (100 - RH) / 5)
        dewpoint = temp - ((100.0 - rh) / 5.0)
        out[f"{stn}_dewpoint"]  = dewpoint
        out[f"{stn}_dew_depr"]  = temp - dewpoint

        # Temperature trend over 1h
        out[f"{stn}_temp_trend_1h"] = _tend(temp, W["1h"])

        # Rain accumulations
        out[f"{stn}_rain_sum_30min"] = _roll_sum(rain, W["30min"])
        out[f"{stn}_rain_sum_1h"]    = _roll_sum(rain, W["1h"])
        out[f"{stn}_rain_sum_2h"]    = _roll_sum(rain, W["2h"])
        out[f"{stn}_rain_sum_3h"]    = _roll_sum(rain, W["3h"])
        out[f"{stn}_rain_sum_6h"]    = _roll_sum(rain, W["6h"])
        out[f"{stn}_rain_sum_12h"]   = _roll_sum(rain, W["12h"])
        out[f"{stn}_rain_sum_24h"]   = _roll_sum(rain, W["24h"])
        out[f"{stn}_rain_sum_48h"]   = _roll_sum(rain, W["48h"])

        # Rain transition ratios (avoid division by zero)
        eps = 1e-6
        out[f"{stn}_rain_tr_1h_3h"]  = out[f"{stn}_rain_sum_1h"]  / (out[f"{stn}_rain_sum_3h"]  + eps)
        out[f"{stn}_rain_tr_3h_12h"] = out[f"{stn}_rain_sum_3h"]  / (out[f"{stn}_rain_sum_12h"] + eps)
        out[f"{stn}_rain_tr_6h_48h"] = out[f"{stn}_rain_sum_6h"]  / (out[f"{stn}_rain_sum_48h"] + eps)

        # Current rain (alias)
        out[f"{stn}_rain_now"] = rain

        # Soil moisture tendencies + rolling means
        for sm_col, sm_ser in [("sm1", sm1), ("sm2", sm2), ("sm3", sm3)]:
            out[f"{stn}_{sm_col}_tend_1h"] = _tend(sm_ser, W["1h"])
            out[f"{stn}_{sm_col}_tend_3h"] = _tend(sm_ser, W["3h"])
            out[f"{stn}_{sm_col}_tend_6h"] = _tend(sm_ser, W["6h"])
            out[f"{stn}_{sm_col}_roll_6h"] = _roll_mean(sm_ser, W["6h"])

        # Solar features
        out[f"{stn}_solar_mean_3h"]  = _roll_mean(solar, W["3h"])
        out[f"{stn}_solar_drop_1h"]  = _tend(solar, W["1h"])

        # Leaf wetness rolling means (always 0)
        out[f"{stn}_lw_mean_1h"] = _roll_mean(lw, W["1h"])
        out[f"{stn}_lw_mean_6h"] = _roll_mean(lw, W["6h"])

    # ── 5. Cross-station gradient features ───────────────────────────────────
    jun_r  = out["jun_rain_mm"]
    cer_r  = out["cer_rain_mm"]
    merc_r = out["merc_rain_mm"]
    mira_r = out["mira_rain_mm"]

    gradients = {
        "grad_highland_lowland": jun_r  - cer_r,
        "grad_highland_coastal": jun_r  - merc_r,
        "grad_mid_coastal":      merc_r - cer_r,
        "grad_mid_inland":       mira_r - jun_r,
        "grad_summit_diff":      pd.concat([jun_r, mira_r], axis=1).max(axis=1)
                                 - pd.concat([cer_r, merc_r], axis=1).min(axis=1),
    }
    _win_map = {"1h": W["1h"], "3h": W["3h"], "6h": W["6h"], "12h": W["12h"]}
    for grad_name, grad_series in gradients.items():
        for label, n in _win_map.items():
            out[f"{grad_name}_{label}"] = _roll_mean(grad_series, n)

    # ── 6. Cross-station aggregates ───────────────────────────────────────────
    out["avg_soil_highland"] = (out["jun_soil_moist_1"] + out["mira_soil_moist_1"]) / 2.0
    out["avg_soil_lowland"]  = (out["cer_soil_moist_1"] + out["merc_soil_moist_1"]) / 2.0

    # ── 7. Lag features ───────────────────────────────────────────────────────
    out["lag_merc_jun_15m"] = (merc_r.shift(1)  - jun_r).fillna(0.0)
    out["lag_merc_jun_1h"]  = (merc_r.shift(W["1h"])  - jun_r).fillna(0.0)
    out["lag_mira_cer_15m"] = (mira_r.shift(1)  - cer_r).fillna(0.0)
    out["lag_mira_cer_1h"]  = (mira_r.shift(W["1h"])  - cer_r).fillna(0.0)
    out["lag_mira_jun_1h"]  = (mira_r.shift(W["1h"])  - jun_r).fillna(0.0)
    out["lag_mira_jun_3h"]  = (mira_r.shift(W["3h"])  - jun_r).fillna(0.0)

    # ── Final selection + NaN fill ────────────────────────────────────────────
    result = out[FEATURE_COLS].fillna(0.0)
    return result


def extract_latest_row(feat_df: pd.DataFrame) -> np.ndarray:
    """Return the last row of an engineered feature DataFrame as shape (1, 237).

    Args:
        feat_df: output of build_pkl_features()

    Returns:
        numpy array of shape (1, 237), dtype float64, NaN-free.
    """
    return feat_df[FEATURE_COLS].iloc[[-1]].fillna(0.0).values.astype(np.float64)
