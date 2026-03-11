"""
Feature engineering — replicates the exact transformations from notebook cell 28.

Takes raw station data (DataFrame with 15-min rows) and produces the same
derived features the model was trained on.
"""
import numpy as np
import pandas as pd

from app.core.config import settings

STATIONS = settings.STATIONS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering from the training notebook.

    Args:
        df: Wide DataFrame with columns like {stn}_{var} (e.g. jun_temp_c).
            Must have a DatetimeIndex at 15-min resolution.
    Returns:
        DataFrame with all derived features added.
    """
    # ── Cyclical time features ────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    for stn in STATIONS:
        # ── Wind vector decomposition ─────────────────────────────
        wd_col, ws_col = f"{stn}_wind_dir", f"{stn}_wind_speed_ms"
        if wd_col in df.columns and ws_col in df.columns:
            wd_rad = np.deg2rad(df[wd_col])
            df[f"{stn}_wind_x"] = df[ws_col] * np.cos(wd_rad)
            df[f"{stn}_wind_y"] = df[ws_col] * np.sin(wd_rad)

        # ── Dewpoint (Magnus formula) ─────────────────────────────
        temp_col, rh_col = f"{stn}_temp_c", f"{stn}_rh_avg"
        if temp_col in df.columns and rh_col in df.columns:
            T = df[temp_col]
            RH = df[rh_col].clip(lower=1)
            alpha = (17.27 * T) / (237.3 + T) + np.log(RH / 100)
            df[f"{stn}_dewpoint"] = (237.3 * alpha) / (17.27 - alpha)
            df[f"{stn}_dewpoint_depression"] = T - df[f"{stn}_dewpoint"]

        # ── Soil moisture tendency ────────────────────────────────
        sm_col = f"{stn}_soil_moisture_1"
        if sm_col in df.columns:
            df[f"{stn}_soil_moist_tend_3h"] = df[sm_col].diff(periods=12)

        # ── Rolling statistics (1h, 3h, 6h) ──────────────────────
        for window, wlabel in [(4, "1h"), (12, "3h"), (24, "6h")]:
            rain_col = f"{stn}_rain_mm"
            if rain_col in df.columns:
                df[f"{stn}_rain_sum_{wlabel}"] = (
                    df[rain_col].rolling(window, min_periods=1).sum()
                )
            if temp_col in df.columns:
                df[f"{stn}_temp_mean_{wlabel}"] = (
                    df[temp_col].rolling(window, min_periods=1).mean()
                )
                df[f"{stn}_temp_std_{wlabel}"] = (
                    df[temp_col].rolling(window, min_periods=1).std()
                )
            if ws_col in df.columns:
                df[f"{stn}_wind_mean_{wlabel}"] = (
                    df[ws_col].rolling(window, min_periods=1).mean()
                )
            if rh_col in df.columns:
                df[f"{stn}_rh_mean_{wlabel}"] = (
                    df[rh_col].rolling(window, min_periods=1).mean()
                )

    return df


def extract_window(df: pd.DataFrame, feature_cols: list[str],
                    lookback: int = 96) -> np.ndarray | None:
    """Extract the last `lookback` rows as a numpy window for model input.

    Returns None if there aren't enough rows.
    """
    if len(df) < lookback:
        return None

    # Ensure all feature_cols exist (fill missing with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    window = df[feature_cols].iloc[-lookback:].values.astype(np.float32)

    # Replace any remaining NaN with 0
    window = np.nan_to_num(window, nan=0.0)
    return window
