"""
PklRegistry — loads the 12 LightGBM rainfall pkl models at startup and
provides a single predict() entry point.

Each pkl file is a dict:
  model         : lightgbm.Booster  (3-class: no_rain / light_rain / heavy_rain)
  tl            : float  low probability threshold  (no_rain → light_rain)
  th            : float  high probability threshold (light_rain → heavy_rain)
  feature_cols  : list[str]  237 feature names in exact model order
"""
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STATIONS = ["cer", "jun", "merc", "mira"]
HORIZONS = [1, 3, 6]

CLASS_LABELS = {0: "Sin lluvia", 1: "Lluvia leve", 2: "Lluvia intensa"}


class PklRegistry:
    """Loads and serves all 12 pkl LightGBM rainfall models."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        # key: (station, horizon_h)  →  {"model": Booster, "tl": float, "th": float, "feature_cols": list}
        self.models: dict[tuple, dict] = {}

    def load_all(self):
        """Scan models_dir for v4_model_*.pkl files and load each one."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return

        for stn in STATIONS:
            for h in HORIZONS:
                path = self.models_dir / f"v4_model_{stn}_{h}h.pkl"
                if not path.exists():
                    logger.warning(f"  ⚠ pkl model not found, skipping: {path}")
                    continue
                try:
                    with open(path, "rb") as fh:
                        entry = pickle.load(fh)
                    self.models[(stn, h)] = entry
                    n_feats = len(entry.get("feature_cols", []))
                    logger.info(
                        f"  ✓ pkl {stn} {h}h — tl={entry.get('tl')}, "
                        f"th={entry.get('th')}, features={n_feats}"
                    )
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {path.name}: {e}")

    def predict(self, station: str, horizon: int, feature_row: np.ndarray) -> dict:
        """Run the LightGBM model for the given station / horizon.

        Args:
            station     : one of cer | jun | merc | mira
            horizon     : 1, 3, or 6 (hours)
            feature_row : numpy array of shape (1, 237)

        Returns:
            {
                "pred_class"       : int   0 / 1 / 2
                "pred_prob_no_rain": float
                "pred_prob_light"  : float
                "pred_prob_heavy"  : float
                "pred_prob"        : float  (probability of the predicted class)
                "class_label"      : str
                "tl"               : float
                "th"               : float
            }
        """
        key = (station, horizon)
        if key not in self.models:
            raise ValueError(
                f"No pkl model loaded for station='{station}' horizon={horizon}h. "
                f"Available: {list(self.models.keys())}"
            )

        entry = self.models[key]
        booster = entry["model"]
        tl: float = entry.get("tl", 0.30)
        th: float = entry.get("th", 0.39)

        # booster.predict() returns shape (n_samples, n_classes) for multiclass
        probs = booster.predict(feature_row)  # (1, 3)
        p_no_rain, p_light, p_heavy = float(probs[0][0]), float(probs[0][1]), float(probs[0][2])

        pred_class = int(np.argmax([p_no_rain, p_light, p_heavy]))

        return {
            "pred_class": pred_class,
            "pred_prob_no_rain": round(p_no_rain, 6),
            "pred_prob_light": round(p_light, 6),
            "pred_prob_heavy": round(p_heavy, 6),
            "pred_prob": round([p_no_rain, p_light, p_heavy][pred_class], 6),
            "class_label": CLASS_LABELS[pred_class],
            "tl": tl,
            "th": th,
        }
