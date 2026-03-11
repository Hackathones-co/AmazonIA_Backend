"""
Model Registry — loads all .pt checkpoints from the models/ directory
and provides a clean interface for the API layer.
"""
import logging
from pathlib import Path

from app.ml.classifier import LoadedModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton-like registry that holds all loaded models.

    Expected checkpoint files (produced by Section 12 of each notebook):
        models/best_heavy_rain_3h.pt
        models/best_high_wind_3h.pt
        models/best_low_soil_moisture_3h.pt
        models/best_fog_event_3h.pt
    """

    def __init__(self, models_dir: str = "models", device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.models: dict[str, LoadedModel] = {}

    def load_all(self):
        """Scan models_dir for .pt files and load each one."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return

        for pt_file in sorted(self.models_dir.glob("*.pt")):
            try:
                loaded = LoadedModel(str(pt_file), device=self.device)
                self.models[loaded.event_type] = loaded
                logger.info(
                    f"  ✓ {loaded.event_type} ({loaded.model.rnn_type.upper()}) "
                    f"— {len(loaded.feature_cols)} features, "
                    f"threshold={loaded.threshold:.4f}"
                )
            except Exception as e:
                logger.error(f"  ✗ Failed to load {pt_file.name}: {e}")

    def get(self, event_type: str) -> LoadedModel | None:
        return self.models.get(event_type)

    def predict_all(self, window) -> dict[str, dict]:
        """Run all 4 models on the same input window.

        Returns:
            {
                "heavy_rain": {"probability": 0.12, "alert": False, ...},
                "high_wind": {"probability": 0.87, "alert": True, ...},
                ...
            }
        """
        results = {}
        for event_type, model in self.models.items():
            try:
                results[event_type] = model.predict(window)
            except Exception as e:
                logger.error(f"Prediction failed for {event_type}: {e}")
                results[event_type] = {
                    "event": event_type,
                    "probability": None,
                    "alert": None,
                    "error": str(e),
                }
        return results
