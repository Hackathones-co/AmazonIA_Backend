"""
ML Model definitions — mirrors the notebook's RecurrentClassifier exactly.
This file is used both for training (notebooks) and inference (API).
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class RecurrentClassifier(nn.Module):
    """Shared architecture for RNN/LSTM/GRU binary classifiers.

    Copied verbatim from the training notebooks to ensure checkpoint compatibility.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, rnn_type: str = "lstm"):
        super().__init__()
        self.rnn_type = rnn_type

        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        output, _ = self.rnn(x)
        last_hidden = output[:, -1, :]       # (batch, hidden_dim)
        logit = self.classifier(last_hidden)  # (batch, 1)
        return logit.squeeze(-1)              # (batch,)


class LoadedModel:
    """A loaded checkpoint ready for inference.

    Wraps the PyTorch model + all metadata exported by Section 12 of the notebooks:
    feature_cols, train_mean, train_std, threshold.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # ── Metadata from notebook export ─────────────────────────
        self.event_type: str = ckpt["event_type"]          # e.g. "heavy_rain"
        self.target: str = ckpt["target"]                  # e.g. "heavy_rain_3h"
        self.lookback: int = ckpt["lookback"]              # 96
        self.feature_cols: list[str] = ckpt["feature_cols"]
        self.threshold: float = ckpt["threshold"]

        # ── Normalization params (fitted on training data) ────────
        self.train_mean: dict = ckpt["train_mean"]    # {col: float}
        self.train_std: dict = ckpt["train_std"]      # {col: float}

        # ── Model ─────────────────────────────────────────────────
        self.model = RecurrentClassifier(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt.get("hidden_dim", 128),
            num_layers=ckpt.get("num_layers", 2),
            dropout=ckpt.get("dropout", 0.3),
            rnn_type=ckpt["rnn_type"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, window: np.ndarray) -> dict:
        """Run inference on a single window.

        Args:
            window: np.ndarray of shape (lookback, n_features) — raw (unnormalized) values
                    Column order must match self.feature_cols.

        Returns:
            dict with probability, binary prediction, and threshold.
        """
        # ── Normalize using training statistics ───────────────────
        mean = np.array([self.train_mean[c] for c in self.feature_cols], dtype=np.float32)
        std = np.array([self.train_std[c] for c in self.feature_cols], dtype=np.float32)
        std[std == 0] = 1.0  # avoid division by zero
        normalized = (window - mean) / std

        # ── Inference ─────────────────────────────────────────────
        x = torch.from_numpy(normalized).unsqueeze(0).to(self.device)  # (1, lookback, features)
        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()

        return {
            "event": self.event_type,
            "probability": round(prob, 4),
            "alert": prob >= self.threshold,
            "threshold": round(self.threshold, 4),
        }

    def predict_batch(self, windows: np.ndarray) -> list[dict]:
        """Run inference on multiple windows.

        Args:
            windows: np.ndarray of shape (batch, lookback, n_features)
        """
        mean = np.array([self.train_mean[c] for c in self.feature_cols], dtype=np.float32)
        std = np.array([self.train_std[c] for c in self.feature_cols], dtype=np.float32)
        std[std == 0] = 1.0
        normalized = (windows - mean) / std

        x = torch.from_numpy(normalized).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy()

        return [
            {
                "event": self.event_type,
                "probability": round(float(p), 4),
                "alert": float(p) >= self.threshold,
                "threshold": round(self.threshold, 4),
            }
            for p in probs
        ]
