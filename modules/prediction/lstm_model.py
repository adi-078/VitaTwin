"""
🔮 Predictive AI Layer — LSTM Fatigue Forecaster (FIXED VERSION)
"""

import numpy as np
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import (
    LSTM_INPUT_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, LSTM_SEQUENCE_LEN, LSTM_PREDICTION_STEPS,
    LSTM_MODEL_PATH, LSTM_MIN_SESSIONS,
)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not found. Using fallback prediction.")


# ── MODEL ─────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class FatigueLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=LSTM_INPUT_DIM,
                hidden_size=LSTM_HIDDEN_DIM,
                num_layers=LSTM_NUM_LAYERS,
                dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM // 2),
                nn.ReLU(),
                nn.Dropout(LSTM_DROPOUT),
                nn.Linear(LSTM_HIDDEN_DIM // 2, LSTM_PREDICTION_STEPS),
                nn.Sigmoid(),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last) * 100.0


# ── PREDICTOR ─────────────────────────────────────────────────────────────────

class FatiguePredictor:

    def __init__(self, session_count: int = 0):
        self._session_count = session_count
        self._feature_buf = []
        self._fatigue_buf = []
        self._model = None

        if TORCH_AVAILABLE:
            self._model = FatigueLSTM()
            if os.path.exists(LSTM_MODEL_PATH):
                try:
                    state = torch.load(LSTM_MODEL_PATH, map_location="cpu")
                    self._model.load_state_dict(state)
                    self._model.eval()
                    print(f"[Predictor] Loaded model")
                except Exception as e:
                    print(f"[Predictor] Load failed: {e}")

    # ── SAFE PUSH ────────────────────────────────────────────────────────────
    def push(self, feature_vector, current_fatigue):

        # Convert to numpy safely
        try:
            fv = np.asarray(feature_vector, dtype=np.float32).flatten()
        except Exception:
            return

        # Ensure correct size
        if len(fv) != LSTM_INPUT_DIM:
            return  # ❗ skip bad feature

        if not np.all(np.isfinite(fv)):
            return  # ❗ skip NaN/inf

        self._feature_buf.append(fv)
        self._fatigue_buf.append(float(current_fatigue))

        max_len = LSTM_SEQUENCE_LEN + 50
        if len(self._feature_buf) > max_len:
            self._feature_buf = self._feature_buf[-max_len:]
            self._fatigue_buf = self._fatigue_buf[-max_len:]

    # ── SAFE PREDICT ─────────────────────────────────────────────────────────
    def predict(self) -> "PredictionResult":

        if len(self._feature_buf) < LSTM_SEQUENCE_LEN:
            return self._extrapolate()

        if not TORCH_AVAILABLE or self._model is None:
            return self._extrapolate()

        # Filter valid features
        valid = [f for f in self._feature_buf if len(f) == LSTM_INPUT_DIM]

        if len(valid) < LSTM_SEQUENCE_LEN:
            return self._extrapolate()

        seq = valid[-LSTM_SEQUENCE_LEN:]

        try:
            fv = np.stack(seq).astype(np.float32)
        except Exception:
            return self._extrapolate()

        inp = torch.from_numpy(fv).unsqueeze(0)

        with torch.no_grad():
            forecast = self._model(inp).squeeze(0).numpy()

        return PredictionResult(
            forecast=forecast,
            current_fatigue=self._fatigue_buf[-1],
            method="lstm",
        )

    # ── FALLBACK ─────────────────────────────────────────────────────────────
    def _extrapolate(self) -> "PredictionResult":

        n = len(self._fatigue_buf)

        if n < 2:
            current = self._fatigue_buf[-1] if n > 0 else 0.0
            forecast = np.full(LSTM_PREDICTION_STEPS, current, dtype=np.float32)
            return PredictionResult(forecast, current, "cold_start")

        window = min(n, 30)
        y = np.array(self._fatigue_buf[-window:])
        x = np.arange(window)

        slope = float(np.polyfit(x, y, 1)[0])
        current = self._fatigue_buf[-1]

        future = current + slope * np.arange(1, LSTM_PREDICTION_STEPS + 1)
        forecast = np.clip(future, 0, 100).astype(np.float32)

        return PredictionResult(forecast, current, "linear")

    @property
    def ready(self):
        return len(self._feature_buf) >= LSTM_SEQUENCE_LEN


# ── RESULT ───────────────────────────────────────────────────────────────────

class PredictionResult:

    def __init__(self, forecast, current_fatigue, method):
        self.forecast = forecast.astype(np.float32)
        self.current_fatigue = current_fatigue
        self.method = method

        from config import FATIGUE_ALERT_THRESHOLD, FATIGUE_UPDATE_HZ

        self.seconds_to_alert: Optional[float] = None
        crosses = np.where(self.forecast >= FATIGUE_ALERT_THRESHOLD)[0]

        if len(crosses) > 0:
            self.seconds_to_alert = float(crosses[0]) / FATIGUE_UPDATE_HZ

    @property
    def horizon_sec(self):
        from config import LSTM_PREDICTION_STEPS, FATIGUE_UPDATE_HZ
        return LSTM_PREDICTION_STEPS / FATIGUE_UPDATE_HZ

    @property
    def peak_predicted(self):
        return float(np.max(self.forecast))

    @property
    def trend(self):
        if len(self.forecast) < 2:
            return "stable"

        slope = float(np.polyfit(np.arange(len(self.forecast)), self.forecast, 1)[0])

        if slope > 0.5:
            return "rising"
        if slope < -0.5:
            return "falling"
        return "stable"

    def __repr__(self):
        return f"PredictionResult(method={self.method}, peak={self.peak_predicted:.1f})"