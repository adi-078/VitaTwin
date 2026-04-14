"""
🧠 Cognitive Fatigue Module
Converts EAR, blink rate and drowsiness signals into a 0-100
cognitive fatigue score.

Design note:
  - Low EAR + sustained → drowsy / eye-closure fatigue
  - Abnormal blink rate (too high OR too low) → cognitive load / strain
  - Drowsy EMA from FaceExtractor captures sustained impairment
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    FATIGUE_SMOOTHING_ALPHA,
    EAR_DROWSY_THRESHOLD,
    NORMAL_BLINK_RATE, HIGH_BLINK_RATE, LOW_BLINK_RATE,
)
from modules.vision.face_extractor import FaceFrame


@dataclass
class CognitiveFatigueScore:
    """Decomposed cognitive fatigue, each component 0-100."""
    total:         float = 0.0

    ear_score:     float = 0.0   # from eye closure
    blink_score:   float = 0.0   # from blink-rate deviation
    drowsy_score:  float = 0.0   # from sustained low-EAR EMA

    # For explainability
    blink_rate_pm: float = 0.0
    mean_ear:      float = 0.3
    is_drowsy:     bool  = False


class CognitiveFatigueEstimator:
    """
    Estimates cognitive fatigue from face-tracking signals.

    Usage:
        cog = CognitiveFatigueEstimator()
        score = cog.update(face_frame)
    """

    WEIGHTS = {
        "ear":    0.45,
        "blink":  0.25,
        "drowsy": 0.30,
    }

    def __init__(self):
        self._ema:     float = 0.0
        self._history: deque = deque(maxlen=3000)

    def update(self, face_frame: Optional[FaceFrame]) -> CognitiveFatigueScore:
        if face_frame is None or not face_frame.is_valid:
            # Propagate previous EMA when face not detected
            self._history.append(self._ema)
            return CognitiveFatigueScore(total=round(self._ema, 1))

        # ── EAR component ─────────────────────────────────────────────────────
        ear = float(np.clip(face_frame.ear, 0.05, 0.45))
        # Linear: EAR at threshold → 30, EAR=0.10 → 100, EAR=0.35 → 0
        ear_s = float(np.clip(
            (EAR_DROWSY_THRESHOLD - ear) / (EAR_DROWSY_THRESHOLD - 0.10) * 100.0,
            0.0, 100.0
        ))

        # ── Blink-rate component ──────────────────────────────────────────────
        bpm = face_frame.blink_rate_pm
        if bpm < LOW_BLINK_RATE:
            blink_s = float(np.clip((LOW_BLINK_RATE - bpm) / LOW_BLINK_RATE * 80.0, 0.0, 80.0))
        elif bpm > HIGH_BLINK_RATE:
            blink_s = float(np.clip((bpm - HIGH_BLINK_RATE) / HIGH_BLINK_RATE * 80.0, 0.0, 80.0))
        else:
            blink_s = 0.0

        # ── Drowsiness EMA component ──────────────────────────────────────────
        drowsy_s = float(np.clip(face_frame.drowsy_score * 100.0, 0.0, 100.0))

        # ── Composite ────────────────────────────────────────────────────────
        raw = (
            self.WEIGHTS["ear"]    * ear_s +
            self.WEIGHTS["blink"]  * blink_s +
            self.WEIGHTS["drowsy"] * drowsy_s
        )
        self._ema = FATIGUE_SMOOTHING_ALPHA * raw + (1 - FATIGUE_SMOOTHING_ALPHA) * self._ema
        self._history.append(self._ema)

        return CognitiveFatigueScore(
            total=round(float(np.clip(self._ema, 0.0, 100.0)), 1),
            ear_score=round(ear_s, 1),
            blink_score=round(blink_s, 1),
            drowsy_score=round(drowsy_s, 1),
            blink_rate_pm=round(bpm, 1),
            mean_ear=round(ear, 3),
            is_drowsy=(face_frame.drowsy_score > 0.4),
        )

    def get_history(self) -> np.ndarray:
        return np.array(self._history, dtype=np.float32)

    def reset(self):
        self._ema = 0.0
        self._history.clear()