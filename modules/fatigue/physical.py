"""
📊 Fatigue Estimation — Physical Fatigue Module
Combines kinematic features + biomechanical stress into a 0-100
physical fatigue score, personalised to the user's baseline.

Sub-scores (all 0-100):
  • Speed decay      — how much slower vs YOUR baseline
  • ROM decay        — how much your range of motion has shrunk
  • Stability decay  — how much joint instability has increased
  • Posture shift    — deviation from YOUR neutral posture
  • Biomechanical    — joint stress accumulation (from digital twin)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import FATIGUE_SMOOTHING_ALPHA
from modules.personalization.normalizer import NormalizedFeatures
from modules.digital_twin.biomechanics import BiomechanicsFrame


@dataclass
class PhysicalFatigueScore:
    """Decomposed physical fatigue, each component 0-100."""
    total:            float = 0.0   # weighted composite

    speed_score:      float = 0.0   # contribution from speed drop
    rom_score:        float = 0.0   # contribution from ROM reduction
    stability_score:  float = 0.0   # contribution from jitter increase
    posture_score:    float = 0.0   # contribution from posture deviation
    biomech_score:    float = 0.0   # contribution from joint stress

    # Human-readable reasons (for coach / explainer)
    contributing_factors: Dict[str, float] = field(default_factory=dict)


class PhysicalFatigueEstimator:
    """
    Estimates physical fatigue in real time from normalised features
    and biomechanical stress.

    The estimator applies a slow EMA so the score can't spike on a
    single anomalous frame — but it does track genuine trends.

    Usage:
        est = PhysicalFatigueEstimator()
        score = est.update(norm_features, biomech_frame)
    """

    # Component weights (must sum to 1.0)
    WEIGHTS = {
        "speed":     0.30,
        "rom":       0.25,
        "stability": 0.20,
        "posture":   0.15,
        "biomech":   0.10,
    }

    def __init__(self):
        self._ema: float = 0.0
        self._history: deque = deque(maxlen=3000)   # ~10 min at 5 Hz

    # ──────────────────────────────────────────────────────────────────────────

    def update(self, features: NormalizedFeatures,
               biomech: Optional[BiomechanicsFrame] = None) -> PhysicalFatigueScore:
        """
        Compute and return the latest physical fatigue score.
        Also updates the internal EMA.
        """
        # ── Speed component ──────────────────────────────────────────────────
        # Map speed_ratio [0..1] → [100..0] with a sigmoid-like curve
        # speed_ratio = 1.0 → fresh; 0.75 → alert threshold; 0.5 → severe
        spd_r   = float(np.clip(features.speed_ratio, 0.1, 1.5))
        speed_s = float(np.clip((1.0 - spd_r) * 125.0, 0.0, 100.0))

        # ── ROM component ────────────────────────────────────────────────────
        rom_r  = float(np.clip(features.mean_rom_ratio, 0.1, 1.5))
        rom_s  = float(np.clip((1.0 - rom_r) * 125.0, 0.0, 100.0))

        # ── Stability component ──────────────────────────────────────────────
        stab_r = float(np.clip(features.stability_ratio, 0.1, 5.0))
        stab_s = float(np.clip((stab_r - 1.0) * 50.0, 0.0, 100.0))

        # ── Posture component ────────────────────────────────────────────────
        post_s = float(np.clip(features.mean_posture_dev * 25.0, 0.0, 100.0))

        # ── Biomechanical component ──────────────────────────────────────────
        biom_s = biomech.biomech_fatigue if biomech is not None else 0.0

        # ── Weighted composite ───────────────────────────────────────────────
        raw = (
            self.WEIGHTS["speed"]     * speed_s +
            self.WEIGHTS["rom"]       * rom_s +
            self.WEIGHTS["stability"] * stab_s +
            self.WEIGHTS["posture"]   * post_s +
            self.WEIGHTS["biomech"]   * biom_s
        )

        # EMA smoothing — prevents single-frame spikes
        self._ema = FATIGUE_SMOOTHING_ALPHA * raw + (1 - FATIGUE_SMOOTHING_ALPHA) * self._ema
        self._history.append(self._ema)

        contributing = {
            "speed_drop":        round(speed_s, 1),
            "rom_reduction":     round(rom_s, 1),
            "instability":       round(stab_s, 1),
            "posture_deviation": round(post_s, 1),
            "joint_stress":      round(biom_s, 1),
        }

        return PhysicalFatigueScore(
            total=round(float(np.clip(self._ema, 0.0, 100.0)), 1),
            speed_score=round(speed_s, 1),
            rom_score=round(rom_s, 1),
            stability_score=round(stab_s, 1),
            posture_score=round(post_s, 1),
            biomech_score=round(biom_s, 1),
            contributing_factors=contributing,
        )

    def get_history(self) -> np.ndarray:
        return np.array(self._history, dtype=np.float32)

    def reset(self):
        self._ema = 0.0
        self._history.clear()