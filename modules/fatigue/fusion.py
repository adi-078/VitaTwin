"""
🔗 Fatigue Fusion Module
Combines physical and cognitive fatigue into a single Total Fatigue Score.

Total Fatigue = w_phys × Physical + w_cog × Cognitive

The weighting adapts slightly based on which modality has more signal:
if the face detector is unavailable (poor lighting / occlusion),
the system falls back to physical-only mode with 100 % weight.

Also records session fatigue history for LSTM training and validation.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    PHYSICAL_FATIGUE_WEIGHT,
    COGNITIVE_FATIGUE_WEIGHT,
    FATIGUE_ALERT_THRESHOLD,
    FATIGUE_CRITICAL_THRESHOLD,
)
from modules.fatigue.physical import PhysicalFatigueScore
from modules.fatigue.cognitive import CognitiveFatigueScore


@dataclass
class TotalFatigueScore:
    """Master fatigue score + metadata for display and coaching."""
    timestamp:    float

    total:        float   # 0-100  (THE number)
    physical:     float   # 0-100  physical sub-score
    cognitive:    float   # 0-100  cognitive sub-score

    # Decomposed drivers (for explainable dashboard)
    speed_contribution:       float = 0.0
    rom_contribution:         float = 0.0
    stability_contribution:   float = 0.0
    posture_contribution:     float = 0.0
    biomech_contribution:     float = 0.0
    eye_contribution:         float = 0.0
    blink_contribution:       float = 0.0

    # State flags
    alert:    bool = False   # score ≥ FATIGUE_ALERT_THRESHOLD
    critical: bool = False   # score ≥ FATIGUE_CRITICAL_THRESHOLD
    cognitive_available: bool = True


class FatigueAggregator:
    """
    Fuses physical + cognitive fatigue into a TotalFatigueScore
    and maintains a rolling time-series for LSTM input.

    Usage:
        agg = FatigueAggregator()
        total = agg.update(physical_score, cognitive_score)
        series = agg.get_series()        # for LSTM
    """

    def __init__(self):
        self._series:    deque = deque(maxlen=9000)   # ~30 min at 5 Hz
        self._ts_series: deque = deque(maxlen=9000)

    def update(self, phys: PhysicalFatigueScore,
               cog:  Optional[CognitiveFatigueScore],
               timestamp: Optional[float] = None) -> TotalFatigueScore:

        ts = timestamp or time.time()
        cog_available = (cog is not None and cog.total > 0)

        if cog_available:
            w_p = PHYSICAL_FATIGUE_WEIGHT
            w_c = COGNITIVE_FATIGUE_WEIGHT
        else:
            w_p, w_c = 1.0, 0.0

        total = float(np.clip(w_p * phys.total + w_c * (cog.total if cog else 0), 0, 100))

        self._series.append(total)
        self._ts_series.append(ts)

        return TotalFatigueScore(
            timestamp=ts,
            total=round(total, 1),
            physical=phys.total,
            cognitive=cog.total if cog else 0.0,
            speed_contribution       = round(phys.speed_score     * w_p * PHYSICAL_FATIGUE_WEIGHT / 100, 2),
            rom_contribution         = round(phys.rom_score        * w_p * PHYSICAL_FATIGUE_WEIGHT / 100, 2),
            stability_contribution   = round(phys.stability_score  * w_p * PHYSICAL_FATIGUE_WEIGHT / 100, 2),
            posture_contribution     = round(phys.posture_score     * w_p * PHYSICAL_FATIGUE_WEIGHT / 100, 2),
            biomech_contribution     = round(phys.biomech_score     * w_p * PHYSICAL_FATIGUE_WEIGHT / 100, 2),
            eye_contribution         = round((cog.ear_score    if cog else 0) * w_c * COGNITIVE_FATIGUE_WEIGHT / 100, 2),
            blink_contribution       = round((cog.blink_score  if cog else 0) * w_c * COGNITIVE_FATIGUE_WEIGHT / 100, 2),
            alert=    (total >= FATIGUE_ALERT_THRESHOLD),
            critical= (total >= FATIGUE_CRITICAL_THRESHOLD),
            cognitive_available=cog_available,
        )

    def get_series(self) -> np.ndarray:
        """Returns the full fatigue time-series as a 1-D float32 array."""
        return np.array(self._series, dtype=np.float32)

    def get_timestamps(self) -> np.ndarray:
        return np.array(self._ts_series, dtype=np.float64)

    def downsample_to_60(self) -> List[float]:
        """Return session fatigue curve downsampled to 60 points (for storage)."""
        arr = self.get_series()
        if len(arr) < 2:
            return [0.0] * 60
        indices = np.linspace(0, len(arr) - 1, 60).astype(int)
        return [float(arr[i]) for i in indices]

    def reset(self):
        self._series.clear()
        self._ts_series.clear()

    @property
    def current(self) -> float:
        return float(self._series[-1]) if self._series else 0.0

    @property
    def peak(self) -> float:
        return float(np.max(self._series)) if self._series else 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self._series)) if self._series else 0.0