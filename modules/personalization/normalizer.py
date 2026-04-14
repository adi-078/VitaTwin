"""
⚖️ Personalization Engine — Feature Normalizer
Scales all inputs relative to the user's own baseline profile.

The system NEVER compares the user to a population average.
Every feature is expressed as: (current - baseline_mean) / baseline_std
or as a ratio: current / baseline_mean.

This is the core personalisation innovation.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    FATIGUE_WINDOW_SEC,
    TARGET_FPS,
    SPEED_DROP_THRESHOLD,
    ROM_DROP_THRESHOLD,
    STABILITY_DROP_THRESHOLD,
)
from modules.vision.pose_extractor import PoseFrame
from modules.personalization.baseline import BaselineProfile


@dataclass
class NormalizedFeatures:
    """
    All fatigue-relevant features expressed relative to THIS user's baseline.
    Values > 1.0 mean 'worse than your personal norm'.
    """
    # ── Speed ─────────────────────────────────────────────────────────────────
    speed_ratio:       float = 1.0   # current / baseline  (<1.0 = slower)
    speed_z:           float = 0.0   # z-score vs baseline

    # ── Range of Motion per joint ─────────────────────────────────────────────
    rom_ratio:         Dict[str, float] = field(default_factory=dict)
    rom_z:             Dict[str, float] = field(default_factory=dict)
    mean_rom_ratio:    float = 1.0   # average across key joints

    # ── Stability ─────────────────────────────────────────────────────────────
    stability_ratio:   float = 1.0   # current / baseline  (>1.0 = more jitter)
    stability_z:       float = 0.0

    # ── Posture deviation ─────────────────────────────────────────────────────
    posture_deviation: Dict[str, float] = field(default_factory=dict)  # z-scores
    mean_posture_dev:  float = 0.0

    # ── Composite raw signal (for LSTM input) ─────────────────────────────────
    feature_vector:    np.ndarray = field(default_factory=lambda: np.zeros(12))

    # ── Flags ─────────────────────────────────────────────────────────────────
    speed_fatigued:     bool = False
    rom_fatigued:       bool = False
    stability_fatigued: bool = False


class FeatureNormalizer:
    """
    Maintains a rolling window of recent pose frames and produces
    NormalizedFeatures against the stored BaselineProfile.

    Usage:
        norm = FeatureNormalizer(baseline_profile)
        norm.update(pose_frame)
        features = norm.get_features()
    """

    KEY_JOINTS = [
        "left_knee", "right_knee",
        "left_hip",  "right_hip",
        "left_elbow", "right_elbow",
        "trunk_lean",
    ]

    def __init__(self, baseline: BaselineProfile):
        self._bl = baseline
        window_len = int(FATIGUE_WINDOW_SEC * TARGET_FPS)
        self._speed_buf:     deque = deque(maxlen=window_len)
        self._stability_buf: deque = deque(maxlen=window_len)
        self._angle_bufs:    Dict[str, deque] = {
            j: deque(maxlen=window_len) for j in self.KEY_JOINTS
        }

    def update_baseline(self, baseline: BaselineProfile) -> None:
        self._bl = baseline

    def update(self, frame: Optional[PoseFrame]) -> None:
        """Push one new PoseFrame into the rolling window."""
        if frame is None or not frame.is_valid:
            return
        self._speed_buf.append(frame.movement_speed)
        self._stability_buf.append(frame.joint_stability)
        for j in self.KEY_JOINTS:
            if j in frame.joint_angles:
                self._angle_bufs[j].append(frame.joint_angles[j])

    def get_features(self) -> NormalizedFeatures:
        """
        Compute NormalizedFeatures from the current rolling window.
        Returns a default (un-fatigued) feature set if insufficient data.
        """
        if len(self._speed_buf) < 5:
            return NormalizedFeatures()

        # ── Speed ────────────────────────────────────────────────────────────
        curr_speed  = float(np.mean(self._speed_buf))
        speed_ratio = curr_speed / (self._bl.speed_mean + 1e-8)
        speed_z     = (curr_speed - self._bl.speed_mean) / (self._bl.speed_std + 1e-8)

        # ── Stability ────────────────────────────────────────────────────────
        curr_stab     = float(np.mean(self._stability_buf))
        stab_ratio    = curr_stab / (self._bl.stability_mean + 1e-8)
        stab_z        = (curr_stab - self._bl.stability_mean) / (self._bl.stability_std + 1e-8)

        # ── ROM & Posture per joint ───────────────────────────────────────────
        rom_ratio, rom_z = {}, {}
        posture_dev = {}

        for j in self.KEY_JOINTS:
            buf = self._angle_bufs[j]
            if len(buf) < 5:
                rom_ratio[j] = 1.0
                rom_z[j] = 0.0
                posture_dev[j] = 0.0
                continue

            arr = np.array(buf)
            curr_rom  = float(np.percentile(arr, 95) - np.percentile(arr, 5))
            bl_rom    = self._bl.rom_mean.get(j, curr_rom) + 1e-3
            rom_ratio[j] = curr_rom / bl_rom
            rom_z[j]     = (curr_rom - bl_rom) / (self._bl.rom_std.get(j, 1.0) + 1e-3)

            curr_angle = float(np.mean(arr))
            bl_angle   = self._bl.angle_mean.get(j, curr_angle)
            bl_astd    = self._bl.angle_std.get(j, 5.0)
            posture_dev[j] = abs(curr_angle - bl_angle) / (bl_astd + 1e-3)

        mean_rom_ratio    = float(np.mean(list(rom_ratio.values())))
        mean_posture_dev  = float(np.mean(list(posture_dev.values())))

        # ── 12-D feature vector for LSTM ─────────────────────────────────────
        key_rom_ratios = [rom_ratio.get(j, 1.0) for j in self.KEY_JOINTS[:4]]
        key_posture    = [posture_dev.get(j, 0.0) for j in self.KEY_JOINTS[:4]]
        fv = np.array([
            speed_ratio, speed_z,
            stab_ratio,  stab_z,
            mean_rom_ratio, mean_posture_dev,
            *key_rom_ratios,   # 4 values
        ], dtype=np.float32)   # total = 12
        fv = np.clip(fv, -5.0, 5.0)

        return NormalizedFeatures(
            speed_ratio=speed_ratio,
            speed_z=speed_z,
            rom_ratio=rom_ratio,
            rom_z=rom_z,
            mean_rom_ratio=mean_rom_ratio,
            stability_ratio=stab_ratio,
            stability_z=stab_z,
            posture_deviation=posture_dev,
            mean_posture_dev=mean_posture_dev,
            feature_vector=fv,
            speed_fatigued=(speed_ratio < SPEED_DROP_THRESHOLD),
            rom_fatigued=(mean_rom_ratio < ROM_DROP_THRESHOLD),
            stability_fatigued=(stab_ratio > STABILITY_DROP_THRESHOLD),
        )