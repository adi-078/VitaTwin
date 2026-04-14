"""
🧬 Personalization Engine — Baseline Calibration
Learns the user's natural movement profile during the first ~3 minutes
of an exercise session (or from stored historical data).

Captures:
  • Natural movement speed
  • Range of motion per joint
  • Joint stability (jitter baseline)
  • Trunk posture tendency
"""
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Optional, Deque
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    CALIBRATION_DURATION_SEC,
    CALIBRATION_MIN_FRAMES,
    CALIBRATION_WARMUP_SEC,
)
from modules.vision.pose_extractor import PoseFrame


@dataclass
class BaselineProfile:
    """
    Captured baseline statistics for one user.
    All values represent that user's 'fresh / unfatigued' norm.
    """
    # ── Speed ──────────────────────────────────────────────
    speed_mean:   float = 0.0
    speed_std:    float = 0.01

    # ── Range of Motion per joint (degrees) ─────────────────
    rom_mean:     Dict[str, float] = field(default_factory=dict)
    rom_std:      Dict[str, float] = field(default_factory=dict)

    # ── Stability (joint jitter) ─────────────────────────────
    stability_mean: float = 0.001
    stability_std:  float = 0.0005

    # ── Joint angles at 'neutral' ────────────────────────────
    angle_mean:   Dict[str, float] = field(default_factory=dict)
    angle_std:    Dict[str, float] = field(default_factory=dict)

    # ── Meta ────────────────────────────────────────────────
    n_frames:     int   = 0
    capture_time: float = field(default_factory=time.time)
    is_ready:     bool  = False   # True once CALIBRATION_MIN_FRAMES collected

    def summary(self) -> str:
        lines = [
            f"Baseline ready: {self.is_ready}  (frames: {self.n_frames})",
            f"  Speed   : {self.speed_mean:.4f} ± {self.speed_std:.4f}",
            f"  Stability: {self.stability_mean:.5f} ± {self.stability_std:.5f}",
        ]
        for joint, mean in sorted(self.angle_mean.items()):
            std = self.angle_std.get(joint, 0)
            lines.append(f"  {joint:<20}: {mean:.1f}° ± {std:.1f}°")
        return "\n".join(lines)


class BaselineCalibrator:
    """
    Collects pose frames during the calibration window and builds
    a BaselineProfile for the user.

    Usage:
        cal = BaselineCalibrator()
        while not cal.is_complete:
            frame = pose_extractor.process_frame(img)
            cal.update(frame)
        profile = cal.get_profile()
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._speeds:     Deque[float] = deque()
        self._stabilities: Deque[float] = deque()
        self._angles:     Dict[str, Deque[float]] = {}
        self._n:          int = 0

    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_complete(self) -> bool:
        return (self._n >= CALIBRATION_MIN_FRAMES and
                self._elapsed >= CALIBRATION_DURATION_SEC)

    @property
    def progress(self) -> float:
        """Calibration progress 0.0 → 1.0"""
        if self._start_time is None:
            return 0.0
        time_prog  = min(self._elapsed / CALIBRATION_DURATION_SEC, 1.0)
        frame_prog = min(self._n / max(CALIBRATION_MIN_FRAMES, 1), 1.0)
        return min(time_prog, frame_prog)

    @property
    def frames_collected(self) -> int:
        return self._n

    def update(self, pose_frame: Optional[PoseFrame]) -> None:
        """Feed a new PoseFrame into the calibration buffer."""
        if pose_frame is None or not pose_frame.is_valid:
            return

        if self._start_time is None:
            self._start_time = pose_frame.timestamp

        # Discard warmup period
        if (pose_frame.timestamp - self._start_time) < CALIBRATION_WARMUP_SEC:
            return

        if self.is_complete:
            return

        self._speeds.append(pose_frame.movement_speed)
        self._stabilities.append(pose_frame.joint_stability)

        for joint, angle in pose_frame.joint_angles.items():
            if joint not in self._angles:
                self._angles[joint] = deque()
            self._angles[joint].append(angle)

        self._n += 1

    def get_profile(self) -> BaselineProfile:
        """
        Compute and return the BaselineProfile from collected data.
        Falls back to sensible population defaults if insufficient data.
        """
        if self._n == 0:
            return BaselineProfile()

        speeds = np.array(self._speeds)
        stabs  = np.array(self._stabilities)

        rom_mean, rom_std = {}, {}
        angle_mean, angle_std = {}, {}

        for joint, vals in self._angles.items():
            arr = np.array(vals)
            angle_mean[joint] = float(np.mean(arr))
            angle_std[joint]  = float(np.std(arr)) + 0.5   # +0.5° floor to avoid /0

            # ROM = peak-to-peak angular excursion over the window
            rom_mean[joint] = float(np.percentile(arr, 95) - np.percentile(arr, 5))
            rom_std[joint]  = float(np.std([
                arr[i:i+30].max() - arr[i:i+30].min()
                for i in range(0, max(len(arr)-30, 1), 10)
            ]) + 0.5)

        return BaselineProfile(
            speed_mean=float(np.mean(speeds)),
            speed_std=float(np.std(speeds)) + 1e-5,
            rom_mean=rom_mean,
            rom_std=rom_std,
            stability_mean=float(np.mean(stabs)),
            stability_std=float(np.std(stabs)) + 1e-6,
            angle_mean=angle_mean,
            angle_std=angle_std,
            n_frames=self._n,
            capture_time=time.time(),
            is_ready=(self._n >= CALIBRATION_MIN_FRAMES),
        )

    def reset(self) -> None:
        self._start_time = None
        self._speeds.clear()
        self._stabilities.clear()
        self._angles.clear()
        self._n = 0

    @property
    def _elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time